#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Experiment:
    eid: str
    tag: str
    config: Path
    work_dir: Path
    checkpoint: Path


EXPERIMENTS: Sequence[Tuple[str, str]] = (
    ("E0002", "E0002_lora-r16_fc1fc2_seed0"),
    ("E0004", "E0004_lora-r16_fc2"),
    ("E0003", "E0003_linear-probe"),
    ("E0019", "E0019_full-ft"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Grad-CAM overlays on the shared 500-image val subset, then package into zips."
    )
    parser.add_argument("--pkg-dir", required=True, help="Output package directory (zips/manifest/readme)")
    parser.add_argument("--export-root", required=True, help="Where to write per-experiment export outputs")
    parser.add_argument("--val-ann", required=True, help="Subset COCO JSON for val (500 images)")
    parser.add_argument(
        "--gpu-mem-md",
        default=None,
        help="Optional GPU/RAM stats markdown file to include in the package (e.g. artifacts/work_dirs/<...>/GPU_MEMORY.md)",
    )
    parser.add_argument("--env-name", default="sar_lora_dino", help="Conda env name (default: sar_lora_dino)")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="CUDA_VISIBLE_DEVICES to use (default: leave unchanged)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip export/zip if output zip already exists (resume mode).",
    )
    parser.add_argument(
        "--target-layer",
        default="backbone.backbone.stages_3",
        help="Grad-CAM target layer (default: backbone.backbone.stages_3)",
    )
    parser.add_argument("--topk", type=int, default=100, help="Grad target: sum(topk sigmoid cls scores)")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Overlay alpha in [0,1]")
    parser.add_argument("--colormap", default="JET", help="OpenCV colormap name (default: JET)")
    parser.add_argument("--max-retries", type=int, default=1, help="Max retries per export (default: 1)")
    return parser.parse_args()


def _md5_file(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _pick_best_checkpoint(work_dir: Path) -> Path:
    best = sorted(work_dir.glob("best_coco_*_epoch_*.pth"))
    if best:
        return best[0]
    best = sorted(work_dir.glob("best_*.pth"))
    if best:
        return best[0]
    last = work_dir / "last_checkpoint"
    if last.is_file():
        target = last.read_text(encoding="utf-8").strip()
        if target:
            ckpt = work_dir / target
            if ckpt.is_file():
                return ckpt
    epochs = list(work_dir.glob("epoch_*.pth"))

    def _epoch_num(p: Path) -> int:
        m = re.search(r"epoch_(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    epochs.sort(key=_epoch_num, reverse=True)
    if epochs:
        return epochs[0]
    raise FileNotFoundError(f"No checkpoint found in {work_dir}")


def _load_experiments_from_results_tsv(repo_root: Path) -> List[Experiment]:
    results_path = repo_root / "artifacts/experiments/experiment_results.tsv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing {results_path}")

    wanted = {eid for eid, _ in EXPERIMENTS}
    tag_by_eid = {eid: tag for eid, tag in EXPERIMENTS}

    rows: List[Dict[str, str]] = []
    with results_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("EID") not in wanted:
                continue
            if row.get("Split") != "val":
                continue
            if row.get("Source") != "ledger":
                continue
            rows.append(row)

    selected: Dict[str, Dict[str, str]] = {}
    for row in rows:
        eid = row["EID"]
        if eid not in selected:
            selected[eid] = row
            continue
        if selected[eid].get("Run") != "seed0" and row.get("Run") == "seed0":
            selected[eid] = row

    missing = wanted - set(selected)
    if missing:
        raise RuntimeError(
            "Missing experiments in artifacts/experiments/experiment_results.tsv: "
            f"{sorted(missing)}"
        )

    experiments: List[Experiment] = []
    for eid, _ in EXPERIMENTS:
        row = selected[eid]
        config = repo_root / row["Config"]
        metrics_json = repo_root / row["MetricsJSON"]
        work_dir = metrics_json.parent
        checkpoint = _pick_best_checkpoint(work_dir)
        experiments.append(
            Experiment(
                eid=eid,
                tag=tag_by_eid[eid],
                config=config,
                work_dir=work_dir,
                checkpoint=checkpoint,
            )
        )
    return experiments


def _pick_existing_zip(pkg_dir: Path, zip_prefix: str) -> Optional[Path]:
    candidates = [
        p
        for p in pkg_dir.glob(zip_prefix + "_*.zip")
        if p.is_file() and re.search(r"_\d{8}_\d{6}\.zip$", p.name)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def _zip_dir(repo_root: Path, src_dir: Path, zip_path: Path) -> Tuple[int, int]:
    files = sorted([p for p in src_dir.rglob("*") if p.is_file()], key=lambda p: p.as_posix())
    if not files:
        raise RuntimeError(f"No files to zip in {src_dir}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for fpath in files:
            arcname = fpath.resolve().relative_to(repo_root.resolve()).as_posix()
            zf.write(fpath, arcname)

    return len(files), zip_path.stat().st_size


def _run_export(
    *,
    repo_root: Path,
    env_name: str,
    experiment: Experiment,
    ann_file: Path,
    img_prefix: str,
    out_dir: Path,
    target_layer: str,
    topk: int,
    overlay_alpha: float,
    colormap: str,
    cuda_visible_devices: Optional[str],
    max_retries: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "export.log"

    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        str(repo_root / "visualization/mmdet_gradcam_export.py"),
        "--config",
        str(experiment.config),
        "--checkpoint",
        str(experiment.checkpoint),
        "--ann-file",
        str(ann_file.resolve()),
        "--img-prefix",
        img_prefix,
        "--out-dir",
        str(out_dir),
        "--device",
        "cuda",
        "--num-workers",
        "0",
        "--target-layer",
        target_layer,
        "--topk",
        str(int(topk)),
        "--overlay-alpha",
        str(float(overlay_alpha)),
        "--colormap",
        colormap,
    ]

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    for attempt in range(1, max_retries + 1):
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(
                f"\n===== [{time.strftime('%F %T')}] {experiment.tag} val attempt {attempt}/{max_retries}\n"
            )
            logf.write("CMD: " + " ".join(cmd) + "\n\n")
            logf.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        meta_path = out_dir / "meta.json"
        if proc.returncode == 0 and meta_path.is_file() and meta_path.stat().st_size > 0:
            return
        if attempt < max_retries:
            time.sleep(5)

    raise RuntimeError(f"Export failed: {experiment.tag} (see {log_path})")


def main() -> int:
    args = _parse_args()
    repo_root = Path.cwd()

    pkg_dir = Path(args.pkg_dir)
    export_root = Path(args.export_root)
    val_ann = Path(args.val_ann).resolve()

    pkg_dir.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)

    # include subset json inside the package for traceability
    subsets_dir = pkg_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(val_ann, subsets_dir / val_ann.name)

    if args.gpu_mem_md is not None:
        gpu_mem_src = (
            (repo_root / args.gpu_mem_md).resolve()
            if not Path(args.gpu_mem_md).is_absolute()
            else Path(args.gpu_mem_md).resolve()
        )
        if not gpu_mem_src.is_file():
            raise FileNotFoundError(f"GPU/RAM stats file not found: {gpu_mem_src}")
        shutil.copy2(gpu_mem_src, pkg_dir / "GPU_MEMORY.md")

    experiments = _load_experiments_from_results_tsv(repo_root)

    rows: List[Dict[str, str]] = []
    for exp in experiments:
        split = "val"
        ann_file = val_ann
        img_prefix = "JPEGImages/val/"

        export_ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = export_root / exp.tag / split / export_ts

        zip_prefix = f"GradCAM_SAMPLE500_{exp.tag}_{split}"
        existing_zip = _pick_existing_zip(pkg_dir, zip_prefix) if args.skip_existing else None
        if existing_zip is not None:
            zip_path = existing_zip
            zip_name = zip_path.name
            zip_md5 = _md5_file(zip_path)
            zip_bytes = zip_path.stat().st_size
            print(f"[SKIP] reuse {zip_name} ({zip_bytes/1024/1024:.1f} MiB)")
            # best-effort: count files in zip
            with zipfile.ZipFile(zip_path, "r") as zf:
                files_in_zip = [n for n in zf.namelist() if n and not n.endswith("/")]
            files_in_zip.sort()
            list_name = zip_name.replace(".zip", ".txt")
            list_path = pkg_dir / list_name
            if not list_path.is_file():
                with list_path.open("w", encoding="utf-8") as f:
                    for n in files_in_zip:
                        f.write(n + "\n")
            overlay_count = sum(1 for n in files_in_zip if n.lower().endswith((".jpg", ".jpeg", ".png")))
            export_dir_rel = ""
            overlay_dir_rel = ""
            export_ts = ""
        else:
            print(f"[RUN] {exp.tag} {split}")
            _run_export(
                repo_root=repo_root,
                env_name=args.env_name,
                experiment=exp,
                ann_file=ann_file,
                img_prefix=img_prefix,
                out_dir=out_dir,
                target_layer=args.target_layer,
                topk=args.topk,
                overlay_alpha=args.overlay_alpha,
                colormap=args.colormap,
                cuda_visible_devices=args.cuda_visible_devices,
                max_retries=args.max_retries,
            )

            overlay_dir = out_dir / "overlay"
            if not overlay_dir.is_dir():
                raise RuntimeError(f"Missing overlay dir: {overlay_dir}")

            overlay_images = sorted([p for p in overlay_dir.iterdir() if p.is_file()], key=lambda p: p.name)
            overlay_count = len(overlay_images)
            if overlay_count == 0:
                raise RuntimeError(f"No overlay images in {overlay_dir}")

            zip_name = f"{zip_prefix}_{export_ts}.zip"
            zip_path = pkg_dir / zip_name
            list_name = f"{zip_prefix}_{export_ts}.txt"
            list_path = pkg_dir / list_name

            with list_path.open("w", encoding="utf-8") as f:
                for img in overlay_images:
                    arcname = img.resolve().relative_to(repo_root.resolve()).as_posix()
                    f.write(arcname + "\n")

            _file_count, zip_bytes = _zip_dir(repo_root, out_dir, zip_path)
            zip_md5 = _md5_file(zip_path)
            zip_bytes = zip_path.stat().st_size
            export_dir_rel = out_dir.resolve().relative_to(repo_root.resolve()).as_posix()
            overlay_dir_rel = overlay_dir.resolve().relative_to(repo_root.resolve()).as_posix()
            print(f"[OK] {zip_name} ({overlay_count} overlays, {zip_bytes/1024/1024:.1f} MiB)")

        rows.append(
            {
                "eid": exp.eid,
                "tag": exp.tag,
                "split": split,
                "subset_ann": str((subsets_dir / val_ann.name).relative_to(pkg_dir).as_posix()),
                "config": exp.config.as_posix(),
                "checkpoint": exp.checkpoint.as_posix(),
                "target_layer": args.target_layer,
                "topk": str(args.topk),
                "overlay_alpha": str(args.overlay_alpha),
                "colormap": args.colormap,
                "export_dir": export_dir_rel,
                "export_timestamp": export_ts,
                "overlay_dir": overlay_dir_rel,
                "overlays": str(overlay_count),
                "zip_name": zip_name,
                "zip_md5": zip_md5,
                "zip_bytes": str(zip_bytes),
                "list_file": list_name,
            }
        )

    manifest_path = pkg_dir / "manifest.tsv"
    cols = [
        "eid",
        "tag",
        "split",
        "subset_ann",
        "config",
        "checkpoint",
        "target_layer",
        "topk",
        "overlay_alpha",
        "colormap",
        "export_dir",
        "export_timestamp",
        "overlay_dir",
        "overlays",
        "zip_name",
        "zip_md5",
        "zip_bytes",
        "list_file",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    readme_path = pkg_dir / "README.md"
    if not readme_path.is_file():
        readme_path.write_text(
            "# Grad-CAM 可视化抽样包（val 500 张，同一批图片便于对比）\n\n"
            "本目录用于导出并打包 Dinov3 系列（LoRA/No-LoRA）的 Grad-CAM-like 热力图叠加结果。\n"
            "所有方法使用同一份 `subsets/val_500_seed20260111.json`（500 张）以保证逐图可比。\n\n"
            "包含内容：\n"
            "- `GradCAM_SAMPLE500_*.zip`：每个方法一个 ZIP，内含 `overlay/*.jpg` + `meta.json` + `export.log`。\n"
            "- `GradCAM_SAMPLE500_*.txt`：ZIP 内文件清单（repo-relative）。\n"
            "- `manifest.tsv`：每个 ZIP 的 config/ckpt/参数/MD5/大小等。\n"
            "- `GPU_MEMORY.md`：训练/推理显存 + RSS（可选；仅当传入 `--gpu-mem-md` 时包含）。\n"
            "- `md5sum.txt`：本目录文件 MD5 校验。\n\n"
            "生成方式（示例）：\n"
            "```bash\n"
            "python visualization/export_gradcam_sample500_bundle.py \\\n"
            f"  --pkg-dir {pkg_dir.as_posix()} \\\n"
            f"  --export-root {export_root.as_posix()} \\\n"
            f"  --val-ann {val_ann.as_posix()} \\\n"
            "  --env-name sar_lora_dino \\\n"
            "  --cuda-visible-devices 9\n"
            "```\n",
            encoding="utf-8",
        )

    md5sum_path = pkg_dir / "md5sum.txt"
    all_files = sorted(
        [p for p in pkg_dir.rglob("*") if p.is_file() and p.name != "md5sum.txt"], key=lambda p: p.as_posix()
    )
    with md5sum_path.open("w", encoding="utf-8") as f:
        for p in all_files:
            rel = p.relative_to(pkg_dir).as_posix()
            f.write(f"{_md5_file(p)}  {rel}\n")

    print(f"[DONE] wrote {manifest_path}")
    print(f"[DONE] wrote {md5sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
