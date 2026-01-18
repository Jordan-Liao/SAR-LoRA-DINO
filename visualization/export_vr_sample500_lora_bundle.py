#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Experiment:
    eid: str
    tag: str
    config: Path
    work_dir: Path
    checkpoint: Path


LORA_EXPERIMENTS: Sequence[Tuple[str, str]] = (
    ("E0002", "E0002_lora-r16_fc1fc2_seed0"),
    ("E0004", "E0004_lora-r16_fc2"),
    ("E0005", "E0005_lora-r4_fc1fc2"),
    ("E0006", "E0006_lora-r8_fc1fc2"),
    ("E0007", "E0007_lora-r16_fc1fc2_unfreeze-stage3"),
    ("E0012", "E0012_lora-r4_fc2"),
    ("E0013", "E0013_lora-r8_fc2"),
    ("E0014", "E0014_lora-r4_fc1fc2_unfreeze-stage3"),
    ("E0015", "E0015_lora-r8_fc1fc2_unfreeze-stage3"),
    ("E0016", "E0016_lora-r16_fc2_unfreeze-stage3"),
    ("E0017", "E0017_lora-r4_fc2_unfreeze-stage3"),
    ("E0018", "E0018_lora-r8_fc2_unfreeze-stage3"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export VR visualizations on a shared 500-image subset for Dinov3 LoRA experiments, then package into zips."
    )
    parser.add_argument("--pkg-dir", required=True, help="Output package directory (zips/manifest/readme)")
    parser.add_argument("--export-root", required=True, help="Where to write per-experiment export outputs")
    parser.add_argument("--val-ann", required=True, help="Subset COCO JSON for val")
    parser.add_argument("--test-ann", required=True, help="Subset COCO JSON for test")
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
        "--max-retries",
        type=int,
        default=1,
        help="Max retries for a single export command (default: 1).",
    )
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
            ckpt = (work_dir / target)
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

    wanted = {eid for eid, _ in LORA_EXPERIMENTS}
    tag_by_eid = {eid: tag for eid, tag in LORA_EXPERIMENTS}

    rows: List[Dict[str, str]] = []
    with results_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("EID") not in wanted:
                continue
            if row.get("Split") != "val":
                continue
            rows.append(row)

    # pick representative row per EID (prefer seed0 for E0002)
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
            "Missing LoRA experiments in artifacts/experiments/experiment_results.tsv: "
            f"{sorted(missing)}"
        )

    experiments: List[Experiment] = []
    for eid, _ in LORA_EXPERIMENTS:
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


def _find_latest_timestamp_dir(split_out: Path) -> Path:
    # MMEngine default is like YYYYMMDD_HHMMSS
    candidates = [p for p in split_out.iterdir() if p.is_dir() and re.match(r"^\d{8}_\d{6}$", p.name)]
    if not candidates:
        raise RuntimeError(f"No timestamp dir found under {split_out}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


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


def _parse_timestamp_from_zip_name(zip_path: Path) -> str:
    m = re.search(r"_(\d{8}_\d{6})\.zip$", zip_path.name)
    if not m:
        raise ValueError(f"Cannot parse timestamp from zip name: {zip_path.name}")
    return m.group(1)


def _zip_vis_dir(repo_root: Path, vis_dir: Path, zip_path: Path) -> Tuple[int, int]:
    images = sorted([p for p in vis_dir.iterdir() if p.is_file()], key=lambda p: p.name)
    if not images:
        raise RuntimeError(f"No files to zip in {vis_dir}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for img in images:
            arcname = img.resolve().relative_to(repo_root.resolve()).as_posix()
            zf.write(img, arcname)

    return len(images), zip_path.stat().st_size


def _copy_pkg_metrics(metrics_json: Path, pkg_dir: Path, tag: str, split: str) -> Path:
    if not metrics_json.is_file():
        raise FileNotFoundError(f"Missing metrics.json: {metrics_json}")
    dst = pkg_dir / "metrics" / tag / split / "metrics.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_json, dst)
    return dst


def _run_export(
    *,
    repo_root: Path,
    env_name: str,
    split: str,
    ann_file: Path,
    img_prefix: str,
    experiment: Experiment,
    split_out: Path,
    metrics_json: Path,
    show_dir: str,
    cuda_visible_devices: Optional[str],
    max_retries: int,
) -> None:
    split_out.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    log_path = split_out / "export.log"

    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        str(repo_root / "visualization/mmdet_test_export.py"),
        "--config",
        str(experiment.config),
        "--checkpoint",
        str(experiment.checkpoint),
        "--work-dir",
        str(split_out),
        "--out-metrics",
        str(metrics_json),
        "--show-dir",
        show_dir,
        "--cfg-options",
        f"test_dataloader.dataset.ann_file={ann_file}",
        f"test_dataloader.dataset.data_prefix.img={img_prefix}",
        f"test_evaluator.ann_file={ann_file}",
        "model.test_cfg.score_thr=0.0",
    ]

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    for attempt in range(1, max_retries + 1):
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"\n===== [{time.strftime('%F %T')}] {experiment.tag} {split} attempt {attempt}/{max_retries}\n")
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
        if proc.returncode == 0 and metrics_json.is_file() and metrics_json.stat().st_size > 0:
            return
        if attempt < max_retries:
            time.sleep(5)

    raise RuntimeError(f"Export failed: {experiment.tag} split={split} (see {log_path})")


def main() -> int:
    args = _parse_args()
    repo_root = Path.cwd()

    pkg_dir = Path(args.pkg_dir)
    export_root = Path(args.export_root)
    val_ann = Path(args.val_ann).resolve()
    test_ann = Path(args.test_ann).resolve()

    pkg_dir.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)

    experiments = _load_experiments_from_results_tsv(repo_root)

    rows: List[Dict[str, str]] = []
    for exp in experiments:
        for split in ("val", "test"):
            ann_file = val_ann if split == "val" else test_ann
            img_prefix = "JPEGImages/val/" if split == "val" else "JPEGImages/test/"

            split_out = export_root / exp.tag / split
            metrics_json = split_out / "metrics.json"

            # zip name includes export timestamp for traceability
            zip_prefix = f"VR_SAMPLE500_{exp.tag}_{split}"

            existing_zip = _pick_existing_zip(pkg_dir, zip_prefix) if args.skip_existing else None
            if existing_zip is not None:
                zip_path = existing_zip
                ts_name = _parse_timestamp_from_zip_name(existing_zip)
                ts_dir = split_out / ts_name
                vis_dir = ts_dir / "vis"
                zip_name = zip_path.name
                list_name = f"{zip_prefix}_{ts_name}.txt"
                list_path = pkg_dir / list_name
                if not list_path.is_file():
                    # Fallback: if export vis dir is missing, build list from zip entries.
                    if vis_dir.is_dir():
                        images = sorted([p for p in vis_dir.iterdir() if p.is_file()], key=lambda p: p.name)
                        with list_path.open("w", encoding="utf-8") as f:
                            for img in images:
                                arcname = img.resolve().relative_to(repo_root.resolve()).as_posix()
                                f.write(arcname + "\n")
                    else:
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            names = [n for n in zf.namelist() if not n.endswith("/") and n]
                        names.sort()
                        with list_path.open("w", encoding="utf-8") as f:
                            for n in names:
                                f.write(n + "\n")

                with list_path.open("r", encoding="utf-8") as f:
                    img_count = sum(1 for _ in f)
                zip_bytes = zip_path.stat().st_size
                zip_md5 = _md5_file(zip_path)
                print(f"[SKIP] reuse {zip_name} ({img_count} images, {zip_bytes/1024/1024:.1f} MiB)")
            else:
                print(f"[RUN] {exp.tag} split={split}")
                _run_export(
                    repo_root=repo_root,
                    env_name=args.env_name,
                    split=split,
                    ann_file=ann_file,
                    img_prefix=img_prefix,
                    experiment=exp,
                    split_out=split_out,
                    metrics_json=metrics_json,
                    show_dir="vis",
                    cuda_visible_devices=args.cuda_visible_devices,
                    max_retries=args.max_retries,
                )

                ts_dir = _find_latest_timestamp_dir(split_out)
                vis_dir = ts_dir / "vis"
                if not vis_dir.is_dir():
                    raise RuntimeError(f"Missing vis dir: {vis_dir}")

                zip_name = f"{zip_prefix}_{ts_dir.name}.zip"
                zip_path = pkg_dir / zip_name
                list_name = f"{zip_prefix}_{ts_dir.name}.txt"
                list_path = pkg_dir / list_name

                images = sorted([p for p in vis_dir.iterdir() if p.is_file()], key=lambda p: p.name)
                with list_path.open("w", encoding="utf-8") as f:
                    for img in images:
                        arcname = img.resolve().relative_to(repo_root.resolve()).as_posix()
                        f.write(arcname + "\n")

                img_count, zip_bytes = _zip_vis_dir(repo_root, vis_dir, zip_path)
                zip_md5 = _md5_file(zip_path)
                print(f"[OK] {zip_name} ({img_count} images, {zip_bytes/1024/1024:.1f} MiB)")

            pkg_metrics = _copy_pkg_metrics(metrics_json, pkg_dir, exp.tag, split)

            rows.append(
                {
                    "eid": exp.eid,
                    "tag": exp.tag,
                    "split": split,
                    "subset_ann": str(ann_file.relative_to(pkg_dir.resolve().parent.resolve()))
                    if str(ann_file).startswith(str(pkg_dir.resolve().parent.resolve()))
                    else str(ann_file),
                    "config": exp.config.as_posix(),
                    "checkpoint": exp.checkpoint.as_posix(),
                    "export_split_dir": split_out.resolve().relative_to(repo_root.resolve()).as_posix(),
                    "export_timestamp": ts_dir.name,
                    "vis_dir": vis_dir.resolve().relative_to(repo_root.resolve()).as_posix(),
                    "images": str(img_count),
                    "zip_name": zip_name,
                    "zip_md5": zip_md5,
                    "zip_bytes": str(zip_bytes),
                    "list_file": list_name,
                    "pkg_metrics_json": pkg_metrics.relative_to(pkg_dir).as_posix(),
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
        "export_split_dir",
        "export_timestamp",
        "vis_dir",
        "images",
        "zip_name",
        "zip_md5",
        "zip_bytes",
        "list_file",
        "pkg_metrics_json",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md5sum_path = pkg_dir / "md5sum.txt"
    all_files = sorted([p for p in pkg_dir.rglob("*") if p.is_file() and p.name != "md5sum.txt"], key=lambda p: p.as_posix())
    with md5sum_path.open("w", encoding="utf-8") as f:
        for p in all_files:
            rel = p.relative_to(pkg_dir).as_posix()
            f.write(f"{_md5_file(p)}  {rel}\n")

    print(f"[DONE] wrote {manifest_path}")
    print(f"[DONE] wrote {md5sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
