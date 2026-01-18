#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence


TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export SARDet VR visualizations on a full split (default: test), then pack into a single zip."
        )
    )
    parser.add_argument(
        "--name", required=True, help="VR export name/tag under artifacts/visualizations/VR/<name>/..."
    )
    parser.add_argument("--config", required=True, help="MMDet config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split to export (default: test)")
    parser.add_argument("--env-name", default="sar_lora_dino", help="Conda env name (default: sar_lora_dino)")
    parser.add_argument(
        "--out-root",
        default="artifacts/visualizations/VR",
        help="VR output root (default: artifacts/visualizations/VR)",
    )
    parser.add_argument("--cuda-visible-devices", default=None, help="CUDA_VISIBLE_DEVICES for export (optional)")
    parser.add_argument(
        "--pkg-dir",
        default=None,
        help="Package output dir (default: artifacts/visualizations/VR_fullsplit_<name>_<split>_<timestamp>)",
    )
    parser.add_argument("--skip-export", action="store_true", help="Skip VR export and only pack existing outputs")
    parser.add_argument("--overwrite-pkg", action="store_true", help="Overwrite existing pkg-dir if it exists")
    parser.add_argument(
        "--preds-only",
        action="store_true",
        help="Pack only metrics.json + predictions.pkl (+ export.log) and skip per-image visualizations.",
    )
    parser.add_argument(
        "--expected-vis-count",
        type=int,
        default=None,
        help="Expected number of visualization images (optional; if unset, try to infer from COCO ann json)",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: Sequence[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== [{time.strftime('%F %T')}] CMD: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)} (see {log_path})")


def _md5_file(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _find_latest_timestamp_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and TIMESTAMP_RE.match(p.name)]
    if not candidates:
        raise RuntimeError(f"No timestamp dir under {root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from (q for q in p.rglob("*") if q.is_file())


def _infer_expected_vis_count(split: str) -> Optional[int]:
    repo_root = _repo_root()
    sardet_root = Path(os.environ.get("SARDET100K_ROOT", str(repo_root / "data" / "SARDet_100K"))).resolve()
    ann = sardet_root / "Annotations" / f"{split}.json"
    if not ann.is_file():
        return None
    try:
        data = json.loads(ann.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    images = data.get("images")
    if isinstance(images, list):
        return len(images)
    return None


def _zip_vr_split(repo_root: Path, split_dir: Path, timestamp_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    split_files = [p for p in split_dir.iterdir() if p.is_file()]
    ts_files = list(_iter_files([timestamp_dir]))
    files = sorted(split_files + ts_files, key=lambda p: p.as_posix())

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in files:
            arcname = p.resolve().relative_to(repo_root.resolve()).as_posix()
            zf.write(p, arcname)


def _zip_vr_preds_only(repo_root: Path, split_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    wanted = ["metrics.json", "predictions.pkl", "export.log"]
    files = [split_dir / name for name in wanted if (split_dir / name).is_file()]
    if not files:
        raise RuntimeError(f"No metrics/predictions found under {split_dir}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in files:
            arcname = p.resolve().relative_to(repo_root.resolve()).as_posix()
            zf.write(p, arcname)


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()

    config = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    checkpoint = (
        (repo_root / args.checkpoint).resolve()
        if not Path(args.checkpoint).is_absolute()
        else Path(args.checkpoint).resolve()
    )
    out_root = (repo_root / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root).resolve()

    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    split_dir = out_root / args.name / args.split

    export_log = repo_root / "work_dirs" / "_vr_export_logs" / f"{args.name}_{args.split}.log"
    if not args.skip_export:
        env = os.environ.copy()
        env["ENV_NAME"] = args.env_name
        env["OUT_ROOT"] = str(out_root)
        env["SPLITS"] = args.split
        if args.preds_only:
            env["EXPORT_VIS"] = "0"
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        _run(
            [
                "bash",
                str(repo_root / "visualization/export_sardet_vr.sh"),
                "--name",
                args.name,
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
            ],
            cwd=repo_root,
            env=env,
            log_path=export_log,
        )

    if not split_dir.is_dir():
        raise RuntimeError(f"Missing split dir: {split_dir}")

    ts_dir = _find_latest_timestamp_dir(split_dir) if not args.preds_only else None
    vis_count: Optional[int] = None
    if not args.preds_only:
        assert ts_dir is not None
        vis_dir = ts_dir / "vis"
        if not vis_dir.is_dir():
            raise RuntimeError(f"Missing vis dir: {vis_dir}")

        expected = (
            args.expected_vis_count if args.expected_vis_count is not None else _infer_expected_vis_count(args.split)
        )
        vis_files = [p for p in vis_dir.iterdir() if p.is_file()]
        vis_count = len(vis_files)
        if expected is not None and vis_count != expected:
            raise RuntimeError(f"Vis image count mismatch under {vis_dir}: expected={expected} got={vis_count}")

    pkg_dir: Path
    pkg_ts: str
    if args.pkg_dir is None:
        if args.preds_only:
            pkg_ts = time.strftime("%Y%m%d_%H%M%S")
            pkg_dir = repo_root / "artifacts" / "visualizations" / f"VR_preds_{args.name}_{args.split}_{pkg_ts}"
        else:
            assert ts_dir is not None
            pkg_ts = ts_dir.name
            pkg_dir = (
                repo_root / "artifacts" / "visualizations" / f"VR_fullsplit_{args.name}_{args.split}_{pkg_ts}"
            )
    else:
        pkg_dir = (repo_root / args.pkg_dir).resolve()
        pkg_ts = time.strftime("%Y%m%d_%H%M%S") if args.preds_only else (ts_dir.name if ts_dir is not None else time.strftime("%Y%m%d_%H%M%S"))

    if pkg_dir.exists() and args.overwrite_pkg:
        for p in sorted(pkg_dir.glob("*")):
            if p.is_file():
                p.unlink()
            else:
                # Avoid accidental destructive delete of non-file outputs.
                raise RuntimeError(f"Refuse to overwrite non-empty pkg dir with subdirs: {pkg_dir}")
    pkg_dir.mkdir(parents=True, exist_ok=True)

    if args.preds_only:
        zip_name = f"VR_PREDS_{args.name}_{args.split}_{pkg_ts}.zip"
    else:
        assert ts_dir is not None
        zip_name = f"VR_FULLSPLIT_{args.name}_{args.split}_{pkg_ts}.zip"
    zip_path = pkg_dir / zip_name
    if args.preds_only:
        _zip_vr_preds_only(repo_root, split_dir, zip_path)
    else:
        assert ts_dir is not None
        _zip_vr_split(repo_root, split_dir, ts_dir, zip_path)

    readme = pkg_dir / "README.md"
    if args.preds_only:
        readme.write_text(
            "\n".join(
                [
                    f"# {args.name} 全量 {args.split.upper()} 预测导出（无逐图可视化）",
                    "",
                    f"本目录提供 `{args.name}` 在 SARDet_100K **{args.split} 全量**上的导出打包：仅包含 metrics + predictions（不包含逐图可视化图片）。",
                    "",
                    "## 1) 包含内容",
                    "",
                    f"- `{zip_name}`",
                    f"  - `artifacts/visualizations/VR/{args.name}/{args.split}/metrics.json`",
                    f"  - `artifacts/visualizations/VR/{args.name}/{args.split}/predictions.pkl`",
                    f"  - `artifacts/visualizations/VR/{args.name}/{args.split}/export.log`（若存在）",
                    "- `md5sum.txt`：本目录文件 MD5 校验",
                    "",
                    "## 2) 生成方式（可复现）",
                    "",
                    "```bash",
                    f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices or '<auto>'} SPLITS={args.split} OUT_ROOT={args.out_root} EXPORT_VIS=0 ENV_NAME={args.env_name} \\",
                    "  bash visualization/export_sardet_vr.sh \\",
                    f"    --name {args.name} \\",
                    f"    --config {config.relative_to(repo_root)} \\",
                    f"    --checkpoint {checkpoint.relative_to(repo_root)}",
                    "```",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        assert ts_dir is not None and vis_count is not None
        readme.write_text(
            "\n".join(
                [
                    f"# {args.name} 全量 {args.split.upper()} 可视化导出",
                    "",
                    f"本目录提供 `{args.name}` 在 SARDet_100K **{args.split} 全量**上的 VR 导出打包（逐图可视化 + metrics.json + predictions.pkl）。",
                    "",
                    "## 1) 包含内容",
                    "",
                    f"- `{zip_name}`",
                    f"  - `artifacts/visualizations/VR/{args.name}/{args.split}/metrics.json`",
                    f"  - `artifacts/visualizations/VR/{args.name}/{args.split}/predictions.pkl`",
                    f"  - `artifacts/visualizations/VR/{args.name}/{args.split}/export.log`",
                    f"  - `artifacts/visualizations/VR/{args.name}/{args.split}/{ts_dir.name}/vis/*`（{vis_count} 张可视化图）",
                    "- `md5sum.txt`：本目录文件 MD5 校验",
                    "",
                    "## 2) 生成方式（可复现）",
                    "",
                    "在仓库根目录执行（强制 `score_thr=0.0` 便于可视化不漏框）：",
                    "",
                    "```bash",
                    f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices or '<auto>'} SPLITS={args.split} OUT_ROOT={args.out_root} EXPORT_VIS=1 ENV_NAME={args.env_name} \\",
                    "  bash visualization/export_sardet_vr.sh \\",
                    f"    --name {args.name} \\",
                    f"    --config {config.relative_to(repo_root)} \\",
                    f"    --checkpoint {checkpoint.relative_to(repo_root)}",
                    "```",
                    "",
                    "## 3) 打包与分享",
                    "",
                    "你可以把该目录（或 zip 文件）上传到 GitHub Releases / Kaggle / 云盘等作为补充材料；`md5sum.txt` 可用于校验文件一致性。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    md5sum_path = pkg_dir / "md5sum.txt"
    md5sum_path.write_text(
        "\n".join(
            [
                f"{_md5_file(readme)}  {readme.name}",
                f"{_md5_file(zip_path)}  {zip_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"OK: {pkg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
