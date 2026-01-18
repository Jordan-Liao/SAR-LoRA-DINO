#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rows(tsv_path: Path) -> List[Dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _resolve(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (repo_root / p)


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    out.sort(key=lambda x: x.as_posix())
    return out


def _pick_best_checkpoint(work_dir: Path) -> Optional[Path]:
    if not work_dir.is_dir():
        return None
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
    latest = work_dir / "latest.pth"
    if latest.is_file():
        return latest
    epochs = sorted(work_dir.glob("epoch_*.pth"), key=lambda p: p.name)
    if epochs:
        return epochs[-1]
    any_pth = sorted(work_dir.glob("*.pth"), key=lambda p: p.name)
    if any_pth:
        return any_pth[-1]
    return None


def _collect_work_dirs_from_results(rows: Sequence[Dict[str, str]]) -> List[Path]:
    out: List[Path] = []
    for r in rows:
        mp = (r.get("MetricsJSON") or "").strip()
        if not mp:
            continue
        if not mp.startswith("artifacts/work_dirs/"):
            continue
        out.append(Path(mp).parent)
    return _unique_paths(out)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            shutil.copy2(src, dst)
            return
    raise ValueError(f"Unknown link mode: {mode}")


def _zip_files(repo_root: Path, files: Sequence[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in files:
            arcname = p.resolve().relative_to(repo_root.resolve()).as_posix()
            zf.write(p, arcname)


def _split_zip_parts(
    *,
    repo_root: Path,
    files: Sequence[Path],
    out_dir: Path,
    base_name: str,
    max_zip_bytes: int,
) -> List[Path]:
    uniq = _unique_paths(files)
    if not uniq:
        return []

    groups: List[List[Path]] = []
    cur_group: List[Path] = []
    cur_bytes = 0

    def _flush_group() -> None:
        nonlocal cur_group, cur_bytes
        if not cur_group:
            return
        groups.append(cur_group)
        cur_group = []
        cur_bytes = 0

    for p in uniq:
        size = p.stat().st_size
        if size > max_zip_bytes:
            raise RuntimeError(
                f"File too large for a single zip part (size={size} max={max_zip_bytes}): {p}"
            )
        if cur_group and (cur_bytes + size) > max_zip_bytes:
            _flush_group()
        cur_group.append(p)
        cur_bytes += size

    _flush_group()

    out_paths: List[Path] = []
    for i, grp in enumerate(groups, start=1):
        if len(groups) == 1:
            zip_path = out_dir / f"{base_name}.zip"
        else:
            zip_path = out_dir / f"{base_name}_part{i:02d}.zip"
        _zip_files(repo_root, grp, zip_path)
        out_paths.append(zip_path)
    return out_paths


def _collect_logs(work_dir: Path) -> List[Path]:
    pats = [
        "train.log",
        "test_val.log",
        "test_test.log",
        "smoke_train.log",
        "smoke_test.log",
        "*.log.json",
    ]
    out: List[Path] = []
    for pat in pats:
        out.extend([p for p in work_dir.glob(pat) if p.is_file()])
    return _unique_paths(out)


def _collect_visualization_zips(repo_root: Path) -> List[Path]:
    vis_root = repo_root / "artifacts" / "visualizations"
    zips: List[Path] = []
    if not vis_root.is_dir():
        return zips
    zips.extend([p for p in vis_root.glob("VR_fullsplit_*/*.zip") if p.is_file()])
    zips.extend([p for p in vis_root.glob("VR_preds_*/*.zip") if p.is_file()])
    zips.extend([p for p in (vis_root / "VR_sample500").glob("VR_SAMPLE500_*.zip") if p.is_file()])
    zips.extend([p for p in (vis_root / "VR_sample500_lora").glob("VR_SAMPLE500_*.zip") if p.is_file()])
    zips.extend([p for p in (vis_root / "GradCAM_sample500").glob("GradCAM_SAMPLE500_*.zip") if p.is_file()])
    return _unique_paths(zips)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Release-ready bundle directory (metrics zip + checkpoints/logs zips + visualization zips). "
            "This does not run training; it packages what already exists locally."
        )
    )
    parser.add_argument(
        "--results-tsv",
        default="artifacts/experiments/experiment_results.tsv",
        help="TSV path to read (default: artifacts/experiments/experiment_results.tsv)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output dir (default: artifacts/bundles/release_<timestamp>)",
    )
    parser.add_argument(
        "--link-mode",
        default="hardlink",
        choices=["hardlink", "copy", "symlink"],
        help="How to place existing visualization zips into the bundle (default: hardlink).",
    )
    parser.add_argument("--no-metrics-zip", action="store_true", help="Do not generate metrics_jsons.zip")
    parser.add_argument("--no-ckpt-zip", action="store_true", help="Do not generate checkpoints_best*.zip")
    parser.add_argument("--no-logs-zip", action="store_true", help="Do not generate logs*.zip")
    parser.add_argument(
        "--max-zip-bytes",
        type=int,
        default=1900 * 1024 * 1024,
        help="Max bytes per zip file (default: 1900MiB, below GitHub Releases 2GiB limit).",
    )
    parser.add_argument(
        "--no-vis-zips",
        action="store_true",
        help="Do not link/copy visualization zip files into the bundle.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested category has zero packaged files.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    tsv_path = _resolve(repo_root, args.results_tsv).resolve()
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Missing results TSV: {tsv_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        (repo_root / "artifacts" / "bundles" / f"release_{ts}")
        if args.out_dir is None
        else _resolve(repo_root, args.out_dir).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(tsv_path)
    work_dirs = [_resolve(repo_root, str(p)) for p in _collect_work_dirs_from_results(rows)]

    assets: List[Tuple[str, Path]] = []

    # 1) Metrics zip
    if not args.no_metrics_zip:
        metrics_zip = out_dir / "metrics_jsons.zip"
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "package_metrics_jsons.py"),
            "--out-zip",
            str(metrics_zip),
        ]
        proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out_dir / "metrics_jsons.build.log").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to build metrics zip (see {out_dir / 'metrics_jsons.build.log'})")
        assets.append((metrics_zip.name, metrics_zip))
        # package_metrics_jsons.py also writes a .tsv next to the zip
        tsv_sidecar = metrics_zip.with_suffix(".tsv")
        if tsv_sidecar.is_file():
            assets.append((tsv_sidecar.name, tsv_sidecar))

    # 2) Checkpoints zip
    ckpts: List[Path] = []
    ckpt_manifest_lines = ["work_dir\tcheckpoint"]
    for wd in work_dirs:
        ckpt = _pick_best_checkpoint(wd)
        if ckpt is None:
            ckpt_manifest_lines.append(f"{wd.relative_to(repo_root)}\t")
            continue
        ckpts.append(ckpt)
        ckpt_manifest_lines.append(f"{wd.relative_to(repo_root)}\t{ckpt.relative_to(repo_root)}")

    ckpt_manifest = out_dir / "checkpoints_manifest.tsv"
    ckpt_manifest.write_text("\n".join(ckpt_manifest_lines) + "\n", encoding="utf-8")
    assets.append((ckpt_manifest.name, ckpt_manifest))

    if not args.no_ckpt_zip:
        if ckpts:
            ckpt_parts = _split_zip_parts(
                repo_root=repo_root,
                files=ckpts,
                out_dir=out_dir,
                base_name="checkpoints_best",
                max_zip_bytes=int(args.max_zip_bytes),
            )
            for zp in ckpt_parts:
                assets.append((zp.name, zp))
        elif args.strict:
            raise RuntimeError("No checkpoints found to package.")

    # 3) Logs zip
    logs: List[Path] = []
    logs_manifest_lines = ["work_dir\tlog_file"]
    for wd in work_dirs:
        wd_logs = _collect_logs(wd)
        if not wd_logs:
            logs_manifest_lines.append(f"{wd.relative_to(repo_root)}\t")
            continue
        for lf in wd_logs:
            logs.append(lf)
            logs_manifest_lines.append(f"{wd.relative_to(repo_root)}\t{lf.relative_to(repo_root)}")

    logs_manifest = out_dir / "logs_manifest.tsv"
    logs_manifest.write_text("\n".join(logs_manifest_lines) + "\n", encoding="utf-8")
    assets.append((logs_manifest.name, logs_manifest))

    if not args.no_logs_zip:
        if logs:
            log_parts = _split_zip_parts(
                repo_root=repo_root,
                files=logs,
                out_dir=out_dir,
                base_name="logs",
                max_zip_bytes=int(args.max_zip_bytes),
            )
            for zp in log_parts:
                assets.append((zp.name, zp))
        elif args.strict:
            raise RuntimeError("No logs found to package.")

    # 4) Visualization zip files (already packaged by visualization scripts)
    if not args.no_vis_zips:
        vis_zips = _collect_visualization_zips(repo_root)
        if not vis_zips and args.strict:
            raise RuntimeError("No visualization zip files found to include.")
        for zp in vis_zips:
            rel = zp.relative_to(repo_root)
            dst = out_dir / "visualizations" / rel.name
            _link_or_copy(zp, dst, args.link_mode)
            assets.append((dst.relative_to(out_dir).as_posix(), dst))

    # Write top-level manifest
    manifest = out_dir / "release_manifest.tsv"
    lines = ["asset_path\tbytes\tmd5"]
    for rel, path in sorted(assets, key=lambda x: x[0]):
        if not path.is_file():
            continue
        lines.append(f"{rel}\t{path.stat().st_size}\t{_md5(path)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    notes = out_dir / "release_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# Release Notes (auto-generated)",
                "",
                "This directory is a Release-ready bundle produced by `scripts/build_release_bundle.py`.",
                "",
                "## Assets",
                "",
                "- `metrics_jsons.zip`: all metrics JSONs under `artifacts/` (small).",
                "- `checkpoints_best*.zip`: best checkpoints found under `artifacts/work_dirs/**` (large; may be split).",
                "- `logs*.zip`: selected logs under `artifacts/work_dirs/**` (large; may be split).",
                "- `visualizations/*.zip`: VR fullsplit / sample500 / Grad-CAM zips (large).",
                "",
                "## Upload",
                "",
                "Upload the files in this directory to GitHub Releases (or Kaggle/cloud).",
                "Then paste the download links into `README.md` under a “Releases / Downloads” section.",
                "",
                "## Integrity",
                "",
                "Checksums are recorded in `release_manifest.tsv`.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
