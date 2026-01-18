#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import time
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_metrics_files(repo_root: Path) -> Iterable[Path]:
    # Work dirs: metrics jsons are small, safe to glob recursively.
    wd = repo_root / "artifacts" / "work_dirs"
    if wd.is_dir():
        yield from wd.rglob("*metrics*.json")
        yield from wd.rglob("metrics.json")

    # VR exports: avoid crawling the huge timestamp/vis tree by using fixed depth.
    vr_root = repo_root / "artifacts" / "visualizations" / "VR"
    if vr_root.is_dir():
        yield from vr_root.glob("*/*/metrics.json")

    # Sample500 bundles: keep globs shallow to avoid traversing full VR image trees.
    vis_root = repo_root / "artifacts" / "visualizations"
    if vis_root.is_dir():
        # LoRA sample500 exporter copies metrics into a dedicated folder.
        lora_metrics = vis_root / "VR_sample500_lora" / "metrics"
        if lora_metrics.is_dir():
            yield from lora_metrics.glob("*/*/metrics.json")


def _unique_existing(paths: Iterable[Path]) -> List[Path]:
    seen: set[Path] = set()
    out: List[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file():
            out.append(rp)
    out.sort(key=lambda x: x.as_posix())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Package all metrics JSONs under artifacts/ into a single zip.")
    parser.add_argument(
        "--out-zip",
        default=None,
        help="Output zip path (default: artifacts/bundles/metrics_jsons_<timestamp>.zip)",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_zip = (
        (repo_root / "artifacts" / "bundles" / f"metrics_jsons_{ts}.zip")
        if args.out_zip is None
        else (repo_root / args.out_zip).resolve()
    )
    out_tsv = out_zip.with_suffix(".tsv")
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    files = _unique_existing(_iter_metrics_files(repo_root))
    if not files:
        raise RuntimeError("No metrics JSON files found under artifacts/.")

    rows: List[Tuple[str, int, str]] = []
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in files:
            rel = p.relative_to(repo_root).as_posix()
            zf.write(p, rel)
            rows.append((rel, p.stat().st_size, _md5(p)))

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        f.write("path\tbytes\tmd5\n")
        for rel, size, md5 in rows:
            f.write(f"{rel}\t{size}\t{md5}\n")

    print(str(out_zip))
    print(str(out_tsv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
