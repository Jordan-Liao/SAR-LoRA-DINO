#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _epoch_num(path: Path) -> int:
    m = re.search(r"epoch_(\d+)\.pth$", path.name)
    return int(m.group(1)) if m else -1


def pick_best_checkpoint(work_dir: Path) -> Path:
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

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

    epochs = sorted(work_dir.glob("epoch_*.pth"), key=_epoch_num, reverse=True)
    if epochs:
        return epochs[0]

    raise FileNotFoundError(f"No checkpoint found under: {work_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick the best checkpoint file under a MMDet work_dir.")
    parser.add_argument("--work-dir", required=True, help="Work dir containing checkpoints")
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Print a repo-relative path if possible (default: print absolute path).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = (repo_root / work_dir).resolve()

    ckpt = pick_best_checkpoint(work_dir)
    if args.relative:
        try:
            ckpt = ckpt.relative_to(repo_root)
        except Exception:
            pass
    print(str(ckpt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

