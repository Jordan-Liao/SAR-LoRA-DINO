#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_rows(tsv_path: Path) -> List[Dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _resolve(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (repo_root / p)


def _unique(seq: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for p in seq:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
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
    return _unique(sorted(out, key=lambda p: p.as_posix()))


def _has_any_log(work_dir: Path) -> bool:
    patterns = [
        "train.log",
        "test_val.log",
        "test_test.log",
        "smoke_train.log",
        "smoke_test.log",
        "*.log.json",
    ]
    for pat in patterns:
        if any(work_dir.glob(pat)):
            return True
    return False


def _collect_vr_split_dirs(vr_root: Path) -> List[Path]:
    # Expect: VR/<tag>/<split>/
    out: List[Path] = []
    if not vr_root.is_dir():
        return out
    for tag_dir in sorted([p for p in vr_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        for split_dir in sorted([p for p in tag_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            out.append(split_dir)
    return out


def _vr_split_ok(split_dir: Path) -> Tuple[bool, str]:
    metrics = split_dir / "metrics.json"
    preds = split_dir / "predictions.pkl"
    log = split_dir / "export.log"
    if not metrics.is_file():
        return False, "missing metrics.json"
    if not preds.is_file():
        return False, "missing predictions.pkl"
    if not log.is_file():
        return True, "ok (no export.log)"

    ts_dirs = [p for p in split_dir.iterdir() if p.is_dir() and TIMESTAMP_RE.match(p.name)]
    if not ts_dirs:
        return True, "ok (no timestamp/vis)"
    ts_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    vis_dir = ts_dirs[0] / "vis"
    if not vis_dir.is_dir():
        return True, "ok (no vis/ under latest timestamp)"
    vis_files = [p for p in vis_dir.iterdir() if p.is_file()]
    if not vis_files:
        return True, "ok (empty vis/ under latest timestamp)"
    return True, "ok"


def _glob_count(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    return len([p for p in root.glob(pattern) if p.is_file()])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether Release-level assets exist locally (checkpoints/logs/full VR/sample500/Grad-CAM). "
            "This does not download or run training; it only reports what is present."
        )
    )
    parser.add_argument(
        "--results-tsv",
        default="artifacts/experiments/experiment_results.tsv",
        help="TSV path to read (default: artifacts/experiments/experiment_results.tsv)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any required Release category is empty.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    tsv_path = _resolve(repo_root, args.results_tsv).resolve()
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Missing results TSV: {tsv_path}")

    rows = _read_rows(tsv_path)
    work_dirs = [_resolve(repo_root, str(p)) for p in _collect_work_dirs_from_results(rows)]

    ckpt_ok = 0
    ckpt_missing: List[str] = []
    log_ok = 0
    log_missing: List[str] = []
    for wd in work_dirs:
        ckpt = _pick_best_checkpoint(wd)
        if ckpt is None:
            if len(ckpt_missing) < 30:
                ckpt_missing.append(str(wd.relative_to(repo_root)))
        else:
            ckpt_ok += 1

        if _has_any_log(wd):
            log_ok += 1
        else:
            if len(log_missing) < 30:
                log_missing.append(str(wd.relative_to(repo_root)))

    vr_root = repo_root / "artifacts" / "visualizations" / "VR"
    vr_split_dirs = _collect_vr_split_dirs(vr_root)
    vr_preds_ok = 0
    vr_full_ok = 0
    vr_exports_bad: List[str] = []
    for sd in vr_split_dirs:
        ok, reason = _vr_split_ok(sd)
        if ok:
            vr_preds_ok += 1
            if reason == "ok":
                vr_full_ok += 1
        else:
            if len(vr_exports_bad) < 30:
                vr_exports_bad.append(f"{sd.relative_to(repo_root)} ({reason})")

    vis_root = repo_root / "artifacts" / "visualizations"
    vr_fullsplit_zips = _glob_count(vis_root, "VR_fullsplit_*/*.zip")
    vr_preds_zips = _glob_count(vis_root, "VR_preds_*/*.zip")
    vr_sample500_zips = _glob_count(vis_root / "VR_sample500", "VR_SAMPLE500_*.zip")
    vr_sample500_lora_zips = _glob_count(vis_root / "VR_sample500_lora", "VR_SAMPLE500_*.zip")
    gradcam_sample500_zips = _glob_count(vis_root / "GradCAM_sample500", "GradCAM_SAMPLE500_*.zip")

    summary = {
        "results_tsv": str(tsv_path.relative_to(repo_root)),
        "work_dirs": {
            "total": len(work_dirs),
            "ckpt_present": ckpt_ok,
            "ckpt_missing": len(work_dirs) - ckpt_ok,
            "ckpt_missing_examples": ckpt_missing,
            "logs_present": log_ok,
            "logs_missing": len(work_dirs) - log_ok,
            "logs_missing_examples": log_missing,
        },
        "visualizations": {
            "vr_exports_split_dirs_total": len(vr_split_dirs),
            "vr_exports_preds_ok": vr_preds_ok,
            "vr_exports_full_ok": vr_full_ok,
            "vr_exports_bad_examples": vr_exports_bad,
            "vr_fullsplit_zips": vr_fullsplit_zips,
            "vr_preds_zips": vr_preds_zips,
            "vr_sample500_zips": vr_sample500_zips,
            "vr_sample500_lora_zips": vr_sample500_lora_zips,
            "gradcam_sample500_zips": gradcam_sample500_zips,
        },
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"results_tsv: {summary['results_tsv']}")
        wd = summary["work_dirs"]
        print(
            "work_dirs:"
            f" total={wd['total']}"
            f" ckpt_present={wd['ckpt_present']}"
            f" logs_present={wd['logs_present']}"
        )
        vis = summary["visualizations"]
        print(
            "visualizations:"
            f" vr_exports_preds_ok={vis['vr_exports_preds_ok']}/{vis['vr_exports_split_dirs_total']}"
            f" vr_exports_full_ok={vis['vr_exports_full_ok']}/{vis['vr_exports_split_dirs_total']}"
            f" vr_fullsplit_zips={vis['vr_fullsplit_zips']}"
            f" vr_preds_zips={vis['vr_preds_zips']}"
            f" vr_sample500_zips={vis['vr_sample500_zips']}"
            f" vr_sample500_lora_zips={vis['vr_sample500_lora_zips']}"
            f" gradcam_sample500_zips={vis['gradcam_sample500_zips']}"
        )
        if wd["ckpt_missing"]:
            print("missing checkpoints examples:")
            for p in wd["ckpt_missing_examples"]:
                print(f"- {p}")
        if wd["logs_missing"]:
            print("missing logs examples:")
            for p in wd["logs_missing_examples"]:
                print(f"- {p}")
        if vis["vr_exports_bad_examples"]:
            print("incomplete VR exports examples:")
            for p in vis["vr_exports_bad_examples"]:
                print(f"- {p}")

    required_missing = (
        ckpt_ok == 0
        or log_ok == 0
        or (vr_fullsplit_zips + vr_preds_zips) == 0
        or (vr_sample500_zips + vr_sample500_lora_zips) == 0
        or gradcam_sample500_zips == 0
    )
    return 1 if (args.strict and required_missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
