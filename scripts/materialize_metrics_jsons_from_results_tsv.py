#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_rows(tsv_path: Path) -> List[Dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _parse_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _metrics_from_row(row: Dict[str, str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, val in row.items():
        if not key:
            continue
        if not key.startswith("coco/"):
            continue
        fv = _parse_float(val)
        if fv is None:
            continue
        metrics[key] = fv
    return metrics


def _resolve_path(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (repo_root / p)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize small metrics JSON files under artifacts/work_dirs/ from the recorded "
            "paper results table (artifacts/experiments/experiment_results.tsv)."
        )
    )
    parser.add_argument(
        "--results-tsv",
        default="artifacts/experiments/experiment_results.tsv",
        help="TSV path to read (default: artifacts/experiments/experiment_results.tsv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metrics JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing files.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    tsv_path = _resolve_path(repo_root, args.results_tsv).resolve()
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Missing results TSV: {tsv_path}")

    rows = _read_rows(tsv_path)
    created = 0
    skipped = 0
    overwritten = 0
    empty = 0

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        out_rel = (row.get("MetricsJSON") or "").strip()
        if not out_rel:
            empty += 1
            continue

        out_path = _resolve_path(repo_root, out_rel)
        exists = out_path.is_file()
        if exists and not args.force:
            skipped += 1
            continue

        metrics = _metrics_from_row(row)
        payload = dict(metrics)
        payload["_meta"] = {
            "generated_at": ts,
            "source_tsv": str(tsv_path.relative_to(repo_root)),
            "eid": (row.get("EID") or "").strip(),
            "name": (row.get("Name") or "").strip(),
            "run": (row.get("Run") or "").strip(),
            "split": (row.get("Split") or "").strip(),
            "note": (
                "This file is materialized from the recorded paper table. "
                "It may not contain every raw evaluator field produced by MMDetection."
            ),
        }

        if args.dry_run:
            print(f"[dry-run] write {out_path.relative_to(repo_root)}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

        if exists:
            overwritten += 1
        else:
            created += 1

    print(
        "materialize_metrics_jsons:"
        f" created={created}"
        f" overwritten={overwritten}"
        f" skipped={skipped}"
        f" empty_metricsjson_rows={empty}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

