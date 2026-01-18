#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_rows(tsv_path: Path) -> List[Dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _file_exists(repo_root: Path, rel: str) -> bool:
    rel = (rel or "").strip()
    if not rel:
        return False
    p = Path(rel)
    if not p.is_absolute():
        p = repo_root / p
    return p.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether paper artifacts exist locally under artifacts/.")
    parser.add_argument(
        "--results-tsv",
        default="artifacts/experiments/experiment_results.tsv",
        help="TSV path to check (default: artifacts/experiments/experiment_results.tsv)",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if anything is missing.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    repo_root = _repo_root()
    tsv_path = (repo_root / args.results_tsv).resolve()
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Missing results TSV: {tsv_path}")

    rows = _read_rows(tsv_path)
    total_rows = len(rows)

    empty_metrics = 0
    missing_metrics = 0
    present_metrics = 0
    missing_paths: List[str] = []
    for r in rows:
        mp = (r.get("MetricsJSON") or "").strip()
        if not mp:
            empty_metrics += 1
            continue
        if _file_exists(repo_root, mp):
            present_metrics += 1
        else:
            missing_metrics += 1
            if len(missing_paths) < 50:
                missing_paths.append(mp)

    # Visualization quick checks (avoid deep traversal of VR vis/ images)
    vr_metrics = sorted(
        (repo_root / "artifacts" / "visualizations" / "VR").glob("*/*/metrics.json")
    )
    vr_fullsplit_zips = sorted(
        (repo_root / "artifacts" / "visualizations").glob("VR_fullsplit_*/*.zip")
    )
    vr_sample500_zips = sorted(
        (repo_root / "artifacts" / "visualizations" / "VR_sample500").glob("VR_SAMPLE500_*.zip")
    )
    vr_sample500_lora_zips = sorted(
        (repo_root / "artifacts" / "visualizations" / "VR_sample500_lora").glob("VR_SAMPLE500_*.zip")
    )
    gradcam_sample500_zips = sorted(
        (repo_root / "artifacts" / "visualizations" / "GradCAM_sample500").glob("GradCAM_SAMPLE500_*.zip")
    )

    summary = {
        "results_tsv": str(tsv_path.relative_to(repo_root)),
        "tsv_rows": total_rows,
        "metricsjson": {
            "present": present_metrics,
            "missing": missing_metrics,
            "empty": empty_metrics,
            "missing_examples": missing_paths,
        },
        "visualizations": {
            "vr_metrics_json": len(vr_metrics),
            "vr_fullsplit_zips": len(vr_fullsplit_zips),
            "vr_sample500_zips": len(vr_sample500_zips),
            "vr_sample500_lora_zips": len(vr_sample500_lora_zips),
            "gradcam_sample500_zips": len(gradcam_sample500_zips),
        },
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"results_tsv: {summary['results_tsv']}")
        print(
            "metricsjson:"
            f" present={present_metrics}"
            f" missing={missing_metrics}"
            f" empty={empty_metrics}"
            f" (rows={total_rows})"
        )
        print(
            "visualizations:"
            f" vr_metrics_json={len(vr_metrics)}"
            f" vr_fullsplit_zips={len(vr_fullsplit_zips)}"
            f" vr_sample500_zips={len(vr_sample500_zips)}"
            f" vr_sample500_lora_zips={len(vr_sample500_lora_zips)}"
            f" gradcam_sample500_zips={len(gradcam_sample500_zips)}"
        )
        if missing_metrics:
            print("missing MetricsJSON examples:")
            for p in missing_paths:
                print(f"- {p}")

    # Empty MetricsJSON means "not applicable" for that row (e.g. visualization-only entries).
    # Strict mode should only fail when a row *claims* to have a MetricsJSON path but it's missing.
    return 1 if (args.strict and missing_metrics > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
