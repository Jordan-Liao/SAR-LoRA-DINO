#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METRIC_KEYS: Tuple[str, ...] = (
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
    "coco/bbox_mAP_s",
    "coco/bbox_mAP_m",
    "coco/bbox_mAP_l",
)


@dataclass(frozen=True)
class Experiment:
    eid: str
    title: str
    fields: Dict[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_experiments(md_text: str) -> List[Experiment]:
    experiments: List[Experiment] = []
    lines = md_text.splitlines()
    i = 0
    header_re = re.compile(r"^###\s+(E\d{4})\s*:\s*(.+?)\s*$")
    while i < len(lines):
        m = header_re.match(lines[i])
        if not m:
            i += 1
            continue
        eid = m.group(1)
        title = m.group(2)
        i += 1

        fields: Dict[str, str] = {}
        # Parse 2-col markdown table: | Field | Value |
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i < len(lines) and lines[i].lstrip().startswith("|"):
            # consume header + separator if present
            # header line
            _ = lines[i]
            i += 1
            if i < len(lines) and lines[i].lstrip().startswith("|") and "---" in lines[i]:
                i += 1
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                raw = lines[i].strip()
                parts = [p.strip() for p in raw.strip("|").split("|")]
                if len(parts) >= 2:
                    field = parts[0]
                    value = "|".join(parts[1:]).strip()
                    fields[field] = value
                i += 1

        experiments.append(Experiment(eid=eid, title=title, fields=fields))
    return experiments


def _find_first_config_path(text: str) -> str:
    if not text:
        return ""

    # Prefer repo configs
    for m in re.finditer(r"(configs/[A-Za-z0-9_./-]+\.py)", text):
        return m.group(1)

    # Fallback: any .py under configs/
    for m in re.finditer(r"(configs/[A-Za-z0-9_./-]+\.py)", text):
        return m.group(1)

    return ""


def _guess_config(exp: Experiment) -> str:
    for key in ("Code path", "Single-GPU script", "Multi-GPU script", "Smoke cmd", "Full cmd"):
        config = _find_first_config_path(exp.fields.get(key, ""))
        if config:
            return config

    # Try parsing from --config in cmd fields
    for key in ("Single-GPU script", "Multi-GPU script", "Smoke cmd", "Full cmd"):
        v = exp.fields.get(key, "")
        m = re.search(r"--config\s+([^\s`]+\.py)", v)
        if m:
            return m.group(1)

    return ""


def _extract_json_candidates(text: str) -> List[str]:
    if not text:
        return []

    candidates: set[str] = set()
    # Backticked segments first
    for seg in re.findall(r"`([^`]+)`", text):
        for p in re.findall(r"[A-Za-z0-9_./<>\-\\*]+\.json", seg):
            candidates.add(p.strip())
    # Then raw text
    for p in re.findall(r"[A-Za-z0-9_./<>\-\\*]+\.json", text):
        candidates.add(p.strip())

    cleaned: List[str] = []
    for p in candidates:
        cleaned.append(p.strip().strip(").,;"))
    return sorted(cleaned)


def _expand_path_patterns(patterns: Iterable[str], repo_root: Path) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for p in patterns:
        p = p.strip().strip("'\"")
        p = p.replace("<seed>", "*")
        p = p.replace("<seed0>", "*")
        p = p.replace("<seed1>", "*")
        p = p.replace("<seed2>", "*")
        path = Path(p)
        if not path.is_absolute():
            path = repo_root / path
        matches = [Path(m) for m in glob.glob(str(path))]
        if not matches and path.exists():
            matches = [path]
        for m in matches:
            if m in seen:
                continue
            seen.add(m)
            out.append(m)
    return sorted(out)


def _infer_split(metrics_path: Path) -> str:
    name = metrics_path.name
    if name == "smoke_metrics.json":
        return "smoke"
    if name == "val_metrics.json":
        return "val"
    if name == "test_metrics.json":
        return "test"
    if name == "metrics.json":
        parts = metrics_path.parts
        if "val" in parts:
            return "val"
        if "test" in parts:
            return "test"
    return ""


def _infer_run(metrics_path: Path) -> str:
    s = str(metrics_path)
    m = re.search(r"seed(\d+)", s)
    if m:
        return f"seed{m.group(1)}"
    m = re.search(r"epoch_(\d+)", s)
    if m:
        return f"epoch{m.group(1)}"
    return ""


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    if key in metrics:
        v = metrics[key]
    else:
        short = key.split("/")[-1]
        v = metrics.get(short)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _relpath(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except Exception:
        return str(path)


def _eid_sort_key(eid: str) -> int:
    m = re.search(r"(\d{4})", eid)
    return int(m.group(1)) if m else 10**9


def main() -> int:
    parser = argparse.ArgumentParser(description="Export experiment results (metrics JSONs) into TSV/MD.")
    parser.add_argument(
        "--experiment-md",
        default="artifacts/experiments/experiment.md",
        help="Path to the experiment ledger Markdown",
    )
    parser.add_argument(
        "--out-tsv", default="artifacts/experiments/experiment_results.tsv", help="Output TSV path"
    )
    parser.add_argument(
        "--out-md", default="artifacts/experiments/experiment_results.md", help="Output Markdown path"
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    exp_md_path = (repo_root / args.experiment_md).resolve()
    experiments = _parse_experiments(_read_text(exp_md_path))

    eid_to_exp: Dict[str, Experiment] = {e.eid: e for e in experiments}
    eid_to_config: Dict[str, str] = {e.eid: _guess_config(e) for e in experiments}
    eid_to_title: Dict[str, str] = {e.eid: e.title for e in experiments}

    rows: List[Dict[str, Any]] = []
    for exp in sorted(experiments, key=lambda e: _eid_sort_key(e.eid)):
        config = eid_to_config.get(exp.eid, "")
        model = exp.fields.get("Model", "")
        weights = exp.fields.get("Weights", "")
        params = exp.fields.get("Params", "")
        smoke_status = exp.fields.get("Smoke", "")
        full_status = exp.fields.get("Full", "")
        fields_blob = "\n".join(
            [exp.fields.get(k, "") for k in ("Metrics (must save)", "Artifacts", "Results", "Logs")]
        )
        json_candidates = _extract_json_candidates(fields_blob)
        json_paths = _expand_path_patterns(json_candidates, repo_root)
        json_paths = [p for p in json_paths if "metrics" in p.name]

        # If no JSONs are discoverable from the ledger, still emit a stub row.
        if not json_paths:
            rows.append(
                {
                    "Source": "ledger",
                    "EID": exp.eid,
                    "Name": exp.title,
                    "Config": config,
                    "Model": model,
                    "Weights": weights,
                    "Params": params,
                    "SmokeStatus": smoke_status,
                    "FullStatus": full_status,
                    "Run": "",
                    "Split": "",
                    "MetricsJSON": "",
                    **{k: "" for k in METRIC_KEYS},
                }
            )
            continue

        for mp in json_paths:
            metrics = _read_json(mp)
            split = _infer_split(mp)
            run = _infer_run(mp)
            row: Dict[str, Any] = {
                "Source": "ledger",
                "EID": exp.eid,
                "Name": exp.title,
                "Config": config,
                "Model": model,
                "Weights": weights,
                "Params": params,
                "SmokeStatus": smoke_status,
                "FullStatus": full_status,
                "Run": run,
                "Split": split,
                "MetricsJSON": _relpath(mp, repo_root),
            }
            for k in METRIC_KEYS:
                v = _metric_value(metrics, k)
                row[k] = "" if v is None else f"{v:.3f}"
            rows.append(row)

    # Add VR exports (full val/test) if present (use ledger config/name mapping)
    vr_candidates: Sequence[Tuple[str, str]] = (
        ("E0003", "artifacts/visualizations/VR/E0003_linear-probe"),
        ("E0019", "artifacts/visualizations/VR/E0019_full-ft_ep3"),
        ("E0019", "artifacts/visualizations/VR/E0019_full-ft"),
    )
    for eid, base in vr_candidates:
        base_path = repo_root / base
        for split in ("val", "test"):
            mp = base_path / split / "metrics.json"
            if not mp.is_file():
                continue
            metrics = _read_json(mp)
            exp = eid_to_exp.get(eid)
            rows.append(
                {
                    "Source": "vr",
                    "EID": eid,
                    "Name": eid_to_title.get(eid, ""),
                    "Config": eid_to_config.get(eid, ""),
                    "Model": "" if exp is None else exp.fields.get("Model", ""),
                    "Weights": "" if exp is None else exp.fields.get("Weights", ""),
                    "Params": "" if exp is None else exp.fields.get("Params", ""),
                    "SmokeStatus": "" if exp is None else exp.fields.get("Smoke", ""),
                    "FullStatus": "" if exp is None else exp.fields.get("Full", ""),
                    "Run": base_path.name,
                    "Split": split,
                    "MetricsJSON": _relpath(mp, repo_root),
                    **{
                        k: ("" if _metric_value(metrics, k) is None else f"{_metric_value(metrics, k):.3f}")
                        for k in METRIC_KEYS
                    },
                }
            )

    header: List[str] = [
        "Source",
        "EID",
        "Name",
        "Config",
        "Model",
        "Weights",
        "Params",
        "SmokeStatus",
        "FullStatus",
        "Run",
        "Split",
        "MetricsJSON",
        *METRIC_KEYS,
    ]

    out_tsv_path = (repo_root / args.out_tsv).resolve()
    out_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=header, delimiter="\t", extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_md_path = (repo_root / args.out_md).resolve()
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    with out_md_path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Results (auto-export)\n\n")
        f.write("Copy-friendly table (TSV source): `" + _relpath(out_tsv_path, repo_root) + "`\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(h, "")).replace("\n", " ") for h in header) + " |\n")

    print(str(out_tsv_path))
    print(str(out_md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
