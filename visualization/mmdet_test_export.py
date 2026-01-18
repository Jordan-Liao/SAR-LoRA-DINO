#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.utils import setup_cache_size_limit_of_dynamo

import sar_lora_dino  # noqa: F401


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_builtin(item())
        except Exception:
            return str(value)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMDet test once and export metrics, predictions, and visualizations."
    )
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--work-dir", required=True, help="Work dir for the runner/outputs")
    parser.add_argument("--out-metrics", required=True, help="Where to write metrics JSON")
    parser.add_argument(
        "--out-pkl",
        default=None,
        help="Optional: dump predictions to a .pkl/.pickle file (DumpDetResults)",
    )
    parser.add_argument("--show", action="store_true", help="Show prediction results")
    parser.add_argument(
        "--show-dir",
        default=None,
        help="Directory name where painted images will be saved under work_dir/timestamp/<show_dir>",
    )
    parser.add_argument("--wait-time", type=float, default=2, help="Show interval (s)")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="Job launcher",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options with key=value pairs",
    )
    return parser.parse_args()


def _validate_out_pkl(out_pkl: Optional[str]) -> Optional[Path]:
    if out_pkl is None:
        return None
    out_path = Path(out_pkl)
    if out_path.suffix not in {".pkl", ".pickle"}:
        raise ValueError(f"--out-pkl must end with .pkl or .pickle, got: {out_pkl}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def main() -> int:
    args = _parse_args()

    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    out_pkl = _validate_out_pkl(args.out_pkl)
    if out_pkl is not None:
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=str(out_pkl)))

    metrics: Dict[str, Any] = runner.test()

    out_metrics_path = Path(args.out_metrics)
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(metrics), f, indent=2, sort_keys=True)
        f.write("\n")

    print(str(out_metrics_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
