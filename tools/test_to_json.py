#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

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
    parser = argparse.ArgumentParser(description="Run MMDet test and save returned metrics to JSON.")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--work-dir", required=True, help="Work dir for the runner")
    parser.add_argument("--out-json", required=True, help="Where to write metrics JSON")
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


def main() -> int:
    args = _parse_args()

    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    metrics: Dict[str, Any] = runner.test()
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(metrics), f, indent=2, sort_keys=True)
        f.write("\n")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
