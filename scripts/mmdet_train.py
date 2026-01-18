#!/usr/bin/env python3
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

import sar_lora_dino  # noqa: F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MMDetection model (MMEngine runner).")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--work-dir", default=None, help="Directory to save logs and checkpoints")
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        default=None,
        help=(
            "Resume from a checkpoint path; if provided without a value, tries to auto-resume "
            "from the latest checkpoint in work_dir."
        ),
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default=None,
        help="Override config options with key=value pairs",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="Job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> int:
    args = _parse_args()

    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])

    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
