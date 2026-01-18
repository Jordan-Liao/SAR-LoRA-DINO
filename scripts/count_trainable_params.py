#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from mmengine.config import Config, DictAction
from mmengine.utils import import_modules_from_strings

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

import sar_lora_dino  # noqa: F401


def _count_params(module, trainable_only: bool) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _maybe_getattr(obj: Any, attr: str) -> Optional[Any]:
    return getattr(obj, attr, None)


def _to_m(num: int) -> float:
    return round(num / 1e6, 3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count total/trainable parameters for an MMDet config.")
    parser.add_argument("--config", required=True, help="Path to config .py")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options with key=value pairs (mmengine style).",
    )
    parser.add_argument("--out", type=str, default="", help="Optional path to write JSON output.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    if "custom_imports" in cfg:
        import_modules_from_strings(**cfg["custom_imports"])
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = MODELS.build(cfg.model)

    total = _count_params(model, trainable_only=False)
    trainable = _count_params(model, trainable_only=True)

    breakdown: Dict[str, Dict[str, Any]] = {}
    for key in ["backbone", "neck", "bbox_head", "roi_head"]:
        sub = _maybe_getattr(model, key)
        if sub is None:
            continue
        breakdown[key] = {
            "total": _count_params(sub, trainable_only=False),
            "trainable": _count_params(sub, trainable_only=True),
        }
        breakdown[key]["total_m"] = _to_m(int(breakdown[key]["total"]))
        breakdown[key]["trainable_m"] = _to_m(int(breakdown[key]["trainable"]))

    out: Dict[str, Any] = {
        "config": str(Path(args.config)),
        "total": total,
        "trainable": trainable,
        "total_m": _to_m(total),
        "trainable_total_m": _to_m(trainable),
        "breakdown": breakdown,
    }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
