#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.apis import init_detector

import sar_lora_dino  # noqa: F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Grad-CAM-like heatmap overlays for MMDetection detectors on a COCO subset."
    )
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--ann-file", required=True, help="COCO JSON (absolute or repo-relative)")
    parser.add_argument(
        "--img-prefix",
        required=True,
        help="Image prefix under data_root, e.g. JPEGImages/val/ or JPEGImages/test/",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device for inference/grad (default: cuda)")
    parser.add_argument(
        "--target-layer",
        default="backbone.backbone.stages_3",
        help="Dotted module path for Grad-CAM target layer (default: backbone.backbone.stages_3)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Use sum(topk sigmoid(cls_scores)) as Grad target (default: 100)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay alpha in [0,1] (default: 0.45)",
    )
    parser.add_argument(
        "--colormap",
        default="JET",
        help="OpenCV colormap name (default: JET). Options: JET/HOT/VIRIDIS/TURBO...",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional: limit number of images (debug)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader num_workers (default: 0 for stability)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options with key=value pairs",
    )
    return parser.parse_args()


def _resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(path.resolve())


def _colormap_from_name(name: str) -> int:
    key = name.strip().upper()
    value = getattr(cv2, f"COLORMAP_{key}", None)
    if value is None:
        raise ValueError(f"Unknown OpenCV colormap: {name} (expected e.g. JET/HOT/VIRIDIS/TURBO)")
    return int(value)


def _compute_topk_target(cls_scores: Tuple[torch.Tensor, ...], topk: int) -> torch.Tensor:
    flat_scores = []
    for score in cls_scores:
        if not torch.is_tensor(score):
            raise TypeError(f"Expected cls_scores tensors, got: {type(score)}")
        flat_scores.append(score.sigmoid().flatten(start_dim=1))
    all_scores = torch.cat(flat_scores, dim=1)
    k = min(int(topk), int(all_scores.shape[1]))
    values = torch.topk(all_scores, k=k, dim=1).values
    return values.sum()


def _normalize_cam(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    cam = torch.relu(cam)
    cam_min = cam.amin(dim=(-2, -1), keepdim=True)
    cam = cam - cam_min
    cam_max = cam.amax(dim=(-2, -1), keepdim=True)
    cam = cam / (cam_max + eps)
    return cam


class _ActivationAndGrad:
    def __init__(self, layer: torch.nn.Module) -> None:
        self.activation: Optional[torch.Tensor] = None
        self.gradient: Optional[torch.Tensor] = None
        self._handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: torch.nn.Module, inputs: Any, output: Any) -> None:
        if not torch.is_tensor(output):
            raise TypeError(
                f"Grad-CAM target layer must output a Tensor, got: {type(output)}. "
                "Pick a different --target-layer."
            )
        self.activation = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self.gradient = grad

    def close(self) -> None:
        self._handle.remove()


def main() -> int:
    args = _parse_args()

    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get("default_scope", "mmdet"))

    ann_file = _resolve_path(args.ann_file)
    cfg.test_dataloader.dataset.ann_file = ann_file
    cfg.test_dataloader.dataset.data_prefix.img = args.img_prefix
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = int(args.num_workers)
    cfg.test_dataloader.persistent_workers = bool(args.num_workers > 0)

    device = args.device
    model = init_detector(cfg, args.checkpoint, device=device)
    model.eval()

    try:
        target_layer = model.get_submodule(args.target_layer)
    except Exception as e:
        raise ValueError(f"Cannot resolve --target-layer={args.target_layer}") from e

    hook = _ActivationAndGrad(target_layer)
    cmap = _colormap_from_name(args.colormap)

    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    meta: Dict[str, Any] = {
        "config": str(Path(args.config).as_posix()),
        "checkpoint": str(Path(args.checkpoint).as_posix()),
        "ann_file": ann_file,
        "img_prefix": args.img_prefix,
        "device": device,
        "target_layer": args.target_layer,
        "topk": int(args.topk),
        "overlay_alpha": float(args.overlay_alpha),
        "colormap": args.colormap,
        "start_time": time.strftime("%F %T"),
        "images_total": len(dataloader),
        "images_done": 0,
        "failures": [],
    }

    max_images = args.max_images if args.max_images is not None else len(dataloader)

    for idx, data in enumerate(dataloader):
        if idx >= max_images:
            break

        data_samples = data.get("data_samples")
        if not isinstance(data_samples, list) or not data_samples:
            raise RuntimeError("Unexpected dataloader output: missing data_samples list")

        img_path = data_samples[0].metainfo.get("img_path")
        if not img_path:
            raise RuntimeError("Missing img_path in data_samples metainfo")

        img_name = Path(img_path).name
        out_path = overlay_dir / img_name
        if out_path.is_file():
            meta["images_done"] += 1
            continue

        try:
            with torch.set_grad_enabled(True):
                processed = model.data_preprocessor(data, training=False)
                inputs = processed["inputs"]
                inputs = inputs.requires_grad_(True)

                model.zero_grad(set_to_none=True)

                feats = model.extract_feat(inputs)
                cls_scores, _bbox_preds = model.bbox_head(feats)
                target = _compute_topk_target(tuple(cls_scores), topk=args.topk)
                target.backward()

                if hook.activation is None or hook.gradient is None:
                    raise RuntimeError("Grad-CAM hook did not capture activation/gradient")
                if hook.activation.ndim != 4 or hook.gradient.ndim != 4:
                    raise RuntimeError(
                        f"Expected 4D activation/gradient, got act={hook.activation.shape}, grad={hook.gradient.shape}"
                    )

                weights = hook.gradient.mean(dim=(2, 3), keepdim=True)
                cam = (weights * hook.activation).sum(dim=1, keepdim=True)
                cam = _normalize_cam(cam)[0, 0].detach().float().cpu().numpy()

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")

            cam_u8 = (cam * 255.0).clip(0, 255).astype(np.uint8)
            cam_color = cv2.applyColorMap(cam_u8, cmap)
            cam_color = cv2.resize(cam_color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            overlay = cv2.addWeighted(img, 1.0 - args.overlay_alpha, cam_color, args.overlay_alpha, 0)

            ok = cv2.imwrite(str(out_path), overlay)
            if not ok:
                raise RuntimeError(f"Failed to write {out_path}")

            meta["images_done"] += 1
        except Exception as e:
            meta["failures"].append({"img": img_name, "error": str(e)})
            continue

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{max_images}] wrote {meta['images_done']} overlays -> {overlay_dir}")

    hook.close()
    meta["end_time"] = time.strftime("%F %T")

    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    print(str(meta_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
