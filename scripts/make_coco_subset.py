#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic COCO subset by sampling images.")
    parser.add_argument("--in-json", required=True, help="Path to input COCO JSON")
    parser.add_argument("--out-json", required=True, help="Path to write subset COCO JSON")
    parser.add_argument("--num-images", type=int, required=True, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow sampling images with zero annotations (default: only non-empty images).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        coco: Dict[str, Any] = json.load(f)

    images: List[Dict[str, Any]] = list(coco.get("images", []))
    annotations: List[Dict[str, Any]] = list(coco.get("annotations", []))

    anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[int(ann["image_id"])].append(ann)

    if args.allow_empty:
        candidate_images = images
    else:
        candidate_images = [img for img in images if int(img["id"]) in anns_by_image]

    if not candidate_images:
        raise SystemExit("No candidate images found (try --allow-empty).")

    candidate_images = sorted(candidate_images, key=lambda x: int(x["id"]))
    rng = random.Random(args.seed)
    rng.shuffle(candidate_images)

    num_images = min(args.num_images, len(candidate_images))
    selected_images = sorted(candidate_images[:num_images], key=lambda x: int(x["id"]))
    selected_image_ids = {int(img["id"]) for img in selected_images}

    selected_annotations = [ann for ann in annotations if int(ann["image_id"]) in selected_image_ids]

    ann_ids = [int(ann["id"]) for ann in selected_annotations if "id" in ann]
    if len(ann_ids) != len(set(ann_ids)):
        raise SystemExit("Selected annotations contain duplicate `id` values.")

    out: Dict[str, Any] = {k: v for k, v in coco.items() if k not in {"images", "annotations"}}
    out["images"] = selected_images
    out["annotations"] = selected_annotations

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(
        json.dumps(
            {
                "in_json": str(Path(args.in_json)),
                "out_json": str(out_path),
                "num_images": len(selected_images),
                "num_annotations": len(selected_annotations),
                "allow_empty": bool(args.allow_empty),
                "seed": int(args.seed),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

