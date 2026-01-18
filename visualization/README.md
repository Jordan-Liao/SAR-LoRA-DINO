# Visualization

This folder contains scripts that generate **qualitative outputs** (painted detections / VR exports / Grad-CAM overlays) and optional packaging helpers.

## VR export (full val/test)

- `visualization/export_sardet_vr.sh`: export full-split VR outputs (metrics + predictions + images).

```bash
OUT_ROOT=artifacts/visualizations/VR SPLITS=val,test ENV_NAME=sar_lora_dino \
  bash visualization/export_sardet_vr.sh \
    --name <TAG> \
    --config <CONFIG.py> \
    --checkpoint <CKPT.pth>
```

## Quick visualization (subset or full)

- `visualization/visualize_sardet.sh`: quick painted detections using `visualization/mmdet_test_export.py` + `--show-dir`.

```bash
bash visualization/visualize_sardet.sh \
  --config <CONFIG.py> \
  --checkpoint <CKPT.pth> \
  --out-dir artifacts/visualizations/vis_demo \
  --split val \
  --num-images 50
```

## Packaging helpers (optional)

- `visualization/run_fullsplit_vr_pack.py`: export (optional) + pack a full split into a single zip.
- `visualization/repack_vr_sample500_from_subset.py`: repack VR images into aligned “sample500” zips based on subset JSONs.
- `visualization/export_vr_sample500_lora_bundle.py`: export + pack VR zips for a LoRA experiment list (paper tooling).
- `visualization/export_gradcam_sample500_bundle.py`: export + pack Grad-CAM overlays on a shared val-500 subset.

## Under the hood

- `visualization/mmdet_test_export.py`: one-shot test runner that writes `metrics.json`, `predictions.pkl`, and `vis/`.
- `visualization/mmdet_gradcam_export.py`: Grad-CAM-like export utility.
