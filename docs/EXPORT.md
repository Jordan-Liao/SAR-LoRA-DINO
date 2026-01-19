# Export / Visualization

Visualization code lives under `visualization/` and writes outputs under `artifacts/visualizations/` by default.

## Painted detections (qualitative)

```bash
bash visualization/visualize_sardet.sh \
  --config configs/sardet100k/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py \
  --checkpoint /path/to/checkpoint.pth \
  --out-dir artifacts/visualizations/painted
```

## VR export (preds + packaged visuals)

```bash
OUT_ROOT=artifacts/visualizations/VR SPLITS=val,test ENV_NAME=sar_lora_dino \
  bash visualization/export_sardet_vr.sh \
    --name E0002_lora \
    --config configs/sardet100k/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py \
    --checkpoint /path/to/checkpoint.pth
```

## Grad-CAM (sample500 bundle)

See `visualization/export_gradcam_sample500_bundle.py` and `docs/RESULTS_BUNDLE.md`.

