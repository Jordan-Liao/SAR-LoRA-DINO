# Getting Started

## 1) Environment

The repo assumes an MMDetection 3.x environment. Example (CUDA 11.8):

```bash
conda create -n sar_lora_dino python=3.8 -y
conda activate sar_lora_dino

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install "mmengine==0.8.4"
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"

pip install -r requirements.txt
pip install -e .
```

Optional: you can also run `ENV_NAME=sar_lora_dino bash scripts/setup_env.sh` as a convenience wrapper.

Sanity check:

```bash
ENV_NAME=sar_lora_dino bash scripts/verify_env.sh
```

## 2) Dataset

Download SARDet-100K and point the code to it by either:

- Setting env var:
  - `export SARDET100K_ROOT=/path/to/SARDet_100K`
- Or symlinking into the repo:
  - `ln -s /path/to/SARDet_100K data/sardet100k`

Expected layout:

```
SARDet_100K/
  Annotations/{train,val,test}.json
  JPEGImages/{train,val,test}/
```

## 3) Smoke run (end-to-end)

```bash
bash scripts/run_sardet_smoke.sh
```

## 4) Smoke run (any config)

```bash
bash scripts/run_sardet_smoke_cfg.sh \
  --config configs/sardet100k/smoke/retinanet_r50_sardet_smoke.py \
  --work-dir artifacts/work_dirs/sardet_smoke_cfg_test
```

## 5) Full train + eval (any config)

```bash
bash scripts/run_sardet_full_cfg.sh \
  --config configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py \
  --work-dir artifacts/work_dirs/E0002_full_seed0 \
  --gpus 4 \
  --seed 0
```

Set `EVAL_SPLITS=val,test` if you also want test evaluation.

## 6) Visualization / VR export

```bash
OUT_ROOT=artifacts/visualizations/VR SPLITS=val,test ENV_NAME=sar_lora_dino \
  bash visualization/export_sardet_vr.sh \
    --name E0002_lora \
    --config configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py \
    --checkpoint artifacts/work_dirs/E0002_full_seed0/best_coco_bbox_mAP_epoch_12.pth
```

## Notes

- Offline mode: set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` if your environment has no internet.
- Outputs are written under `artifacts/work_dirs/` (gitignored).
- Large checkpoints belong in `artifacts/weights/` (gitignored; see `artifacts/weights/README.md`).
