# Reproduce

This document describes a minimal workflow to reproduce the main SAR-LoRA-DINO results on SARDet-100K.

## 1) Install

Follow `docs/INSTALL.md`.

## 2) Dataset

Follow `docs/DATASET.md` and set `SARDET100K_ROOT`.

## 3) (Optional) Pretrained download

If you want to pre-fetch timm weights (instead of downloading at first training run):

```bash
ENV_NAME=sar_lora_dino bash scripts/download_pretrained.sh
```

## 4) Smoke test

```bash
ENV_NAME=sar_lora_dino bash scripts/run_sardet_smoke.sh
```

## 5) Train + eval (paper configs)

Main LoRA setting (frozen backbone, LoRA on `mlp.fc1`+`mlp.fc2`):

```bash
EVAL_SPLITS=val,test ENV_NAME=sar_lora_dino bash scripts/run_sardet_full_cfg.sh \
  --config configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py \
  --work-dir artifacts/work_dirs/E0002_full_seed0 \
  --gpus 4 \
  --seed 0
```

Best LoRA variant (fc2-only + stage3 fine-tune):

```bash
EVAL_SPLITS=val,test ENV_NAME=sar_lora_dino bash scripts/run_sardet_full_cfg.sh \
  --config configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc2_ft_stage3_sardet100k.py \
  --work-dir artifacts/work_dirs/E0016_full \
  --gpus 4 \
  --seed 0
```

## 6) Tables

The experiment ledger and aggregated tables live under `artifacts/experiments/`:

```bash
python scripts/export_experiment_results.py
```

