# SAR-LoRA-DINO

Parameter-Efficient Fine-Tuning (PEFT) for **SAR object detection** with a **DINOv3-pretrained ConvNeXt-S** backbone, built on **MMDetection 3.x**.

Core idea: **freeze the DINOv3 ConvNeXt-S backbone**, inject **LoRA** into the ConvNeXt MLP (`mlp.fc1` / `mlp.fc2`, default `r=16, α=32, dropout=0`), and train **RetinaNet + FPN** on **SARDet-100K**.

## Highlights

- **Backbone frozen + LoRA on MLP**: small trainable footprint vs full fine-tune (see `artifacts/experiments/experiment.md`).
- **MMDetection project/plugin style**: `custom_imports` loads `sar_lora_dino` modules; no need to fork MMDet.
- **Paper-facing configs**: stable, clean names under `configs/sar_lora_dino/` (baselines + LoRA variants).

## Quickstart (recommended)

```bash
# 1) Create env (MMDet 3.x + this repo)
ENV_NAME=sar_lora_dino bash scripts/setup_env.sh

# 2) Link dataset (pick one)
export SARDET100K_ROOT=/path/to/SARDet_100K
# or:
bash scripts/setup_sardet_dataset.sh

# 3) Smoke test (end-to-end, tiny subset)
bash scripts/run_sardet_smoke.sh

# 4) Train + eval (paper config example)
EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh \
  --config configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py \
  --work-dir artifacts/work_dirs/E0002_full_seed0 \
  --gpus 1 \
  --seed 0
```

Notes:
- DINOv3 ConvNeXt weights are pulled by `timm` (`pretrained=True`). To pre-fetch: `bash scripts/download_pretrained.sh`.
- Outputs are written under `artifacts/work_dirs/` and `artifacts/visualizations/` (large files are gitignored).

## Docs

- Install: `docs/INSTALL.md`
- Dataset: `docs/DATASET.md`
- Training / Eval / Export: `docs/TRAINING.md`, `docs/EVALUATION.md`, `docs/EXPORT.md`
- Reproduce: `docs/REPRODUCE.md`
- Model Zoo: `docs/MODEL_ZOO.md`

## Configs

- `configs/sar_lora_dino/`: paper-facing configs (stable names for baselines + main variants).
- `configs/sardet100k/`: expanded configs used by the experiment ledger (ablations, smoke helpers, etc.).

## Results (SARDet-100K Val, COCO bbox mAP)

Authoritative table (all runs/splits): `artifacts/experiments/experiment_results.tsv`

| Setting | Val mAP | Trainable Params (M) | Config |
| --- | ---:| ---:| --- |
| DINOv3 ConvNeXt-S linear probe (frozen) | 0.419 ± 0.006 (3 seeds) | 9.403 | `configs/sar_lora_dino/retinanet_dinov3_convnexts_linear_sardet100k.py` |
| LoRA r=16, fc1+fc2 (frozen backbone) | 0.523 ± 0.028 (3 seeds) | 11.569 | `configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py` |
| LoRA r=16, fc2-only + fine-tune stage3 | 0.569 | 18.876 | `configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc2_ft_stage3_sardet100k.py` |
| DINOv3 ConvNeXt-S full fine-tune | 0.572 | 58.856 | `configs/sar_lora_dino/retinanet_dinov3_convnexts_full_ft_sardet100k.py` |

## Citation

- Software: `CITATION.cff` / `CITATION.bib`
- Dataset: SARDet-100K (see `CITATION.bib`)
