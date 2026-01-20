# SAR-LoRA-DINO

Parameter-Efficient Fine-Tuning (PEFT) for **SAR object detection** with a **DINOv3-pretrained ConvNeXt-S** backbone, built on **MMDetection 3.x**.

Core idea: **freeze the DINOv3 ConvNeXt-S backbone**, inject **LoRA** into the ConvNeXt MLP (`mlp.fc1` / `mlp.fc2`, default `r=16, α=32, dropout=0`), and train **RetinaNet + FPN** on **SARDet-100K**.

## Highlights

- **Backbone frozen + LoRA on MLP**: small trainable footprint vs full fine-tune (see `artifacts/experiments/experiment.md`).
- **MMDetection project/plugin style**: `custom_imports` loads `sar_lora_dino` modules; no need to fork MMDet.
- **Paper-facing configs**: stable, clean names under `configs/sar_lora_dino/` (baselines + LoRA variants).

## Quick links

- Install: `docs/INSTALL.md`
- Dataset: `docs/DATASET.md`
- Training / Eval / Export: `docs/TRAINING.md`, `docs/EVALUATION.md`, `docs/EXPORT.md`
- Reproduce: `docs/REPRODUCE.md`
- Model Zoo: `docs/MODEL_ZOO.md`

## Results (SARDet-100K Val, COCO bbox mAP)

Authoritative table (all runs/splits): `artifacts/experiments/experiment_results.tsv`

| Setting | Val mAP | Trainable Params (M) | Config |
| --- | ---:| ---:| --- |
| DINOv3 ConvNeXt-S linear probe (frozen) | 0.419 ± 0.006 (3 seeds) | 9.403 | `configs/sar_lora_dino/retinanet_dinov3_convnexts_linear_sardet100k.py` |
| LoRA r=16, fc1+fc2 (frozen backbone) | 0.523 ± 0.028 (3 seeds) | 11.569 | `configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py` |
| LoRA r=16, fc2-only + fine-tune stage3 | 0.569 | 18.876 | `configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc2_ft_stage3_sardet100k.py` |
| DINOv3 ConvNeXt-S full fine-tune | 0.572 | 58.856 | `configs/sar_lora_dino/retinanet_dinov3_convnexts_full_ft_sardet100k.py` |

## Installation

See `docs/INSTALL.md`.

## Data preparation

See `docs/DATASET.md`.

## Training & evaluation (example)

Train:

```bash
ENV_NAME=sar_lora_dino bash tools/train.sh \
  configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py \
  --work-dir artifacts/work_dirs/E0002_full_seed0
```

Eval to JSON:

```bash
ENV_NAME=sar_lora_dino bash tools/test.sh \
  --config configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py \
  --checkpoint /path/to/checkpoint.pth \
  --work-dir artifacts/work_dirs/eval_only \
  --out-json artifacts/work_dirs/eval_only/val_metrics.json
```

## Citation

- Software: `CITATION.cff` / `CITATION.bib`
- Dataset: SARDet-100K (see `CITATION.bib`)

