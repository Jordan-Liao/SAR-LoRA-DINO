# Configs Explained

All runnable MMDetection configs live under `configs/`.

## Layout

- `configs/_base_/`: MMDetection-style base templates (Apache-2.0; see `THIRD_PARTY_NOTICES.md`).
- `configs/sardet100k/`: SARDet-100K experiment configs grouped by intent.

## SARDet-100K config groups

- `configs/sardet100k/smoke/`
  - Small subset smoke tests (fast end-to-end validation).
- `configs/sardet100k/dinov3_baselines/`
  - DINOv3 (ConvNeXt) baselines: linear-probe vs full fine-tune.
- `configs/sardet100k/dinov3_lora/`
  - Main method: `Dinov3TimmConvNeXtLoRA` + RetinaNet.
- `configs/sardet100k/dinov3_lora_ablation/`
  - LoRA ablations (rank, target layers, unfreeze stages, no-pretrain).
- `configs/sardet100k/sup_baselines/`
  - Supervised ConvNeXt baselines.

## How custom modules are imported

Configs that use SAR-LoRA-DINO modules define:

- `custom_imports = dict(imports=[...], allow_failed_imports=False)`

so MMDetection can find `sar_lora_dino.*` without modifying `PYTHONPATH`.

## Overriding options at runtime

Runner scripts forward MMEngine overrides via `--cfg-options`:

```bash
conda run -n sar_lora_dino python tools/train.py \
  configs/sardet100k/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py \
  --work-dir artifacts/work_dirs/tmp_run \
  --cfg-options randomness.seed=0 train_cfg.max_epochs=1
```
