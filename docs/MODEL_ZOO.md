# Model Zoo

This repo keeps **configs + paper-facing metrics tables** in git, but does **not** vendor large artifacts
(checkpoints / full logs / image dumps). See `docs/ARTIFACTS_AND_RELEASES.md`.

## What’s included in git

- Configs: `configs/`
- Experiment ledger: `artifacts/experiments/experiment.md`
- Aggregated results:
  - `artifacts/experiments/experiment_results.tsv`
  - `artifacts/experiments/experiment_results.md`
- Lightweight metrics JSONs (per run): `artifacts/work_dirs/**/{smoke,val,test}_metrics.json`

## Key SARDet-100K results (Val)

Numbers below are **COCO bbox mAP** on SARDet-100K Val. For the full table (all runs/splits), use
`artifacts/experiments/experiment_results.tsv`.

| ID | Setting | Config | Val mAP | Notes |
| --- | --- | --- | --- | --- |
| E0002 | DINOv3 ConvNeXt-S + LoRA (r=16, fc1+fc2), backbone frozen | `configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py` | `0.523 ± 0.028` (3 seeds) | Metrics: `artifacts/work_dirs/E0002_full_seed*/val_metrics.json` |
| E0003 | DINOv3 ConvNeXt-S linear probe (no LoRA), backbone frozen | `configs/sar_lora_dino/retinanet_dinov3_convnexts_linear_sardet100k.py` | `0.419 ± 0.006` (3 seeds) | Metrics: `artifacts/work_dirs/E0003_full_seed*/val_metrics.json` |
| E0019 | DINOv3 ConvNeXt-S full fine-tune (no LoRA) | `configs/sar_lora_dino/retinanet_dinov3_convnexts_full_ft_sardet100k.py` | `0.572` | Metrics: `artifacts/work_dirs/E0019_full/val_metrics.json` |
| E0020 | ConvNeXt-S full fine-tune (supervised ImageNet) | `configs/sar_lora_dino/retinanet_convnexts_full_ft_sardet100k.py` | `0.594` | Metrics: `artifacts/work_dirs/E0020_full/val_metrics.json` |

## LoRA ablations (Val/Test)

| ID | Setting | Config | Val mAP | Test mAP | Metrics |
| --- | --- | --- | --- | --- | --- |
| E0004 | LoRA r=16, fc2-only | `configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc2_sardet100k.py` | `0.507` | `0.483` | `artifacts/work_dirs/E0004_full/{val,test}_metrics.json` |
| E0012 | LoRA r=4, fc2-only | `configs/sardet100k/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_1x_sardet_bs64_amp.py` | `0.483` | `0.461` | `artifacts/work_dirs/E0012_full/{val,test}_metrics.json` |
| E0013 | LoRA r=8, fc2-only | `configs/sardet100k/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_1x_sardet_bs64_amp.py` | `0.489` | `0.464` | `artifacts/work_dirs/E0013_full/{val,test}_metrics.json` |
| E0014 | LoRA r=4, fc1+fc2 + unfreeze stage3 | `configs/sardet100k/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` | `0.515` | `0.487` | `artifacts/work_dirs/E0014_full/{val,test}_metrics.json` |
| E0015 | LoRA r=8, fc1+fc2 + unfreeze stage3 | `configs/sardet100k/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` | `0.534` | `0.505` | `artifacts/work_dirs/E0015_full/{val,test}_metrics.json` |
| E0016 | LoRA r=16, fc2-only + fine-tune stage3 | `configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc2_ft_stage3_sardet100k.py` | `0.569` | `0.543` | `artifacts/work_dirs/E0016_full/{val,test}_metrics.json` |
| E0017 | LoRA r=4, fc2-only + unfreeze stage3 | `configs/sardet100k/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` | `0.558` | `0.534` | `artifacts/work_dirs/E0017_full/{val,test}_metrics.json` |
| E0018 | LoRA r=8, fc2-only + unfreeze stage3 | `configs/sardet100k/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` | `0.562` | `0.538` | `artifacts/work_dirs/E0018_full/{val,test}_metrics.json` |
