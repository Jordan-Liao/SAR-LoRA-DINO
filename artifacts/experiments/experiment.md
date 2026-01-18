# Experiments

## Overview
- Goal: Evaluate Dinov3TimmConvNeXtLoRA + RetinaNet on SARDet-100K with consistent baselines, ablations, and saved artifacts.
- Baseline: Dinov3 ConvNeXt-S linear probe (no LoRA; frozen backbone) (E0003)
- Primary model: Dinov3 ConvNeXt-S LoRA r=16 (fc1+fc2) + RetinaNet (E0002)
- Execution: each experiment table includes runnable “Smoke cmd” / “Full cmd”. Recommended workflow is smoke first (`scripts/run_sardet_smoke*.sh`), then full (`scripts/run_sardet_full_cfg.sh`) with your preferred scheduler/tmux.
- Monitor: check logs under `artifacts/work_dirs/<exp>/` (or your scheduler logs, if used).

## Experiments

### E0001: SARDet-100K smoke (RetinaNet R50-FPN)
| Field | Value |
| --- | --- |
| Objective | Validate end-to-end train + eval on SARDet-100K via small COCO subsets. |
| Baseline | RetinaNet R50-FPN (standard ImageNet-pretrained ResNet-50). |
| Model | RetinaNet R50-FPN, `num_classes=6`, `test_cfg.score_thr=0.0` to avoid empty-result eval on tiny runs. |
| Weights | Backbone: `torchvision://resnet50` (from `mmdet_toolkit/configs/_base_/models/retinanet_r50_fpn.py`). |
| Code path | `scripts/run_sardet_smoke.sh`, `scripts/mmdet_train.py`, `scripts/mmdet_test_to_json.py`, `scripts/make_coco_subset.py`, `mmdet_toolkit/local_configs/SARDet/smoke/retinanet_r50_sardet_smoke.py` |
| Params | Train subset: 200 images; Val subset: 50 images; Epochs: 1; Img scale: (800,800); Env: `sar_lora_dino`; GPU: auto-selected via `CUDA_VISIBLE_DEVICES` if unset. |
| Metrics (must save) | `artifacts/work_dirs/sardet_smoke/smoke_metrics.json` (`coco/bbox_mAP`, `coco/bbox_mAP_50`, ...) |
| Checks | Command exits 0; metrics JSON exists and includes `coco/bbox_mAP`. |
| VRAM | ~8 GB (single GPU) |
| Time/epoch | ~1 min |
| Total time | ~2 min |
| Single-GPU script | `bash scripts/run_sardet_smoke.sh` |
| Multi-GPU script | `MAX_EPOCHS=1 TRAIN_NUM_IMAGES=200 VAL_NUM_IMAGES=50 bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/smoke/retinanet_r50_sardet_smoke.py --work-dir artifacts/work_dirs/sardet_smoke_dist --gpus 2 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke.sh` |
| Full cmd | `bash scripts/run_sardet_smoke.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/sardet_smoke/smoke_train.log`, `artifacts/work_dirs/sardet_smoke/smoke_test.log` |
| Artifacts | `data/sardet_subsets/train_200.json`, `data/sardet_subsets/val_50.json`, `artifacts/work_dirs/sardet_smoke/epoch_1.pth`, `artifacts/work_dirs/sardet_smoke/smoke_metrics.json` |
| Results | `coco/bbox_mAP = 0.003`, `coco/bbox_mAP_50 = 0.008` (see `artifacts/work_dirs/sardet_smoke/smoke_metrics.json`). |


### E0002: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=16, fc1+fc2) (ours)
| Field | Value |
| --- | --- |
| Objective | Main method: freeze backbone, train LoRA adapters injected into `mlp.fc1/fc2`, and train FPN/head. |
| Baseline | E0003 (Dinov3 linear probe, no LoRA). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `model.backbone.model_name=convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py`, `mmdet_toolkit/sar_lora_dino/models/backbones/dinov3_timm_convnext_lora.py`, `mmdet_toolkit/sar_lora_dino/models/lora.py`, `scripts/run_sardet_smoke_cfg.sh`, `scripts/mmdet_test_to_json.py`, `scripts/count_trainable_params.py` |
| Params | LoRA: target=`(mlp.fc1,mlp.fc2)`, r=16, alpha=32; backbone frozen; neck/head trained. Full runs use seeds {0,1,2} via `--cfg-options randomness.seed=<seed>`. |
| Metrics (must save) | For each seed: `artifacts/work_dirs/E0002_full_seed<seed>/val_metrics.json`, `artifacts/work_dirs/E0002_full_seed<seed>/test_metrics.json` (SARDet-100K Val/Test COCO bbox metrics). |
| Checks | `coco/bbox_mAP` exists; compute mean±std over seeds; record trainable params p(M). |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0002_full_seed0 --gpus 1 --seed 0` |
| Multi-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0002_full_seed0 --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0002_smoke` |
| Full cmd | `for s in 0 1 2; do EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0002_full_seed${s} --gpus 4 --seed ${s}; done` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0002_smoke/smoke_train.log`, `artifacts/work_dirs/E0002_smoke/smoke_test.log`, `artifacts/work_dirs/E0002_full_seed<seed>/train.log`, `artifacts/work_dirs/E0002_full_seed<seed>/test_val.log`, `artifacts/work_dirs/E0002_full_seed<seed>/test_test.log` |
| Artifacts | `artifacts/work_dirs/E0002_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0002_full_seed<seed>/*.pth`, `artifacts/work_dirs/E0002_full_seed<seed>/val_metrics.json`, `artifacts/work_dirs/E0002_full_seed<seed>/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.000` (see `artifacts/work_dirs/E0002_smoke/smoke_metrics.json`); trainable p(M)=11.569 (backbone=2.166, neck=4.475, head=4.928). Full Val `coco/bbox_mAP`: seed0=0.539, seed1=0.491, seed2=0.539 (mean=0.523±0.028) (see `artifacts/work_dirs/E0002_full_seed*/val_metrics.json`). |


### E0003: Dinov3 ConvNeXt-S + RetinaNet (Linear probe, no LoRA)
| Field | Value |
| --- | --- |
| Objective | Control group: freeze backbone, train FPN/head, no LoRA. |
| Baseline | Compare against E0002 to isolate LoRA gain. |
| Model | `TimmConvNeXt` (Dinov3 weights) + RetinaNet. |
| Weights | `model.backbone.model_name=convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py`, `mmdet_toolkit/sar_lora_dino/models/backbones/timm_convnext.py`, `scripts/run_sardet_smoke_cfg.sh` |
| Params | Backbone frozen; neck/head trained; no LoRA modules. Full runs use seeds {0,1,2} via `--cfg-options randomness.seed=<seed>`. |
| Metrics (must save) | For each seed: `artifacts/work_dirs/E0003_full_seed<seed>/val_metrics.json` (SARDet-100K Val COCO bbox metrics). |
| Checks | `coco/bbox_mAP` exists; compute mean±std over seeds; record trainable params p(M). |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0003_full_seed0 --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0003_full_seed0 --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/dinov3_lp_smoke` |
| Full cmd | `for s in 0 1 2; do bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0003_full_seed${s} --gpus 4 --seed ${s}; done` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/dinov3_lp_smoke/smoke_train.log`, `artifacts/work_dirs/dinov3_lp_smoke/smoke_test.log`, `artifacts/work_dirs/E0003_full_seed<seed>/train.log`, `artifacts/work_dirs/E0003_full_seed<seed>/test_val.log` |
| Artifacts | `artifacts/work_dirs/dinov3_lp_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0003_full_seed<seed>/*.pth`, `artifacts/work_dirs/E0003_full_seed<seed>/val_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.000` (see `artifacts/work_dirs/dinov3_lp_smoke/smoke_metrics.json`); trainable p(M)=9.403 (backbone=0.000, neck=4.475, head=4.928). Full Val `coco/bbox_mAP`: seed0=0.413, seed1=0.419, seed2=0.425 (mean=0.419±0.006) (see `artifacts/work_dirs/E0003_full_seed*/val_metrics.json`). |


### E0004: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=16, fc2-only)
| Field | Value |
| --- | --- |
| Objective | Ablation: isolate LoRA injection position (fc2-only vs fc1+fc2). |
| Baseline | E0002 (fc1+fc2). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc2)`, r=16, alpha=32. |
| Metrics (must save) | `artifacts/work_dirs/E0004_full/val_metrics.json`, `artifacts/work_dirs/E0004_full/test_metrics.json` |
| Checks | Compare `coco/bbox_mAP` vs E0002; record p(M). |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0004_full --gpus 1 --seed 0` |
| Multi-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0004_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/dinov3_lora_fc2_smoke` |
| Full cmd | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0004_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/dinov3_lora_fc2_smoke/smoke_train.log`, `artifacts/work_dirs/dinov3_lora_fc2_smoke/smoke_test.log`, `artifacts/work_dirs/E0004_full/train.log`, `artifacts/work_dirs/E0004_full/test_val.log`, `artifacts/work_dirs/E0004_full/test_test.log` |
| Artifacts | `artifacts/work_dirs/dinov3_lora_fc2_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0004_full/*.pth`, `artifacts/work_dirs/E0004_full/val_metrics.json`, `artifacts/work_dirs/E0004_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.000` (see `artifacts/work_dirs/dinov3_lora_fc2_smoke/smoke_metrics.json`); trainable p(M)=10.486 (backbone=1.083, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.507` (see `artifacts/work_dirs/E0004_full/val_metrics.json`). |


### E0005: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=4, fc1+fc2)
| Field | Value |
| --- | --- |
| Objective | Ablation: LoRA rank sensitivity (r=4). |
| Baseline | E0002 (r=16). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc1,mlp.fc2)`, r=4, alpha=8. |
| Metrics (must save) | `artifacts/work_dirs/E0005_full/val_metrics.json`, `artifacts/work_dirs/E0005_full/test_metrics.json` |
| Checks | Compare `coco/bbox_mAP` vs r=8/16; record p(M). |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0005_full --gpus 1 --seed 0` |
| Multi-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0005_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0005_smoke` |
| Full cmd | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0005_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0005_smoke/smoke_*.log`, `artifacts/work_dirs/E0005_full/train.log`, `artifacts/work_dirs/E0005_full/test_val.log`, `artifacts/work_dirs/E0005_full/test_test.log` |
| Artifacts | `artifacts/work_dirs/E0005_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0005_full/*.pth`, `artifacts/work_dirs/E0005_full/val_metrics.json`, `artifacts/work_dirs/E0005_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.000` (see `artifacts/work_dirs/E0005_smoke/smoke_metrics.json`); trainable p(M)=9.944 (backbone=0.541, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.508` (see `artifacts/work_dirs/E0005_full/val_metrics.json`). |


### E0006: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=8, fc1+fc2)
| Field | Value |
| --- | --- |
| Objective | Ablation: LoRA rank sensitivity (r=8). |
| Baseline | E0002 (r=16). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc1,mlp.fc2)`, r=8, alpha=16. |
| Metrics (must save) | `artifacts/work_dirs/E0006_full/val_metrics.json`, `artifacts/work_dirs/E0006_full/test_metrics.json` |
| Checks | Compare `coco/bbox_mAP` vs r=4/16; record p(M). |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0006_full --gpus 1 --seed 0` |
| Multi-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0006_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0006_smoke` |
| Full cmd | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0006_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0006_smoke/smoke_*.log`, `artifacts/work_dirs/E0006_full/train.log`, `artifacts/work_dirs/E0006_full/test_val.log`, `artifacts/work_dirs/E0006_full/test_test.log` |
| Artifacts | `artifacts/work_dirs/E0006_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0006_full/*.pth`, `artifacts/work_dirs/E0006_full/val_metrics.json`, `artifacts/work_dirs/E0006_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.000` (see `artifacts/work_dirs/E0006_smoke/smoke_metrics.json`); trainable p(M)=10.486 (backbone=1.083, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.519` (see `artifacts/work_dirs/E0006_full/val_metrics.json`). |


### E0007: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=16, fc1+fc2, unfreeze stage3)
| Field | Value |
| --- | --- |
| Objective | Ablation: freeze strategy (freeze-all vs unfreeze last stage). |
| Baseline | E0002 (freeze-all). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc1,mlp.fc2)`, r=16, alpha=32; unfreeze stage3. |
| Metrics (must save) | `artifacts/work_dirs/E0007_full/val_metrics.json`, `artifacts/work_dirs/E0007_full/test_metrics.json` |
| Checks | Compare `coco/bbox_mAP` vs E0002; record p(M). |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0007_full --gpus 1 --seed 0` |
| Multi-GPU script | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0007_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0007_smoke` |
| Full cmd | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0007_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0007_smoke/smoke_*.log`, `artifacts/work_dirs/E0007_full/train.log`, `artifacts/work_dirs/E0007_full/test_val.log`, `artifacts/work_dirs/E0007_full/test_test.log` |
| Artifacts | `artifacts/work_dirs/E0007_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0007_full/*.pth`, `artifacts/work_dirs/E0007_full/val_metrics.json`, `artifacts/work_dirs/E0007_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.001` (see `artifacts/work_dirs/E0007_smoke/smoke_metrics.json`); trainable p(M)=12.872 (backbone=3.469, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.539` (see `artifacts/work_dirs/E0007_full/val_metrics.json`). |


### E0008: ConvNeXt-S (ImageNet-supervised) + RetinaNet (Linear probe, no LoRA)
| Field | Value |
| --- | --- |
| Objective | Baseline: supervised ConvNeXt-S with frozen backbone; train FPN/head; no LoRA. |
| Baseline | Compare against E0003 (Dinov3 linear probe) and E0002 (Dinov3-LoRA). |
| Model | `TimmConvNeXt` (model_name=`convnext_small`) + RetinaNet. |
| Weights | `model.backbone.model_name=convnext_small` (timm ImageNet-supervised; `pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py` |
| Params | Backbone frozen; neck/head trained. Set `EVAL_SPLITS=val,test` to also write `test_metrics.json`. |
| Metrics (must save) | `artifacts/work_dirs/E0008_full/val_metrics.json` (and `artifacts/work_dirs/E0008_full/test_metrics.json` if `EVAL_SPLITS=val,test`) |
| Checks | Record `coco/bbox_mAP` and trainable params p(M). |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0008_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0008_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0008_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0008_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0008_smoke/smoke_*.log`, `artifacts/work_dirs/E0008_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0008_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0008_full/*.pth`, `artifacts/work_dirs/E0008_full/val_metrics.json` |
| Results | Trainable p(M)=9.403 (backbone=0.000, neck=4.475, head=4.928); Smoke `coco/bbox_mAP = 0.002` (see `artifacts/work_dirs/E0008_smoke/smoke_metrics.json`); Full Val `coco/bbox_mAP = 0.401` (see `artifacts/work_dirs/E0008_full/val_metrics.json`). |


### E0010: Visualize detections (no LoRA)
| Field | Value |
| --- | --- |
| Objective | Export qualitative detection images without LoRA (e.g., linear probe checkpoint). |
| Baseline | Compare against E0011 (with LoRA). |
| Model | Any non-LoRA model checkpoint from E0003/E0008. |
| Weights | A finished checkpoint (e.g. `artifacts/work_dirs/E0003_full/best_*.pth`). |
| Code path | `visualization/visualize_sardet.sh`, `visualization/mmdet_test_export.py` |
| Params | Use `--show-dir` to write painted images. |
| Metrics (must save) | N/A |
| Checks | Output directory contains painted images. |
| VRAM | TBD |
| Time/epoch | N/A |
| Total time | TBD |
| Single-GPU script | `bash visualization/visualize_sardet.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --checkpoint <ckpt> --out-dir artifacts/work_dirs/E0010_vis` |
| Multi-GPU script | N/A |
| Smoke cmd | `bash visualization/visualize_sardet.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --checkpoint artifacts/work_dirs/dinov3_lp_smoke/epoch_1.pth --out-dir artifacts/work_dirs/E0010_vis_smoke` |
| Full cmd | `bash visualization/visualize_sardet.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_linear-probe_1x_sardet_bs64_amp.py --checkpoint artifacts/work_dirs/dinov3_lp_smoke/epoch_1.pth --out-dir artifacts/work_dirs/E0010_vis_smoke` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0010_vis*/**/*.log` |
| Artifacts | `artifacts/work_dirs/E0010_vis*/` |
| Results | Smoke wrote Val-50 visualizations under `artifacts/work_dirs/E0010_vis_smoke/20260107_152431/vis/` (no LoRA). |


### E0011: Visualize detections (with LoRA)
| Field | Value |
| --- | --- |
| Objective | Export qualitative detection images with LoRA (ours). |
| Baseline | Compare against E0010 (no LoRA). |
| Model | LoRA model checkpoint from E0002/E0004/E0005/E0006/E0007. |
| Weights | A finished checkpoint (e.g. `artifacts/work_dirs/E0002_full/best_*.pth`). |
| Code path | `visualization/visualize_sardet.sh`, `visualization/mmdet_test_export.py` |
| Params | Use the same split and `--show-dir` as E0010 for fair visuals. |
| Metrics (must save) | N/A |
| Checks | Output directory contains painted images. |
| VRAM | TBD |
| Time/epoch | N/A |
| Total time | TBD |
| Single-GPU script | `bash visualization/visualize_sardet.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py --checkpoint <ckpt> --out-dir artifacts/work_dirs/E0011_vis` |
| Multi-GPU script | N/A |
| Smoke cmd | `bash visualization/visualize_sardet.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py --checkpoint artifacts/work_dirs/E0002_smoke/epoch_1.pth --out-dir artifacts/work_dirs/E0011_vis_smoke` |
| Full cmd | `bash visualization/visualize_sardet.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py --checkpoint artifacts/work_dirs/E0002_smoke/epoch_1.pth --out-dir artifacts/work_dirs/E0011_vis_smoke` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0011_vis*/**/*.log` |
| Artifacts | `artifacts/work_dirs/E0011_vis*/` |
| Results | Smoke wrote Val-50 visualizations under `artifacts/work_dirs/E0011_vis_smoke/20260107_152509/vis/` (with LoRA). |


### E0019: Dinov3 ConvNeXt-S + RetinaNet (Full fine-tune, no LoRA)
| Field | Value |
| --- | --- |
| Objective | Baseline: train backbone + neck/head on SARDet-100K without LoRA (full fine-tune). |
| Baseline | Compare against E0003 (linear probe) and E0002 (LoRA). |
| Model | `TimmConvNeXt` (Dinov3 weights) + RetinaNet. |
| Weights | `model.backbone.model_name=convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_1x_sardet_bs64_amp.py`, `mmdet_toolkit/sar_lora_dino/models/backbones/timm_convnext.py` |
| Params | Backbone trainable; neck/head trained; no LoRA modules. |
| Metrics (must save) | `artifacts/work_dirs/E0019_full/val_metrics.json` |
| Checks | Record `coco/bbox_mAP` and trainable params p(M). |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0019_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0019_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0019_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0019_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0019_smoke/smoke_*.log`, `artifacts/work_dirs/E0019_full/train.log`, `artifacts/work_dirs/E0019_full/test_val.log`, `artifacts/work_dirs/E0019_full/*/*.log` |
| Artifacts | `artifacts/work_dirs/E0019_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0019_full/*.pth`, `artifacts/work_dirs/E0019_full/val_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.002` (see `artifacts/work_dirs/E0019_smoke/smoke_metrics.json`); trainable p(M)=58.856 (backbone=49.453, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.572` (see `artifacts/work_dirs/E0019_full/val_metrics.json`); best ckpt: `artifacts/work_dirs/E0019_full/best_coco_bbox_mAP_epoch_12.pth`. VR export (val+test): `artifacts/visualizations/VR/E0019_full-ft/val/` and `artifacts/visualizations/VR/E0019_full-ft/test/` (see `metrics.json`). |


### E0020: ConvNeXt-S (ImageNet-supervised) + RetinaNet (Full fine-tune, no LoRA)
| Field | Value |
| --- | --- |
| Objective | Baseline: train ImageNet-supervised ConvNeXt-S backbone + neck/head on SARDet-100K (full fine-tune). |
| Baseline | Compare against E0008 (linear probe) and E0002 (Dinov3-LoRA). |
| Model | `TimmConvNeXt` (ImageNet-supervised weights) + RetinaNet. |
| Weights | `model.backbone.model_name=convnext_small` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_full-ft_1x_sardet_bs64_amp.py`, `mmdet_toolkit/sar_lora_dino/models/backbones/timm_convnext.py` |
| Params | Backbone trainable; neck/head trained; no LoRA modules. |
| Metrics (must save) | `artifacts/work_dirs/E0020_full/val_metrics.json` |
| Checks | Record `coco/bbox_mAP` and trainable params p(M). |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_full-ft_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0020_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_full-ft_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0020_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_full-ft_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0020_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/sup_baselines/retinanet_timm-convnext-small_full-ft_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0020_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0020_smoke/smoke_*.log`, `artifacts/work_dirs/E0020_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0020_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0020_full/*.pth`, `artifacts/work_dirs/E0020_full/val_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.009` (see `artifacts/work_dirs/E0020_smoke/smoke_metrics.json`); trainable p(M)=58.856 (backbone=49.453, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.594` (see `artifacts/work_dirs/E0020_full/val_metrics.json`). |


### E0012: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=4, fc2-only)
| Field | Value |
| --- | --- |
| Objective | Ablation: target=`(mlp.fc2)` with low-rank LoRA (r=4). |
| Baseline | Compare against E0004 (fc2-only r=16) and E0005 (fc1+fc2 r=4). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc2)`, r=4, alpha=8. |
| Metrics (must save) | `artifacts/work_dirs/E0012_full/val_metrics.json`, `artifacts/work_dirs/E0012_full/test_metrics.json` |
| Checks | Record `coco/bbox_mAP` and p(M); compare vs E0004/E0005. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0012_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0012_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0012_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0012_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0012_smoke/smoke_*.log`, `artifacts/work_dirs/E0012_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0012_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0012_full/*.pth`, `artifacts/work_dirs/E0012_full/val_metrics.json`, `artifacts/work_dirs/E0012_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.000` (see `artifacts/work_dirs/E0012_smoke/smoke_metrics.json`); trainable p(M)=9.674 (backbone=0.271, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.483` (see `artifacts/work_dirs/E0012_full/val_metrics.json`). |


### E0013: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=8, fc2-only)
| Field | Value |
| --- | --- |
| Objective | Ablation: target=`(mlp.fc2)` with mid-rank LoRA (r=8). |
| Baseline | Compare against E0004 (fc2-only r=16) and E0006 (fc1+fc2 r=8). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc2)`, r=8, alpha=16. |
| Metrics (must save) | `artifacts/work_dirs/E0013_full/val_metrics.json`, `artifacts/work_dirs/E0013_full/test_metrics.json` |
| Checks | Record `coco/bbox_mAP` and p(M); compare vs E0004/E0006. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0013_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0013_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0013_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0013_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0013_smoke/smoke_*.log`, `artifacts/work_dirs/E0013_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0013_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0013_full/*.pth`, `artifacts/work_dirs/E0013_full/val_metrics.json`, `artifacts/work_dirs/E0013_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.000` (see `artifacts/work_dirs/E0013_smoke/smoke_metrics.json`); trainable p(M)=9.944 (backbone=0.541, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.489` (see `artifacts/work_dirs/E0013_full/val_metrics.json`). |


### E0014: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=4, fc1+fc2, unfreeze stage3)
| Field | Value |
| --- | --- |
| Objective | Ablation: unfreeze stage3 with r=4 (fc1+fc2). |
| Baseline | Compare against E0005 (r=4 freeze-all). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc1,mlp.fc2)`, r=4, alpha=8; `unfreeze_stages=(3,)`. |
| Metrics (must save) | `artifacts/work_dirs/E0014_full/val_metrics.json`, `artifacts/work_dirs/E0014_full/test_metrics.json` |
| Checks | Record `coco/bbox_mAP` and p(M); compare vs E0005. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0014_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0014_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0014_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0014_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0014_smoke/smoke_*.log`, `artifacts/work_dirs/E0014_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0014_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0014_full/*.pth`, `artifacts/work_dirs/E0014_full/val_metrics.json`, `artifacts/work_dirs/E0014_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.000` (see `artifacts/work_dirs/E0014_smoke/smoke_metrics.json`); trainable p(M)=11.248 (backbone=1.845, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.515` (see `artifacts/work_dirs/E0014_full/val_metrics.json`). |


### E0015: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=8, fc1+fc2, unfreeze stage3)
| Field | Value |
| --- | --- |
| Objective | Ablation: unfreeze stage3 with r=8 (fc1+fc2). |
| Baseline | Compare against E0006 (r=8 freeze-all). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc1,mlp.fc2)`, r=8, alpha=16; `unfreeze_stages=(3,)`. |
| Metrics (must save) | `artifacts/work_dirs/E0015_full/val_metrics.json`, `artifacts/work_dirs/E0015_full/test_metrics.json` |
| Checks | Record `coco/bbox_mAP` and p(M); compare vs E0006. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0015_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0015_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0015_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc1fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0015_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0015_smoke/smoke_*.log`, `artifacts/work_dirs/E0015_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0015_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0015_full/*.pth`, `artifacts/work_dirs/E0015_full/val_metrics.json`, `artifacts/work_dirs/E0015_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.004` (see `artifacts/work_dirs/E0015_smoke/smoke_metrics.json`); trainable p(M)=11.789 (backbone=2.386, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.534` (see `artifacts/work_dirs/E0015_full/val_metrics.json`). |


### E0016: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=16, fc2-only, unfreeze stage3)
| Field | Value |
| --- | --- |
| Objective | Ablation: combine fc2-only LoRA with unfreeze stage3 (r=16). |
| Baseline | Compare against E0004 (fc2-only freeze-all) and E0007 (fc1+fc2 unfreeze stage3). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc2)`, r=16, alpha=32; `unfreeze_stages=(3,)`. |
| Metrics (must save) | `artifacts/work_dirs/E0016_full/val_metrics.json`, `artifacts/work_dirs/E0016_full/test_metrics.json` |
| Checks | Record `coco/bbox_mAP` and p(M); compare vs E0004/E0007. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0016_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0016_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0016_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0016_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0016_smoke/smoke_*.log`, `artifacts/work_dirs/E0016_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0016_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0016_full/*.pth`, `artifacts/work_dirs/E0016_full/val_metrics.json`, `artifacts/work_dirs/E0016_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.002` (see `artifacts/work_dirs/E0016_smoke/smoke_metrics.json`); trainable p(M)=18.876 (backbone=9.473, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.569` (see `artifacts/work_dirs/E0016_full/val_metrics.json`). |


### E0017: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=4, fc2-only, unfreeze stage3)
| Field | Value |
| --- | --- |
| Objective | Ablation: fc2-only LoRA with unfreeze stage3 (r=4). |
| Baseline | Compare against E0012 (fc2-only r=4 freeze-all) and E0014 (fc1+fc2 r=4 unfreeze stage3). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc2)`, r=4, alpha=8; `unfreeze_stages=(3,)`. |
| Metrics (must save) | `artifacts/work_dirs/E0017_full/val_metrics.json`, `artifacts/work_dirs/E0017_full/test_metrics.json` |
| Checks | Record `coco/bbox_mAP` and p(M); compare vs E0012/E0014. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0017_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0017_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0017_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r4_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0017_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0017_smoke/smoke_*.log`, `artifacts/work_dirs/E0017_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0017_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0017_full/*.pth`, `artifacts/work_dirs/E0017_full/val_metrics.json`, `artifacts/work_dirs/E0017_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.002` (see `artifacts/work_dirs/E0017_smoke/smoke_metrics.json`); trainable p(M)=18.064 (backbone=8.661, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.558` (see `artifacts/work_dirs/E0017_full/val_metrics.json`). |


### E0018: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=8, fc2-only, unfreeze stage3)
| Field | Value |
| --- | --- |
| Objective | Ablation: fc2-only LoRA with unfreeze stage3 (r=8). |
| Baseline | Compare against E0013 (fc2-only r=8 freeze-all) and E0015 (fc1+fc2 r=8 unfreeze stage3). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc2)`, r=8, alpha=16; `unfreeze_stages=(3,)`. |
| Metrics (must save) | `artifacts/work_dirs/E0018_full/val_metrics.json`, `artifacts/work_dirs/E0018_full/test_metrics.json` |
| Checks | Record `coco/bbox_mAP` and p(M); compare vs E0013/E0015. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0018_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0018_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0018_smoke` |
| Full cmd | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r8_fc2_unfreeze-stage3_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0018_full --gpus 4 --seed 0` |
| Smoke | [x] |
| Full | [x] |
| Logs | `artifacts/work_dirs/E0018_smoke/smoke_*.log`, `artifacts/work_dirs/E0018_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0018_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0018_full/*.pth`, `artifacts/work_dirs/E0018_full/val_metrics.json`, `artifacts/work_dirs/E0018_full/test_metrics.json` |
| Results | Smoke `coco/bbox_mAP = 0.002` (see `artifacts/work_dirs/E0018_smoke/smoke_metrics.json`); trainable p(M)=18.335 (backbone=8.932, neck=4.475, head=4.928). Full Val `coco/bbox_mAP = 0.562` (see `artifacts/work_dirs/E0018_full/val_metrics.json`). |


### E0021: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=16, fc1-only)
| Field | Value |
| --- | --- |
| Objective | Ablation: fc1-only LoRA (r=16), to pair with existing fc2-only (E0004) and fc1+fc2 (E0002). |
| Baseline | Compare against E0002 (fc1+fc2) and E0004 (fc2-only). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=True`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc1)`, r=16, alpha=32; backbone frozen; neck/head trained. |
| Metrics (must save) | `artifacts/work_dirs/E0021_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0021_full/val_metrics.json`, `artifacts/work_dirs/E0021_full/test_metrics.json` |
| Checks | Smoke/full commands exit 0; metrics JSON exists and includes `coco/bbox_mAP`. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0021_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0021_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0021_smoke` |
| Full cmd | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0021_full --gpus 4 --seed 0` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `artifacts/work_dirs/E0021_smoke/smoke_*.log`, `artifacts/work_dirs/E0021_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0021_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0021_full/*.pth`, `artifacts/work_dirs/E0021_full/val_metrics.json`, `artifacts/work_dirs/E0021_full/test_metrics.json` |
| Results |  |


### E0022: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=16, fc1+fc2, no pretrain)
| Field | Value |
| --- | --- |
| Objective | No-pretrain setting: `pretrained=False` (random init) with LoRA r=16 injected into fc1+fc2. |
| Baseline | Compare against E0002 (pretrained=True) and E0025 (no LoRA, no pretrain). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=False`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_nopretrain_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc1,mlp.fc2)`, r=16, alpha=32; backbone frozen; neck/head trained; `pretrained=False`. |
| Metrics (must save) | `artifacts/work_dirs/E0022_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0022_full/val_metrics.json`, `artifacts/work_dirs/E0022_full/test_metrics.json` |
| Checks | Smoke/full commands exit 0; metrics JSON exists. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0022_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0022_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0022_smoke` |
| Full cmd | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1fc2_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0022_full --gpus 4 --seed 0` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `artifacts/work_dirs/E0022_smoke/smoke_*.log`, `artifacts/work_dirs/E0022_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0022_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0022_full/*.pth`, `artifacts/work_dirs/E0022_full/val_metrics.json`, `artifacts/work_dirs/E0022_full/test_metrics.json` |
| Results |  |


### E0023: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=16, fc2-only, no pretrain)
| Field | Value |
| --- | --- |
| Objective | No-pretrain setting: `pretrained=False` with LoRA r=16 injected into fc2-only. |
| Baseline | Compare against E0004 (pretrained=True) and E0022 (fc1+fc2 no pretrain). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=False`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_nopretrain_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc2)`, r=16, alpha=32; backbone frozen; `pretrained=False`. |
| Metrics (must save) | `artifacts/work_dirs/E0023_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0023_full/val_metrics.json`, `artifacts/work_dirs/E0023_full/test_metrics.json` |
| Checks | Smoke/full commands exit 0; metrics JSON exists. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0023_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0023_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0023_smoke` |
| Full cmd | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc2_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0023_full --gpus 4 --seed 0` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `artifacts/work_dirs/E0023_smoke/smoke_*.log`, `artifacts/work_dirs/E0023_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0023_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0023_full/*.pth`, `artifacts/work_dirs/E0023_full/val_metrics.json`, `artifacts/work_dirs/E0023_full/test_metrics.json` |
| Results |  |


### E0024: Dinov3 ConvNeXt-S + RetinaNet (LoRA r=16, fc1-only, no pretrain)
| Field | Value |
| --- | --- |
| Objective | No-pretrain setting: `pretrained=False` with LoRA r=16 injected into fc1-only. |
| Baseline | Compare against E0021 (pretrained=True fc1-only) and E0022/E0023 (no pretrain). |
| Model | `Dinov3TimmConvNeXtLoRA` + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=False`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_nopretrain_1x_sardet_bs64_amp.py` |
| Params | target=`(mlp.fc1)`, r=16, alpha=32; backbone frozen; `pretrained=False`. |
| Metrics (must save) | `artifacts/work_dirs/E0024_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0024_full/val_metrics.json`, `artifacts/work_dirs/E0024_full/test_metrics.json` |
| Checks | Smoke/full commands exit 0; metrics JSON exists. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0024_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0024_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0024_smoke` |
| Full cmd | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_lora_ablation/retinanet_dinov3-timm-convnext-small_lora-r16_fc1_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0024_full --gpus 4 --seed 0` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `artifacts/work_dirs/E0024_smoke/smoke_*.log`, `artifacts/work_dirs/E0024_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0024_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0024_full/*.pth`, `artifacts/work_dirs/E0024_full/val_metrics.json`, `artifacts/work_dirs/E0024_full/test_metrics.json` |
| Results |  |


### E0025: Dinov3 ConvNeXt-S + RetinaNet (Full fine-tune, no LoRA, no pretrain)
| Field | Value |
| --- | --- |
| Objective | No-pretrain setting: `pretrained=False` (random init) with full fine-tune (backbone trainable), no LoRA. |
| Baseline | Compare against E0019 (pretrained=True full-ft) and E0022 (LoRA no pretrain). |
| Model | `TimmConvNeXt` (Dinov3 arch) + RetinaNet. |
| Weights | `convnext_small.dinov3_lvd1689m` (`pretrained=False`). |
| Code path | `mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_nopretrain_1x_sardet_bs64_amp.py` |
| Params | Backbone trainable; neck/head trained; `pretrained=False`. |
| Metrics (must save) | `artifacts/work_dirs/E0025_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0025_full/val_metrics.json`, `artifacts/work_dirs/E0025_full/test_metrics.json` |
| Checks | Smoke/full commands exit 0; metrics JSON exists. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0025_full --gpus 1 --seed 0` |
| Multi-GPU script | `bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0025_full --gpus 4 --seed 0` |
| Smoke cmd | `bash scripts/run_sardet_smoke_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0025_smoke` |
| Full cmd | `EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh --config mmdet_toolkit/local_configs/SARDet/dinov3_baselines/retinanet_dinov3-timm-convnext-small_full-ft_nopretrain_1x_sardet_bs64_amp.py --work-dir artifacts/work_dirs/E0025_full --gpus 4 --seed 0` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `artifacts/work_dirs/E0025_smoke/smoke_*.log`, `artifacts/work_dirs/E0025_full/*.log` |
| Artifacts | `artifacts/work_dirs/E0025_smoke/smoke_metrics.json`, `artifacts/work_dirs/E0025_full/*.pth`, `artifacts/work_dirs/E0025_full/val_metrics.json`, `artifacts/work_dirs/E0025_full/test_metrics.json` |
| Results |  |
