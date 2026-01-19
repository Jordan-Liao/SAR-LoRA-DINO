# LoRA Design

This repo provides lightweight LoRA adapters and injection helpers under `sar_lora_dino/models/lora.py`,
plus an MMDetection backbone wrapper `Dinov3TimmConvNeXtLoRA` under `sar_lora_dino/models/backbones/dinov3_timm_convnext_lora.py`.

## What is adapted

By default, `Dinov3TimmConvNeXtLoRA`:

- builds a timm **DINOv3 ConvNeXt** features model (`features_only=True`);
- freezes all backbone parameters;
- injects LoRA into **MLP FC layers** using name keywords:
  - `lora_target_keywords=("mlp.fc1", "mlp.fc2")` (default)

Configs can change:
- `lora_r`, `lora_alpha`, `lora_dropout`
- `lora_target_keywords` (Linear layers)
- `lora_target_conv_keywords` (Conv2d, groups=1 only)
- `unfreeze_stages` (optionally unfreeze timm `stages_{k}` parameters)

## Merge / unmerge

LoRA modules support explicit merge for inference:

- `sar_lora_dino.models.lora.merge_lora(model)`
- `sar_lora_dino.models.lora.unmerge_lora(model)`

Guideline:
- Merge **only** when you want a pure-base weight for inference/export.
- Do not save a merged checkpoint and then merge again at load time.

## Adapter-only weights (recommended for releases)

LoRA-only weights are much smaller than full checkpoints:

- `sar_lora_dino.models.lora.lora_state_dict(model)` returns only `lora_A/lora_B` tensors.
- `sar_lora_dino.models.lora.load_lora_state_dict(model, sd, strict=False)` loads adapters after injection.

Freezing policy helper:
- `sar_lora_dino.models.lora.mark_only_lora_as_trainable(model)`

Note: MMDetection training by default saves full model checkpoints. If you want to publish LoRA-only adapters,
export them from a trained model after loading the full checkpoint.

