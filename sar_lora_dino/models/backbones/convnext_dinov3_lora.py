from __future__ import annotations

from mmdet.registry import MODELS

from sar_lora_dino.models.backbones.dinov3_timm_convnext_lora import Dinov3TimmConvNeXtLoRA


@MODELS.register_module()
class ConvNeXtDinov3LoRA(Dinov3TimmConvNeXtLoRA):
    """Paper-facing alias for :class:`Dinov3TimmConvNeXtLoRA`."""


__all__ = ["ConvNeXtDinov3LoRA"]

