from __future__ import annotations

from typing import Sequence

from torch import nn

from mmdet.registry import MODELS

from sar_lora_dino.models.lora import count_parameters, inject_lora_conv2d, inject_lora_linear


@MODELS.register_module()
class Dinov3TimmConvNeXtLoRA(nn.Module):
    """DINOv3 ConvNeXt (timm) backbone with LoRA adapters."""

    def __init__(
        self,
        model_name: str = "convnext_small.dinov3_lvd1689m",
        pretrained: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_keywords: Sequence[str] = ("mlp.fc1", "mlp.fc2"),
        lora_target_conv_keywords: Sequence[str] = (),
        unfreeze_stages: Sequence[int] = (),
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = tuple(out_indices)

        try:
            import timm
        except Exception as e:
            raise RuntimeError(
                "Dinov3TimmConvNeXtLoRA requires `timm`. Install it in the conda env: `pip install timm>=1.0.17`."
            ) from e

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if unfreeze_stages:
            for stage_idx in unfreeze_stages:
                stage = getattr(self.backbone, f"stages_{int(stage_idx)}", None)
                if stage is None:
                    raise ValueError(
                        f"Dinov3TimmConvNeXtLoRA cannot find stages_{stage_idx} in timm features model for {self.model_name}."
                    )
                for p in stage.parameters():
                    p.requires_grad_(True)

        replaced_linear = inject_lora_linear(
            self.backbone,
            target_keywords=tuple(lora_target_keywords),
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        replaced_conv = 0
        if lora_target_conv_keywords:
            replaced_conv = inject_lora_conv2d(
                self.backbone,
                target_keywords=tuple(lora_target_conv_keywords),
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        if verbose:
            total = count_parameters(self.backbone, trainable_only=False)
            trainable = count_parameters(self.backbone, trainable_only=True)
            try:
                from mmengine.logging import print_log

                print_log(
                    f"[Dinov3TimmConvNeXtLoRA] replaced_linear={replaced_linear}, "
                    f"replaced_conv={replaced_conv}, "
                    f"trainable={trainable:,}/{total:,} ({trainable/total:.2%})",
                    logger="current",
                )
            except Exception:
                pass

    def forward(self, x):
        outs = self.backbone(x)
        return list(outs)
