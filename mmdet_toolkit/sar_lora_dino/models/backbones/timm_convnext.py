from __future__ import annotations

from typing import Sequence

from torch import nn

from mmdet.registry import MODELS

from sar_lora_dino.models.lora import count_parameters


@MODELS.register_module()
class TimmConvNeXt(nn.Module):
    """ConvNeXt (timm, features_only) backbone wrapper with optional freezing."""

    def __init__(
        self,
        model_name: str = "convnext_small",
        pretrained: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        freeze_backbone: bool = False,
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
                "TimmConvNeXt requires `timm`. Install it in the conda env: `pip install timm>=1.0.17`."
            ) from e

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if unfreeze_stages:
            for stage_idx in unfreeze_stages:
                stage = getattr(self.backbone, f"stages_{int(stage_idx)}", None)
                if stage is None:
                    raise ValueError(
                        f"TimmConvNeXt cannot find stages_{stage_idx} in timm features model for {self.model_name}."
                    )
                for p in stage.parameters():
                    p.requires_grad_(True)

        if verbose:
            total = count_parameters(self.backbone, trainable_only=False)
            trainable = count_parameters(self.backbone, trainable_only=True)
            try:
                from mmengine.logging import print_log

                print_log(
                    f"[TimmConvNeXt] model={self.model_name}, "
                    f"trainable={trainable:,}/{total:,} ({trainable/total:.2%})",
                    logger="current",
                )
            except Exception:
                pass

    def forward(self, x):
        outs = self.backbone(x)
        return list(outs)
