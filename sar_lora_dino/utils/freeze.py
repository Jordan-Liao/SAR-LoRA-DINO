from __future__ import annotations

from torch import nn


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)

