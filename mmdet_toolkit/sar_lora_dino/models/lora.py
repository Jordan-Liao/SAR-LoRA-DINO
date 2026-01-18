from __future__ import annotations

import math
from typing import Sequence

from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got: {type(base)}")

        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int, alpha: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        if not isinstance(base, nn.Conv2d):
            raise TypeError(f"LoRAConv2d expects nn.Conv2d, got: {type(base)}")
        if base.groups != 1:
            raise ValueError(f"LoRAConv2d only supports groups=1, got groups={base.groups}")

        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=r,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.lora_B = nn.Conv2d(
            in_channels=r,
            out_channels=base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _get_parent_and_key(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part not in parent._modules:
            raise KeyError(f"Module path not found: {module_name}")
        parent = parent._modules[part]
    return parent, parts[-1]


def inject_lora_linear(
    root: nn.Module,
    target_keywords: Sequence[str],
    r: int,
    alpha: float,
    dropout: float,
) -> int:
    replaced = 0
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(k in name for k in target_keywords):
            continue
        parent, key = _get_parent_and_key(root, name)
        parent._modules[key] = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
        replaced += 1
    return replaced


def inject_lora_conv2d(
    root: nn.Module,
    target_keywords: Sequence[str],
    r: int,
    alpha: float,
    dropout: float,
) -> int:
    replaced = 0
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups != 1:
            continue
        if not any(k in name for k in target_keywords):
            continue
        parent, key = _get_parent_and_key(root, name)
        parent._modules[key] = LoRAConv2d(module, r=r, alpha=alpha, dropout=dropout)
        replaced += 1
    return replaced
