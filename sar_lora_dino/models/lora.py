from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Sequence

import torch
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

        self.merged = False

    def _delta_weight(self) -> torch.Tensor:
        return self.lora_B.weight @ self.lora_A.weight

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.base.weight.add_(self._delta_weight() * self.scaling)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight.sub_(self._delta_weight() * self.scaling)
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base(x)
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

        self.merged = False

    def _delta_weight(self) -> torch.Tensor:
        a = self.lora_A.weight.squeeze(-1).squeeze(-1)  # (r, in)
        b = self.lora_B.weight  # (out, r, kH, kW)
        return torch.einsum("orhw,ri->oihw", b, a)

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.base.weight.add_(self._delta_weight() * self.scaling)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight.sub_(self._delta_weight() * self.scaling)
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base(x)
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


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a LoRA-only state_dict (LoRA adapter weights only).

    This is useful for sharing lightweight adapters without bundling large base model checkpoints.
    """

    sd = model.state_dict()
    return {k: v for k, v in sd.items() if ".lora_A." in k or ".lora_B." in k}


def load_lora_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor], strict: bool = False) -> None:
    """Load a LoRA-only state_dict into a model that already has LoRA modules injected."""

    model.load_state_dict(dict(state_dict), strict=strict)


def mark_only_lora_as_trainable(model: nn.Module, train_bias: bool = False) -> int:
    """Freeze all parameters except LoRA adapters (and optionally base biases)."""

    for p in model.parameters():
        p.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            for p in module.lora_A.parameters():
                p.requires_grad_(True)
            for p in module.lora_B.parameters():
                p.requires_grad_(True)
            if train_bias and getattr(module.base, "bias", None) is not None:
                module.base.bias.requires_grad_(True)

    return count_parameters(model, trainable_only=True)


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """In-place merge LoRA weights into base weights (inference convenience)."""

    merged = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            if not module.merged:
                module.merge()
                merged += 1
    return merged


@torch.no_grad()
def unmerge_lora(model: nn.Module) -> int:
    """Undo a previous merge_lora() call."""

    unmerged = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            if module.merged:
                module.unmerge()
                unmerged += 1
    return unmerged
