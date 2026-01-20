#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-sar_lora_dino}"
MODEL_NAME="${MODEL_NAME:-convnext_small.dinov3_lvd1689m}"
export MODEL_NAME

echo "ENV_NAME=${ENV_NAME}"
echo "MODEL_NAME=${MODEL_NAME}"

conda run -n "${ENV_NAME}" python - <<'PY'
import os

import timm

model_name = os.environ.get("MODEL_NAME", "convnext_small.dinov3_lvd1689m")
print("Downloading (if needed):", model_name)
timm.create_model(model_name, pretrained=True)
print("OK")
PY
