#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sar_lora_dino}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 1
fi

if ! conda env list | awk 'NF {print $1}' | grep -qx "${ENV_NAME}"; then
  echo "conda env '${ENV_NAME}' not found (set ENV_NAME=... to override)" >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python -c "import torch, mmcv, mmengine, mmdet, sar_lora_dino; import mmcv.ops; print('python', __import__('sys').version.replace('\\n',' ')); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('mmcv', mmcv.__version__); print('mmengine', mmengine.__version__); print('mmdet', mmdet.__version__); print('sar_lora_dino', sar_lora_dino.__file__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"
