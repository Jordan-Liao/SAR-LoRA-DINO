#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

ENV_NAME="${ENV_NAME:-sar_lora_dino}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
TORCH_VERSION="${TORCH_VERSION:-2.0.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.0.2}"
CUDA_VERSION="${CUDA_VERSION:-11.8}"
MMENGINE_VERSION="${MMENGINE_VERSION:-0.8.4}"
MMCV_VERSION="${MMCV_VERSION:-2.0.1}"
MMDET_VERSION="${MMDET_VERSION:-3.1.0}"

if conda env list | awk 'NF {print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Reusing existing conda env: ${ENV_NAME}"
else
  echo "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_NAME}"

# install pytorch (example: CUDA 11.8)
conda install -y \
  "pytorch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  "pytorch-cuda=${CUDA_VERSION}" \
  -c pytorch -c nvidia

# install MMDetection stack
pip install -U openmim
mim install "mmengine==${MMENGINE_VERSION}"
mim install "mmcv==${MMCV_VERSION}"
mim install "mmdet==${MMDET_VERSION}"

# install other dependencies
pip install -r requirements.txt

# install package (sar_lora_dino)
pip install -v -e .
