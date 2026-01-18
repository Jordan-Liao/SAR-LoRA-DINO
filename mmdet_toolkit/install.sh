#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

ENV_NAME="${ENV_NAME:-sar_lora_dino}"

# create env
conda create -y -n "${ENV_NAME}" python=3.8
conda activate "${ENV_NAME}"

# install pytorch
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# install dependencies of openmmlab
pip install -U openmim
mim install "mmengine==0.8.4"
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"

# install other dependencies
pip install -r requirements.txt

# install package (sar_lora_dino)
pip install -v -e .
