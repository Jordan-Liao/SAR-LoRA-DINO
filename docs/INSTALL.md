# Installation

SAR-LoRA-DINO is an add-on package for **MMDetection 3.x** (MMEngine/MMCV).

## Requirements (known-good)

- Python: `>=3.8`
- PyTorch: `2.x`
- MMEngine: `0.8.4`
- MMCV: `2.0.1`
- MMDetection: `3.1.0`

## Option A: one-shot (conda)

```bash
ENV_NAME=sar_lora_dino bash scripts/setup_env.sh
```

The script is idempotent: re-running it reuses the env and re-installs the stack.
You can override versions via env vars (see `scripts/setup_env.sh`).

## Option B: conda env create (environment.yml)

This repo includes an example `environment.yml` (CUDA 11.8 + PyTorch 2.0.1).
Adjust versions to match your system if needed.

```bash
conda env create -f environment.yml
conda activate sar_lora_dino
```

## Option C: manual install

```bash
conda create -n sar_lora_dino python=3.8 -y
conda activate sar_lora_dino

# Example: CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install "mmengine==0.8.4"
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"

pip install -r requirements.txt
pip install -e .
```

## Verify

```bash
ENV_NAME=sar_lora_dino bash scripts/verify_env.sh
python -c "import sar_lora_dino; print(sar_lora_dino.__version__)"
```
