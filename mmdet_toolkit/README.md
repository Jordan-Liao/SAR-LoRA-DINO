# MMDetection Toolkit (`mmdet_toolkit/`)

This folder contains the MMDetection 3.x style codebase used by **SAR-LoRA-DINO** (SARDet-100K).

It provides:

- the `sar_lora_dino` python package (custom backbones / LoRA utilities)
- MMDet configs under `local_configs/`

## Install

Example (PyTorch 2.0.1 + CUDA 11.8):

```bash
conda create -n sar_lora_dino python=3.8 -y
conda activate sar_lora_dino

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install "mmengine==0.8.4"
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"

cd mmdet_toolkit
pip install -r requirements.txt
pip install -v -e .
```

Sanity check (from repo root):

```bash
ENV_NAME=sar_lora_dino bash scripts/verify_env.sh
```

## Dataset

Dataset config: `configs/_base_/datasets/SARDet_100k.py`

It expects `SARDET100K_ROOT` (defaults to `<repo>/data/SARDet_100K`):

```bash
export SARDET100K_ROOT=/path/to/SARDet_100K
```

Expected layout:

```
SARDet_100K/
  Annotations/{train,val,test}.json
  JPEGImages/{train,val,test}/
```

## Train / Eval

From repo root (recommended), use the reproducibility wrappers:

- smoke: `bash scripts/run_sardet_smoke.sh`
- full: `bash scripts/run_sardet_full_cfg.sh --config <...> --work-dir <...>`

Low-level entrypoints (no wrappers) live under `../scripts/` and `../visualization/`:

```bash
conda run -n sar_lora_dino python ../scripts/mmdet_train.py local_configs/SARDet/smoke/retinanet_r50_sardet_smoke.py --work-dir ../artifacts/work_dirs/sardet_smoke
conda run -n sar_lora_dino python ../scripts/mmdet_test_to_json.py --config local_configs/SARDet/smoke/retinanet_r50_sardet_smoke.py --checkpoint ../artifacts/work_dirs/sardet_smoke/latest.pth --work-dir ../artifacts/work_dirs/sardet_smoke --out-json ../artifacts/work_dirs/sardet_smoke/val_metrics.json
```

## Configs

Configs are grouped under `local_configs/` (e.g. `local_configs/SARDet/`).
For experiment tables and exact commands, see `../artifacts/experiments/experiment.md`.

## Citation

See `../CITATION.bib` / `../CITATION.cff`.

## License

Apache 2.0 — see `LICENSE`.
