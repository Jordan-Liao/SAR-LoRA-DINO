# Conventions

This repo follows a few conventions to keep experiments reproducible and paper-writing friendly.

## Environment variables

- `ENV_NAME` (default: `sar_lora_dino`)
  - Conda env name used by `scripts/*.sh`.
- `SARDET100K_ROOT` (default: `<repo>/data/sardet100k`)
  - Dataset root containing `Annotations/` and `JPEGImages/`.
- `CUDA_VISIBLE_DEVICES`
  - If unset, runner scripts try to auto-pick the GPU(s) with most free memory.
- `CUDA_EXCLUDE_DEVICES` (default: empty)
  - Comma-separated GPU indices to exclude from auto-pick (e.g. `CUDA_EXCLUDE_DEVICES=0,1`).
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` (default: `0`)
  - Set to `1` if you run in an offline environment.

## Paths and outputs

- Do not vendor datasets, large checkpoints, or run outputs in git.
- Training/eval outputs go to `artifacts/work_dirs/` (kept out of git).
- Qualitative exports (VR / Grad-CAM) go to `artifacts/visualizations/` (kept out of git).
- Paper-facing ledgers/tables live in `artifacts/experiments/` (tracked).

## Experiment naming

### `artifacts/experiments/experiment.md`

- Each experiment has a stable ID: `E0001`, `E0002`, ...
- Each experiment table should include:
  - objective, model/config path, weights/pretrain, trainable params (if relevant),
    runnable smoke/full commands, and where metrics are saved.

### `artifacts/work_dirs/` naming

Recommended patterns:

- Smoke:
  - `artifacts/work_dirs/<tag>_smoke/` (produces `smoke_metrics.json`)
- Full:
  - `artifacts/work_dirs/E####_<tag>_seed<k>/`
  - e.g. `artifacts/work_dirs/E0002_full_seed0/` (produces `val_metrics.json`, optionally `test_metrics.json`)
- VR export:
  - `artifacts/visualizations/VR/<name>/<split>/` (produces `metrics.json`, `predictions.pkl`, and `vis/`)

## Config naming (`configs/`)

Configs are grouped by dataset/task, then named to encode key choices.

Example:

- `configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py`

Suggested schema:

```
<detector>_<backbone>_<train-mode>_<schedule>_<dataset>_bs<global>[_amp].py
```
