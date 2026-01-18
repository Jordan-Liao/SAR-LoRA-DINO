# Project Structure

This repo is organized as a paper-ready code release:

## Top-level

- `mmdet_toolkit/`: core codebase (MMDetection-style) + the `sar_lora_dino` python package.
- `mmdet_toolkit/local_configs/`: runnable experiment configs grouped by dataset/task.
- `scripts/`: reproducibility helpers (smoke/full runners, exporting, analysis).
- `visualization/`: qualitative export tools (VR / painted detections / Grad-CAM) and packaging helpers.
- `docs/`: documentation (setup, conventions, FAQ, project notes).
- `data/`: dataset entrypoint (symlink or local copy; not vendored).
- `artifacts/`: experiment ledger + results tables + weights + run outputs + visualization outputs.

## Key files

- Dataset config: `mmdet_toolkit/configs/_base_/datasets/SARDet_100k.py`
  - Uses `SARDET100K_ROOT` (defaults to `<repo>/data/SARDet_100K`).
- Main runner scripts:
  - `scripts/run_sardet_smoke.sh`
  - `scripts/run_sardet_smoke_cfg.sh`
  - `scripts/run_sardet_full_cfg.sh`
- Experiment ledger: `artifacts/experiments/experiment.md`
