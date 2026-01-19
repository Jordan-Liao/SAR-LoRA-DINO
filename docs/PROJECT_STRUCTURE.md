# Project Structure

This repo is organized as a paper-ready code release:

## Top-level

- `sar_lora_dino/`: core python package (LoRA utilities + DINOv3/ConvNeXt MMDet backbones).
- `configs/`: runnable MMDetection configs (base templates + SARDet-100K experiments).
- `tools/`: stable CLI entrypoints (train/test/stats).
- `scripts/`: reproducibility helpers (smoke/full runners, exporting, analysis).
- `visualization/`: qualitative export tools (VR / painted detections / Grad-CAM) and packaging helpers.
- `docs/`: documentation (setup, conventions, FAQ, project notes).
- `data/`: dataset entrypoint (symlink or local copy; not vendored).
- `artifacts/`: experiment ledger + results tables + weights + run outputs + visualization outputs.

## Key files

- Dataset config: `configs/_base_/datasets/SARDet_100k.py`
  - Uses `SARDET100K_ROOT` (defaults to `<repo>/data/SARDet_100K`).
- Main runner scripts:
  - `scripts/run_sardet_smoke.sh`
  - `scripts/run_sardet_smoke_cfg.sh`
  - `scripts/run_sardet_full_cfg.sh`
- Core CLI tools:
  - `tools/train.py`
  - `tools/test_to_json.py`
- Experiment ledger: `artifacts/experiments/experiment.md`
