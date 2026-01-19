# SAR-LoRA-DINO

This repo focuses on **DINOv3 (ConvNeXt) + LoRA** for SAR object detection on the **SARDet-100K** benchmark, built on an MMDetection 3.x codebase (`mmdet_toolkit/`).

This repo includes runnable configs/scripts, plus an experiment ledger and result tables under `artifacts/`.

## Quick Start

- Install + dataset setup + runnable commands: `docs/GETTING_STARTED.md`
- Repo structure overview: `docs/PROJECT_STRUCTURE.md`
- Conventions (naming / env vars): `docs/CONVENTIONS.md`
- Artifacts & Releases (what goes in git vs Release): `docs/ARTIFACTS_AND_RELEASES.md`
- Release checklist (what’s still missing): `docs/RELEASE_CHECKLIST.md`

## Dataset

**Download**

- Baidu Disk: https://pan.baidu.com/s/1dIFOm4V2pM_AjhmkD1-Usw?pwd=SARD
- Kaggle: https://www.kaggle.com/datasets/greatbird/sardet-100k

**Expected layout**

```
SARDet_100K/
  Annotations/{train,val,test}.json
  JPEGImages/{train,val,test}/
```

Point the code to the dataset:

```bash
export SARDET100K_ROOT=/path/to/SARDet_100K
# or: ln -s /path/to/SARDet_100K data/SARDet_100K
```

## Weights

This repo does not vendor large checkpoints.

- DINOv3 (ConvNeXt) backbones are pulled automatically by `timm` when `pretrained=True` (internet required).
- Our trained SARDet-100K checkpoints: TBD.

## Reproducibility

- Smoke run (end-to-end): `bash scripts/run_sardet_smoke.sh`
- Full train + eval (any config): `bash scripts/run_sardet_full_cfg.sh --config <...> --work-dir <...>`
- VR export: `bash visualization/export_sardet_vr.sh --name <...> --config <...> --checkpoint <...>`
- Experiment ledger + aggregated tables:
  - `artifacts/experiments/experiment.md`
  - `artifacts/experiments/experiment_results.tsv`

## License

This repository is mixed-licensed:

- `LICENSE`: CC BY-NC 4.0 (repository-level assets by default)
- `mmdet_toolkit/LICENSE`: Apache 2.0 (MMDetection-based code)

See `THIRD_PARTY_NOTICES.md` for attributions.
