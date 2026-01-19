# SAR-LoRA-DINO

This repo focuses on **DINOv3 (ConvNeXt) + LoRA** for SAR object detection on the **SARDet-100K** benchmark, built on **MMDetection 3.x** (MMEngine/MMCV).

This repo includes runnable configs/scripts, plus an experiment ledger and result tables under `artifacts/`.

## Quick Start

- Install + dataset setup + runnable commands: `docs/GETTING_STARTED.md`
- Installation (standalone): `docs/INSTALLATION.md`
- Dataset notes: `docs/DATASET_SARDET100K.md`
- Repo structure overview: `docs/PROJECT_STRUCTURE.md`
- Conventions (naming / env vars): `docs/CONVENTIONS.md`
- Configs explained: `docs/CONFIGS_EXPLAINED.md`
- LoRA design & merge/unmerge: `docs/LORA_DESIGN.md`
- Training / evaluation: `docs/TRAINING.md`, `docs/EVALUATION.md`
- Model Zoo (configs + metrics pointers): `docs/MODEL_ZOO.md`
- Export / visualization: `docs/EXPORT.md`
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
- `LICENSES/Apache-2.0.txt`: Apache 2.0 (MMDetection-style configs; see `THIRD_PARTY_NOTICES.md`)

See `THIRD_PARTY_NOTICES.md` for attributions.

## Citation

If you use this repo, see `CITATION.bib` / `CITATION.cff`.

If you use the SARDet-100K dataset, please also cite:

```bibtex
@inproceedings{li2024sardet100k,
  title={SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection},
  author={Yuxuan Li and Xiang Li and Weijie Li and Qibin Hou and Li Liu and Ming-Ming Cheng and Jian Yang},
  year={2024},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
}
```
