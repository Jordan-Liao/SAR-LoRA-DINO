# SARDet-100K Dataset

This repo expects SARDet-100K in **COCO** format (images + `Annotations/*.json`).

## Download

- Baidu Disk: https://pan.baidu.com/s/1dIFOm4V2pM_AjhmkD1-Usw?pwd=SARD
- Kaggle: https://www.kaggle.com/datasets/greatbird/sardet-100k

## Expected layout

```text
SARDet_100K/
  Annotations/{train,val,test}.json
  JPEGImages/{train,val,test}/
```

## Point the code to the dataset

Option A: environment variable:

```bash
export SARDET100K_ROOT=/path/to/SARDet_100K
```

Option B: symlink into this repo (recommended for local runs):

```bash
ln -s /path/to/SARDet_100K data/SARDet_100K
# or:
bash scripts/setup_sardet_dataset.sh
```

Notes:
- `configs/_base_/datasets/SARDet_100k.py` reads `SARDET100K_ROOT` and defaults to `data/SARDet_100K`.
- The dataset itself is not vendored in git (see `.gitignore`).

