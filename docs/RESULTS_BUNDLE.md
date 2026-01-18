# Results Bundle (Experiments + Visualizations)

This repo does **not** track runtime outputs (e.g. `artifacts/work_dirs/`) in git. This note documents a clean, reproducible way to export and package:

- experiment tables (`artifacts/experiments/experiment.md` → `artifacts/experiments/experiment_results.tsv`)
- VR visualizations (full `val/test`)
- optional sample-aligned bundles (e.g. “fixed 500 images”)

## 1) Export VR on full `val/test`

Export qualitative results (metrics + predictions + painted images):

```bash
OUT_ROOT=artifacts/visualizations/VR SPLITS=val,test ENV_NAME=sar_lora_dino \
  bash visualization/export_sardet_vr.sh \
    --name <TAG> \
    --config <CONFIG.py> \
    --checkpoint <CKPT.pth>
```

Outputs land under `artifacts/visualizations/VR/<TAG>/<split>/...` and include `metrics.json`, `predictions.pkl`, and `vis/`.

## 2) Pack a full-split VR export into a zip (optional)

If you want a single zip (convenient for sharing):

```bash
python visualization/run_fullsplit_vr_pack.py \
  --name <TAG> \
  --config <CONFIG.py> \
  --checkpoint <CKPT.pth> \
  --split test
```

Use `--skip-export` if you already ran `visualization/export_sardet_vr.sh` and only want to pack.

If you only want **predictions + metrics** on the full split (much smaller; recommended for GitHub Releases),
use `--preds-only` (no per-image visualization images):

```bash
python visualization/run_fullsplit_vr_pack.py \
  --name <TAG> \
  --config <CONFIG.py> \
  --checkpoint <CKPT.pth> \
  --split test \
  --preds-only
```

## 3) Build aligned 500-image subsets (optional)

Create deterministic COCO subsets:

```bash
python scripts/make_coco_subset.py --in-json "$SARDET100K_ROOT/Annotations/val.json"  --out-json data/sardet_subsets/val_500.json  --num-images 500 --seed 0
python scripts/make_coco_subset.py --in-json "$SARDET100K_ROOT/Annotations/test.json" --out-json data/sardet_subsets/test_500.json --num-images 500 --seed 0
```

Then repack existing VR exports into aligned “sample500” zips (same images across methods):

```bash
python visualization/repack_vr_sample500_from_subset.py \
  --pkg-dir artifacts/visualizations/VR_sample500 \
  --val-subset data/sardet_subsets/val_500.json \
  --test-subset data/sardet_subsets/test_500.json \
  --tags <TAG1> <TAG2> <TAG3>
```

## 3.1) Sample500 VR bundle for LoRA experiments (optional)

If you want a ready-made “LoRA grid” sample500 bundle (exports + zips) driven by the experiment table:

```bash
python visualization/export_vr_sample500_lora_bundle.py \
  --pkg-dir artifacts/visualizations/VR_sample500_lora \
  --export-root artifacts/visualizations/VR_exports_sample500_lora \
  --val-ann data/sardet_subsets/val_500.json \
  --test-ann data/sardet_subsets/test_500.json
```

This requires the corresponding checkpoints/metrics to exist under `artifacts/work_dirs/` (see `artifacts/experiments/experiment_results.tsv`).

## 4) Grad-CAM on a fixed subset (optional)

For Grad-CAM overlays on the shared val-500 subset:

```bash
python visualization/export_gradcam_sample500_bundle.py \
  --pkg-dir artifacts/visualizations/GradCAM_sample500 \
  --export-root artifacts/visualizations/GradCAM_exports_sample500 \
  --val-ann data/sardet_subsets/val_500.json
```

## 5) Sharing

Upload the produced bundle directory (or zip) to your preferred storage (e.g., GitHub Releases, Kaggle, cloud drive).  
If you publish bundles, consider including checksums (these scripts generate `md5sum.txt`).
