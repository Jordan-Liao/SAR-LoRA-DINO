# Artifacts & Releases

This repo keeps **code + paper-facing tables** in git, but keeps **runtime outputs** (checkpoints, logs, visualizations) out of git.

## What lives where

- Tracked (in git):
  - `artifacts/experiments/experiment.md`: experiment ledger (commands + expected outputs).
  - `artifacts/experiments/experiment_results.tsv`: copy-friendly result table (paper table source).
  - `artifacts/experiments/experiment_results.md`: same table (Markdown).
  - `artifacts/work_dirs/**/{smoke,val,test}_metrics.json`: small, paper-facing metrics JSONs (so readers can diff/parse results without re-running).
  - `artifacts/visualizations/VR/*/*/metrics.json`: small metrics JSONs saved alongside VR exports (the image dumps remain out of git).
- Not tracked (local or Release assets):
  - `artifacts/work_dirs/`: training/eval outputs (`*.pth`, logs, ...). (Only the small `*_metrics.json` files are tracked.)
  - `artifacts/visualizations/`: VR / sample500 / Grad-CAM outputs.
  - `artifacts/weights/`: pretrained weights and checkpoints (large).

## Should full/sample500 go into git?

No. Full-split VR images and checkpoints are **too large** for a normal GitHub repo (and can exceed the 100MB/file limit).

Recommended:
- Put large outputs in **GitHub Releases / Kaggle / cloud drive**.
- Keep only the **ledger + tables + small metrics JSONs** tracked in git.

## Quick local setup

1) Link dataset into `data/` (local only; gitignored):

```bash
bash scripts/setup_sardet_dataset.sh
```

2) Sanity-check what artifacts are currently present:

```bash
python scripts/check_artifacts.py
```

If you only have `artifacts/experiments/experiment_results.tsv` but are missing the corresponding
`artifacts/work_dirs/**/{smoke,val,test}_metrics.json`, you can materialize them from the table:

```bash
python scripts/materialize_metrics_jsons_from_results_tsv.py
```

## Generate artifacts (high-level)

Training/eval (metrics + checkpoints):
- Use commands in `artifacts/experiments/experiment.md` (Smoke then Full).
- Outputs go under `artifacts/work_dirs/...`.

Visualizations:
- Full VR export: `bash visualization/export_sardet_vr.sh ...` (writes to `artifacts/visualizations/VR/` by default).
- Sample500 VR repack: `python visualization/repack_vr_sample500_from_subset.py ...`.
- Grad-CAM sample500: `python visualization/export_gradcam_sample500_bundle.py ...`.

See `docs/RESULTS_BUNDLE.md` for concrete commands and recommended output directories.

## Package Release assets (optional)

To bundle all metrics JSONs into a single zip (Release-friendly):

```bash
python scripts/package_metrics_jsons.py
```

To check and bundle Release-level assets (ckpt/log/vis zips) into one directory:

```bash
python scripts/check_release_assets.py
python scripts/build_release_bundle.py
```
