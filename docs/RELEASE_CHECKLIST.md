# Release Checklist (Paper-Level Artifacts)

This repo keeps code + paper tables in git, and keeps large runtime artifacts (checkpoints / logs / visualizations) in Releases or external storage.

See also: `docs/ARTIFACTS_AND_RELEASES.md` and `docs/RESULTS_BUNDLE.md`.

## A. What should exist for a “paper-level” release

1) Metrics JSONs (small; tracked in git)
- `artifacts/work_dirs/**/{smoke,val,test}_metrics.json`
- `artifacts/visualizations/VR/*/*/metrics.json`

2) Training/eval outputs (large; Release only)
- `artifacts/work_dirs/**/`:
  - checkpoints: `*.pth`
  - logs: `train.log`, `test_val.log`, `test_test.log` (and/or your runner logs)

3) Visualization exports (large; Release only)
- Full-split VR exports: `artifacts/visualizations/VR/<TAG>/{val,test}/...`
- Optional packed zips:
  - full-split packs: `artifacts/visualizations/VR_fullsplit_*/**.zip`
  - full-split preds-only packs (recommended for GitHub Releases): `artifacts/visualizations/VR_preds_*/**.zip`
  - sample500 packs: `artifacts/visualizations/VR_sample500/VR_SAMPLE500_*.zip`
  - sample500 LoRA grid: `artifacts/visualizations/VR_sample500_lora/VR_SAMPLE500_*.zip`
  - Grad-CAM sample500: `artifacts/visualizations/GradCAM_sample500/GradCAM_SAMPLE500_*.zip`

## B. Generate (high-level)

1) Dataset
- Set `SARDET100K_ROOT` or run `bash scripts/setup_sardet_dataset.sh`.

2) Train + eval
- Use “Smoke cmd” then “Full cmd” in `artifacts/experiments/experiment.md`.
- Recommended output root: `artifacts/work_dirs/<EID>_.../`.

3) Visualizations
- Full VR export: `bash visualization/export_sardet_vr.sh ...`
- Sample500 subsets + repack: `python visualization/repack_vr_sample500_from_subset.py ...`
- Grad-CAM sample500: `python visualization/export_gradcam_sample500_bundle.py ...`

4) Sanity check
```bash
python scripts/check_artifacts.py --strict
```

5) Release-asset check (ckpt/log/vis packages)
```bash
python scripts/check_release_assets.py
```

## C. Package for Release (recommended)

1) Bundle metrics JSONs
```bash
python scripts/package_metrics_jsons.py
```

2) Build a Release-ready bundle dir (zips + manifest)
```bash
python scripts/build_release_bundle.py
```

3) Upload large artifacts
- Upload zips (VR fullsplit / sample500 / Grad-CAM) + checkpoint bundles to GitHub Releases (or Kaggle / cloud drive).
- After upload, paste links into `README.md` and/or `docs/ARTIFACTS_AND_RELEASES.md`.
