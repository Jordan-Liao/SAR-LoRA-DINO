# Visualizations (outputs)

Visualization **code** lives in `visualization/`.

Recommended output roots:

- VR exports: `artifacts/visualizations/VR/`
- Grad-CAM exports: `artifacts/visualizations/GradCAM/`

These outputs can be large; keep them out of git.

Exception: we do track small per-split `metrics.json` files under `artifacts/visualizations/VR/*/*/metrics.json`
so the repo can carry paper-facing numbers even when the image dumps live in Releases/cloud storage.
