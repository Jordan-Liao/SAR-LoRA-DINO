## Artifacts

This folder is the **single place** for paper/repro artifacts:

- `artifacts/experiments/`: experiment ledger + exported result tables (tracked).
- `artifacts/weights/`: local checkpoints / pretrained weights (not vendored; kept out of git).
- `artifacts/work_dirs/`: training/eval runs (`.pth`, logs, metrics JSONs, etc.).
- `artifacts/visualizations/`: exported VR/Grad-CAM images and packages.

Only small documentation files are tracked in git; large outputs belong in Releases or external storage.
