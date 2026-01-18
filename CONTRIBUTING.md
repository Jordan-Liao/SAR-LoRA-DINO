# Contributing

Thanks for your interest in improving this repository.

## What to include (and what not to include)

- Do not commit datasets, large checkpoints, or training outputs.
  - Dataset root is configured via `SARDET100K_ROOT` (see `docs/GETTING_STARTED.md`).
  - Local checkpoints belong in `artifacts/weights/` (kept out of git).
  - Training/eval outputs belong in `artifacts/work_dirs/` (kept out of git).
- Keep changes focused and reproducible:
  - Update `docs/GETTING_STARTED.md` if you change install/run steps.
  - Update `artifacts/experiments/experiment.md` if you add/modify experiment configs.

## Quick checks before a PR

```bash
bash -n scripts/*.sh visualization/*.sh
python -m compileall -q mmdet_toolkit/sar_lora_dino mmdet_toolkit/tools scripts visualization
```

## Style

- Prefer small, reviewable diffs.
- Keep paths portable (avoid hard-coded absolute paths).
- Follow existing conventions in `mmdet_toolkit/` and `scripts/`.
