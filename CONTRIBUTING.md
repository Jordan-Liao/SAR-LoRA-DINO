# Contributing

Thanks for taking the time to contribute.

## Development setup

- Follow `docs/INSTALL.md` (MMDetection 3.x + `pip install -e .`).
- Verify your environment: `ENV_NAME=sar_lora_dino bash scripts/verify_env.sh`.

## What not to commit

- Datasets (`data/sardet100k/`)
- Large artifacts (checkpoints, logs, visualization dumps)

See `.gitignore` and `docs/ARTIFACTS_AND_RELEASES.md`.

## Pull requests

- Keep changes focused and well-scoped.
- If you change experiment configs or scripts, update `artifacts/experiments/experiment.md` (and regenerate tables via `python scripts/export_experiment_results.py`).
