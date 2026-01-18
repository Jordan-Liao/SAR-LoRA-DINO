# Scripts

## Repro runners

- `scripts/verify_env.sh`: checks the conda env can import `mmcv.ops`, `mmdet`, `sar_lora_dino` and sees CUDA.
- `scripts/run_sardet_smoke.sh`: end-to-end smoke train+eval on small SARDet-100K subsets.
- `scripts/run_sardet_smoke_cfg.sh`: smoke runner for any MMDet config (subset train+eval).
- `scripts/run_sardet_full_cfg.sh`: full train + eval (val/test) runner for any MMDet config.

## Dataset utilities

- `scripts/make_coco_subset.py`: deterministic COCO subset generator (by image sampling).

## Evaluation

- `scripts/mmdet_test_to_json.py`: run MMDet test and save returned metrics to a JSON file.

## Analysis / packaging

- `scripts/count_trainable_params.py`: report trainable parameter counts for a given config.
- `scripts/export_experiment_results.py`: parse `artifacts/experiments/experiment.md` and generate `artifacts/experiments/experiment_results.tsv/.md`.
- `scripts/check_artifacts.py`: check whether expected metrics/visualization packages exist under `artifacts/`.
- `scripts/package_metrics_jsons.py`: bundle all `*metrics*.json` under `artifacts/` into a single zip (Release-friendly).
- `scripts/pick_best_checkpoint.py`: pick a best checkpoint file under a MMDet `work_dir`.

## Visualization

See `visualization/README.md`.
