# Training

The recommended workflow is **Smoke → Full**, with all outputs written under `artifacts/work_dirs/`.

## 1) Smoke (end-to-end sanity)

```bash
bash scripts/run_sardet_smoke.sh
```

This runs a tiny subset and writes `artifacts/work_dirs/sardet_smoke/smoke_metrics.json`.

## 2) Smoke (any config)

```bash
bash scripts/run_sardet_smoke_cfg.sh \
  --config configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py \
  --work-dir artifacts/work_dirs/E0002_smoke
```

## 3) Full train + eval (any config)

```bash
EVAL_SPLITS=val,test bash scripts/run_sardet_full_cfg.sh \
  --config configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py \
  --work-dir artifacts/work_dirs/E0002_full_seed0 \
  --gpus 4 \
  --seed 0
```

Outputs:
- `train.log`
- `val_metrics.json` (and `test_metrics.json` if `EVAL_SPLITS=val,test`)
- checkpoints (gitignored)

## Seeds / multi-run

See the runnable “Full cmd” blocks in `artifacts/experiments/experiment.md` for the exact loops used to produce tables.
