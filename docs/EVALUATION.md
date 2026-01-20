# Evaluation

Evaluation is driven by MMEngine/MMDetection’s runner and the helper script `tools/test_to_json.py`.

## Standard path (recommended)

Use `scripts/run_sardet_full_cfg.sh` which trains and then evaluates, writing:

- `val_metrics.json`
- `test_metrics.json` (optional; set `EVAL_SPLITS=val,test`)

## Run eval only

```bash
conda run -n sar_lora_dino python tools/test_to_json.py \
  --config configs/sar_lora_dino/retinanet_dinov3_convnexts_lora_r16_fc1fc2_sardet100k.py \
  --checkpoint /path/to/checkpoint.pth \
  --work-dir artifacts/work_dirs/eval_only \
  --out-json artifacts/work_dirs/eval_only/val_metrics.json \
  --cfg-options \
    test_dataloader.dataset.ann_file=${SARDET100K_ROOT}/Annotations/val.json \
    test_dataloader.dataset.data_prefix.img=JPEGImages/val/ \
    test_evaluator.ann_file=${SARDET100K_ROOT}/Annotations/val.json \
    model.test_cfg.score_thr=0.0
```

## Metrics format

The JSON mirrors the dict returned by `runner.test()`. For SARDet-100K (COCO bbox), common keys include:

- `coco/bbox_mAP`
- `coco/bbox_mAP_50`
- `coco/bbox_mAP_75`

Aggregated tables are exported to:
- `artifacts/experiments/experiment_results.tsv`
- `artifacts/experiments/experiment_results.md`
