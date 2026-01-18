# FAQ

## `ImportError: cannot import name ...` / `mmcv.ops` not found

This usually means MMCV was installed with an incompatible PyTorch/CUDA.

- Reinstall following the versions in `docs/GETTING_STARTED.md`.
- Run `bash scripts/verify_env.sh` to sanity check imports.

## Dataset path not found

Set:

```bash
export SARDET100K_ROOT=/path/to/SARDet_100K
```

or symlink:

```bash
ln -s /path/to/SARDet_100K data/SARDet_100K
```

## CUDA OOM

Try one or more of:

- Reduce `TRAIN_BATCH_SIZE` (for `scripts/run_sardet_full_cfg.sh`).
- Reduce input size in the dataset pipeline (config change).
- Use fewer `num_workers`.

## NCCL multi-GPU errors

Some hosts have global NCCL env vars that break distributed training.
`scripts/run_sardet_full_cfg.sh` already unsets common NCCL variables before launch.

