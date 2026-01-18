#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-sar_lora_dino}"
SARDET100K_ROOT="${SARDET100K_ROOT:-${REPO_ROOT}/data/SARDet_100K}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_DISABLE_TELEMETRY

CONFIG=""
WORK_DIR=""
GPUS="${GPUS:-1}"
SEED="${SEED:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
TRAIN_NUM_IMAGES="${TRAIN_NUM_IMAGES:-}"
VAL_NUM_IMAGES="${VAL_NUM_IMAGES:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-}"
EVAL_SPLITS="${EVAL_SPLITS:-val}"
CUDA_EXCLUDE_DEVICES="${CUDA_EXCLUDE_DEVICES:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_sardet_full_cfg.sh --config <config.py> --work-dir <dir> [--gpus N] [--seed S] [--resume <ckpt.pth>]

Environment variables:
  ENV_NAME=sar_lora_dino
  SARDET100K_ROOT=/path/to/SARDet_100K
  GPUS=1
  SEED=0
  RESUME_CKPT=          # optional checkpoint to resume from
  MAX_EPOCHS=            # optional override (e.g. 1 for quick verification)
  TRAIN_NUM_IMAGES=      # optional: sample N train images (subset)
  VAL_NUM_IMAGES=        # optional: sample N val images (subset)
  TRAIN_BATCH_SIZE=      # optional: override train_dataloader.batch_size
  TRAIN_NUM_WORKERS=     # optional: override train_dataloader.num_workers
  VAL_NUM_WORKERS=       # optional: override val/test dataloader num_workers
  EVAL_SPLITS=val[,test] # default: val

Outputs (always under <work_dir>/):
  - train.log
  - val_metrics.json (+ test_metrics.json if EVAL_SPLITS includes test)
  - test_val.log (+ test_test.log if EVAL_SPLITS includes test)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --resume)
      RESUME_CKPT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CONFIG}" || -z "${WORK_DIR}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"
CONFIG_ABS="$(realpath "${CONFIG}")"
WORK_DIR_ABS="$(realpath "${WORK_DIR}")"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  pick_gpus() {
    local need="$1"
    local q
    q="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)"
    if [[ -n "${CUDA_EXCLUDE_DEVICES}" ]]; then
      q="$(
        echo "${q}" | awk -F',' -v excl="${CUDA_EXCLUDE_DEVICES}" '
          BEGIN{
            n=split(excl, a, ",");
            for(i=1;i<=n;i++){
              gsub(/^[ \t]+|[ \t]+$/, "", a[i]);
              if(a[i]!="") ex[a[i]]=1;
            }
          }
          {
            idx=$1;
            gsub(/^[ \t]+|[ \t]+$/, "", idx);
            if(!(idx in ex)) print $0;
          }'
      )"
    fi

    if [[ "${need}" == "1" ]]; then
      echo "${q}" | sort -t, -k2 -nr | head -n 1 | cut -d, -f1 | tr -d ' '
    else
      echo "${q}" | sort -t, -k2 -nr | head -n "${need}" | cut -d, -f1 | tr -d ' ' | paste -sd, -
    fi
  }

  export CUDA_VISIBLE_DEVICES="$(pick_gpus "${GPUS}")"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

bash "${REPO_ROOT}/scripts/verify_env.sh"

TRAIN_ANN="${SARDET100K_ROOT}/Annotations/train.json"
VAL_ANN="${SARDET100K_ROOT}/Annotations/val.json"

if [[ -n "${TRAIN_NUM_IMAGES}" ]]; then
  TRAIN_ANN="${WORK_DIR_ABS}/train_subset_${TRAIN_NUM_IMAGES}_seed${SEED}.json"
  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/make_coco_subset.py" \
    --in-json "${SARDET100K_ROOT}/Annotations/train.json" \
    --out-json "${TRAIN_ANN}" \
    --num-images "${TRAIN_NUM_IMAGES}" \
    --seed "${SEED}"
fi

if [[ -n "${VAL_NUM_IMAGES}" ]]; then
  VAL_ANN="${WORK_DIR_ABS}/val_subset_${VAL_NUM_IMAGES}_seed${SEED}.json"
  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/make_coco_subset.py" \
    --in-json "${SARDET100K_ROOT}/Annotations/val.json" \
    --out-json "${VAL_ANN}" \
    --num-images "${VAL_NUM_IMAGES}" \
    --seed "${SEED}"
fi

CFG_OVERRIDES=(
  "randomness.seed=${SEED}"
  "train_dataloader.dataset.ann_file=${TRAIN_ANN}"
  "val_dataloader.dataset.ann_file=${VAL_ANN}"
  "val_evaluator.ann_file=${VAL_ANN}"
)

if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
  CFG_OVERRIDES+=("train_dataloader.batch_size=${TRAIN_BATCH_SIZE}")
fi
if [[ -n "${TRAIN_NUM_WORKERS}" ]]; then
  CFG_OVERRIDES+=("train_dataloader.num_workers=${TRAIN_NUM_WORKERS}")
fi
if [[ -n "${VAL_NUM_WORKERS}" ]]; then
  CFG_OVERRIDES+=(
    "val_dataloader.num_workers=${VAL_NUM_WORKERS}"
    "test_dataloader.num_workers=${VAL_NUM_WORKERS}"
  )
fi

if [[ -n "${MAX_EPOCHS}" ]]; then
  CFG_OVERRIDES+=(
    "train_cfg.max_epochs=${MAX_EPOCHS}"
    "train_cfg.val_interval=1"
    "default_hooks.logger.interval=20"
    "default_hooks.checkpoint.interval=1"
  )
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT}" ]]; then
  if [[ ! -f "${RESUME_CKPT}" ]]; then
    echo "Resume checkpoint not found: ${RESUME_CKPT}" >&2
    exit 1
  fi
  RESUME_CKPT_ABS="$(realpath "${RESUME_CKPT}")"
  RESUME_ARGS+=(--resume "${RESUME_CKPT_ABS}")
fi

if [[ "${GPUS}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/mmdet_train.py" \
    "${CONFIG_ABS}" \
    --work-dir "${WORK_DIR_ABS}" \
    "${RESUME_ARGS[@]}" \
    --cfg-options "${CFG_OVERRIDES[@]}" 2>&1 | tee "${WORK_DIR_ABS}/train.log"
else
  if [[ -z "${PORT:-}" ]]; then
    base_port="$(
      python -c 'import sys,zlib; print(20000 + (zlib.adler32(sys.argv[1].encode()) % 10000))' "${WORK_DIR_ABS}"
    )"
    for i in $(seq 0 49); do
      candidate_port="$((base_port + i))"
      if python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "${candidate_port}" >/dev/null 2>&1; then
        export PORT="${candidate_port}"
        break
      fi
    done
  fi
  echo "PORT=${PORT:-<unset>}"
  env -u NCCL_PROTO -u NCCL_P2P_LEVEL -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS -u NCCL_P2P_DISABLE -u NCCL_IB_DISABLE \
    conda run -n "${ENV_NAME}" python -m torch.distributed.run \
      --nproc_per_node "${GPUS}" \
      --master_addr "${MASTER_ADDR:-127.0.0.1}" \
      --master_port "${PORT}" \
      "${REPO_ROOT}/scripts/mmdet_train.py" \
      "${CONFIG_ABS}" \
      --launcher pytorch \
      "${RESUME_ARGS[@]}" \
      --work-dir "${WORK_DIR_ABS}" \
      --cfg-options "${CFG_OVERRIDES[@]}" 2>&1 | tee "${WORK_DIR_ABS}/train.log"
fi

CKPT="${WORK_DIR_ABS}/latest.pth"
if [[ ! -f "${CKPT}" && -n "${MAX_EPOCHS}" ]]; then
  CKPT="${WORK_DIR_ABS}/epoch_${MAX_EPOCHS}.pth"
fi
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(ls -1t "${WORK_DIR_ABS}"/best_*.pth "${WORK_DIR_ABS}"/*.pth "${WORK_DIR_ABS}"/best_*/*.pth 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "No checkpoint found under '${WORK_DIR_ABS}'" >&2
  exit 1
fi
echo "Checkpoint: ${CKPT}"

eval_split() {
  local split="$1"
  local ann="$2"
  local img_prefix="$3"
  local out_json="$4"
  local out_log="$5"

  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/mmdet_test_to_json.py" \
    --config "${CONFIG_ABS}" \
    --checkpoint "${CKPT}" \
    --work-dir "${WORK_DIR_ABS}" \
    --out-json "${out_json}" \
    --cfg-options \
    "test_dataloader.dataset.ann_file=${ann}" \
    "test_dataloader.dataset.data_prefix.img=${img_prefix}" \
    "test_evaluator.ann_file=${ann}" \
    "model.test_cfg.score_thr=0.0" 2>&1 | tee "${out_log}"

  test -s "${out_json}"
  echo "OK: ${out_json}"
}

IFS=',' read -r -a SPLITS <<<"${EVAL_SPLITS}"
for split in "${SPLITS[@]}"; do
  case "${split}" in
    val)
      eval_split "val" "${VAL_ANN}" "JPEGImages/val/" "${WORK_DIR_ABS}/val_metrics.json" "${WORK_DIR_ABS}/test_val.log"
      ;;
    test)
      eval_split "test" "${SARDET100K_ROOT}/Annotations/test.json" "JPEGImages/test/" "${WORK_DIR_ABS}/test_metrics.json" "${WORK_DIR_ABS}/test_test.log"
      ;;
    *)
      echo "Unknown split in EVAL_SPLITS: '${split}' (expected val,test)" >&2
      exit 1
      ;;
  esac
done
