#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-sar_lora_dino}"
SARDET100K_ROOT="${SARDET100K_ROOT:-${REPO_ROOT}/data/SARDet_100K}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
TRAIN_NUM_IMAGES="${TRAIN_NUM_IMAGES:-200}"
VAL_NUM_IMAGES="${VAL_NUM_IMAGES:-50}"
SEED="${SEED:-0}"
CUDA_EXCLUDE_DEVICES="${CUDA_EXCLUDE_DEVICES:-}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_DISABLE_TELEMETRY

CONFIG=""
WORK_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_sardet_smoke_cfg.sh --config <config.py> --work-dir <dir>

Environment variables:
  ENV_NAME=sar_lora_dino
  SARDET100K_ROOT=/path/to/SARDet_100K
  MAX_EPOCHS=1
  TRAIN_NUM_IMAGES=200
  VAL_NUM_IMAGES=50
  SEED=0

Notes:
  - If CUDA_VISIBLE_DEVICES is unset, this script auto-picks the GPU with most free memory.
  - Writes: <work_dir>/smoke_train.log, <work_dir>/smoke_test.log, <work_dir>/smoke_metrics.json
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
  pick_gpu() {
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
    echo "${q}" | sort -t, -k2 -nr | head -n 1 | cut -d, -f1 | tr -d ' '
  }

  export CUDA_VISIBLE_DEVICES="$(pick_gpu)"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

bash "${REPO_ROOT}/scripts/verify_env.sh"

TRAIN_SUBSET_JSON="${WORK_DIR_ABS}/train_subset_${TRAIN_NUM_IMAGES}_seed${SEED}.json"
VAL_SUBSET_JSON="${WORK_DIR_ABS}/val_subset_${VAL_NUM_IMAGES}_seed${SEED}.json"

conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/make_coco_subset.py" \
  --in-json "${SARDET100K_ROOT}/Annotations/train.json" \
  --out-json "${TRAIN_SUBSET_JSON}" \
  --num-images "${TRAIN_NUM_IMAGES}" \
  --seed "${SEED}"

conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/make_coco_subset.py" \
  --in-json "${SARDET100K_ROOT}/Annotations/val.json" \
  --out-json "${VAL_SUBSET_JSON}" \
  --num-images "${VAL_NUM_IMAGES}" \
  --seed "${SEED}"

conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/mmdet_train.py" \
  "${CONFIG_ABS}" \
  --work-dir "${WORK_DIR_ABS}" \
  --cfg-options \
  train_dataloader.dataset.ann_file="${TRAIN_SUBSET_JSON}" \
  val_dataloader.dataset.ann_file="${VAL_SUBSET_JSON}" \
  val_evaluator.ann_file="${VAL_SUBSET_JSON}" \
  train_cfg.max_epochs="${MAX_EPOCHS}" \
  train_cfg.val_interval=1 \
  default_hooks.logger.interval=20 \
  default_hooks.checkpoint.interval=1 2>&1 | tee "${WORK_DIR_ABS}/smoke_train.log"

CKPT="${WORK_DIR_ABS}/latest.pth"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${WORK_DIR_ABS}/epoch_${MAX_EPOCHS}.pth"
fi
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(ls -1t "${WORK_DIR_ABS}"/*.pth "${WORK_DIR_ABS}"/best_*/*.pth 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "No checkpoint found under '${WORK_DIR_ABS}'" >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/mmdet_test_to_json.py" \
  --config "${CONFIG_ABS}" \
  --checkpoint "${CKPT}" \
  --work-dir "${WORK_DIR_ABS}" \
  --out-json "${WORK_DIR_ABS}/smoke_metrics.json" \
  --cfg-options \
  test_dataloader.dataset.ann_file="${VAL_SUBSET_JSON}" \
  test_dataloader.dataset.data_prefix.img="JPEGImages/val/" \
  test_evaluator.ann_file="${VAL_SUBSET_JSON}" \
  model.test_cfg.score_thr=0.0 2>&1 | tee "${WORK_DIR_ABS}/smoke_test.log"

test -s "${WORK_DIR_ABS}/smoke_metrics.json"
echo "OK: ${WORK_DIR_ABS}/smoke_metrics.json"
