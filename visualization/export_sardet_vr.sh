#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-sar_lora_dino}"
SARDET100K_ROOT="${SARDET100K_ROOT:-${REPO_ROOT}/data/SARDet_100K}"
OUT_ROOT="${OUT_ROOT:-artifacts/visualizations/VR}"
SPLITS="${SPLITS:-val,test}"
EXPORT_VIS="${EXPORT_VIS:-1}"
CUDA_EXCLUDE_DEVICES="${CUDA_EXCLUDE_DEVICES:-}"

CONFIG=""
CHECKPOINT=""
NAME=""

usage() {
  cat <<'EOF'
Usage:
  bash visualization/export_sardet_vr.sh --name <tag> --config <config.py> --checkpoint <ckpt.pth>

Environment variables:
  ENV_NAME=sar_lora_dino
  SARDET100K_ROOT=/path/to/SARDet_100K
  OUT_ROOT=artifacts/visualizations/VR
  SPLITS=val,test
  EXPORT_VIS=1  # set 0 to skip saving per-image visualizations (metrics + predictions only)

Outputs (per split, under OUT_ROOT/<tag>/<split>/):
  - metrics.json
  - predictions.pkl
  - export.log
  - (if EXPORT_VIS=1) visualizations under: <split>/<timestamp>/vis/

Notes:
  - If CUDA_VISIBLE_DEVICES is unset, this script auto-picks the GPU with most free memory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
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

if [[ -z "${NAME}" || -z "${CONFIG}" || -z "${CHECKPOINT}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
OUT_ROOT_ABS="$(realpath "${OUT_ROOT}")"
CONFIG_ABS="$(realpath "${CONFIG}")"
CHECKPOINT_ABS="$(realpath "${CHECKPOINT}")"
MODEL_OUT="${OUT_ROOT_ABS}/${NAME}"
mkdir -p "${MODEL_OUT}"

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

IFS=',' read -r -a SPLIT_LIST <<<"${SPLITS}"
for split in "${SPLIT_LIST[@]}"; do
  case "${split}" in
    val)
      ANN="${SARDET100K_ROOT}/Annotations/val.json"
      IMG_PREFIX="JPEGImages/val/"
      ;;
    test)
      ANN="${SARDET100K_ROOT}/Annotations/test.json"
      IMG_PREFIX="JPEGImages/test/"
      ;;
    *)
      echo "Unknown split in SPLITS: '${split}' (expected val,test)" >&2
      exit 1
      ;;
  esac

  SPLIT_OUT="${MODEL_OUT}/${split}"
  mkdir -p "${SPLIT_OUT}"

  METRICS_JSON="${SPLIT_OUT}/metrics.json"
  PREDS_PKL="${SPLIT_OUT}/predictions.pkl"
  LOG="${SPLIT_OUT}/export.log"

  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/visualization/mmdet_test_export.py" \
    --config "${CONFIG_ABS}" \
    --checkpoint "${CHECKPOINT_ABS}" \
    --work-dir "${SPLIT_OUT}" \
    --out-metrics "${METRICS_JSON}" \
    --out-pkl "${PREDS_PKL}" \
    $( [[ "${EXPORT_VIS}" != "0" ]] && echo "--show-dir" && echo "vis" ) \
    --cfg-options \
    "test_dataloader.dataset.ann_file=${ANN}" \
    "test_dataloader.dataset.data_prefix.img=${IMG_PREFIX}" \
    "test_evaluator.ann_file=${ANN}" \
    "model.test_cfg.score_thr=0.0" 2>&1 | tee "${LOG}"

  test -s "${METRICS_JSON}"
  test -s "${PREDS_PKL}"
  echo "OK: ${SPLIT_OUT}"
done

echo "OK: ${MODEL_OUT}"
