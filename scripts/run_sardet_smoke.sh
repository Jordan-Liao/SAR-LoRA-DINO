#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-sar_lora_dino}"
SARDET100K_ROOT="${SARDET100K_ROOT:-${REPO_ROOT}/data/SARDet_100K}"
CUDA_EXCLUDE_DEVICES="${CUDA_EXCLUDE_DEVICES:-}"

cd "${REPO_ROOT}"

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

WORK_DIR="${REPO_ROOT}/artifacts/work_dirs/sardet_smoke"
SUBSET_DIR="${WORK_DIR}/subsets"
TRAIN_SUBSET_JSON="${SUBSET_DIR}/train_200.json"
VAL_SUBSET_JSON="${SUBSET_DIR}/val_50.json"
CONFIG="${REPO_ROOT}/mmdet_toolkit/local_configs/SARDet/smoke/retinanet_r50_sardet_smoke.py"

mkdir -p "${WORK_DIR}" "${SUBSET_DIR}"

bash "${REPO_ROOT}/scripts/verify_env.sh"

conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/make_coco_subset.py" \
  --in-json "${SARDET100K_ROOT}/Annotations/train.json" \
  --out-json "${TRAIN_SUBSET_JSON}" \
  --num-images 200 \
  --seed 0

conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/make_coco_subset.py" \
  --in-json "${SARDET100K_ROOT}/Annotations/val.json" \
  --out-json "${VAL_SUBSET_JSON}" \
  --num-images 50 \
  --seed 0

conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/mmdet_train.py" \
  "${CONFIG}" \
  --work-dir "${WORK_DIR}" 2>&1 | tee "${WORK_DIR}/smoke_train.log"

CKPT="${WORK_DIR}/latest.pth"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${WORK_DIR}/epoch_1.pth"
fi
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(ls -1t "${WORK_DIR}"/*.pth "${WORK_DIR}"/best_*/*.pth 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "No checkpoint found under '${WORK_DIR}'" >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/mmdet_test_to_json.py" \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --work-dir "${WORK_DIR}" \
  --out-json "${WORK_DIR}/smoke_metrics.json" 2>&1 | tee "${WORK_DIR}/smoke_test.log"

test -s "${WORK_DIR}/smoke_metrics.json"
echo "OK: ${WORK_DIR}/smoke_metrics.json"
