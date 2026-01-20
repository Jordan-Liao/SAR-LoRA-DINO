#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-sar_lora_dino}"
SARDET100K_ROOT="${SARDET100K_ROOT:-${REPO_ROOT}/data/sardet100k}"
if [[ ! -e "${SARDET100K_ROOT}" && -e "${REPO_ROOT}/data/SARDet_100K" ]]; then
  SARDET100K_ROOT="${REPO_ROOT}/data/SARDet_100K"
fi

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_DISABLE_TELEMETRY

CONFIG=""
CHECKPOINT=""
OUT_DIR=""
SPLIT="val"
NUM_IMAGES="50"
SEED="0"
FULL="0"
CUDA_EXCLUDE_DEVICES="${CUDA_EXCLUDE_DEVICES:-}"

usage() {
  cat <<'EOF'
Usage:
  bash visualization/visualize_sardet.sh --config <config.py> --checkpoint <ckpt.pth> --out-dir <dir> [--split val|test] [--num-images N] [--seed S] [--full]

Notes:
  - This uses `visualization/mmdet_test_export.py` with `--show-dir` to export painted detections.
  - By default, this visualizes a small COCO subset (`--split val --num-images 50`) for faster export.
  - If CUDA_VISIBLE_DEVICES is unset, this script auto-picks the GPU with most free memory.
  - Output is written under: <out_dir>/<timestamp>/vis/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --num-images)
      NUM_IMAGES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --full)
      FULL="1"
      shift 1
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

if [[ -z "${CONFIG}" || -z "${CHECKPOINT}" || -z "${OUT_DIR}" ]]; then
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

mkdir -p "${OUT_DIR}"
CONFIG_ABS="$(realpath "${CONFIG}")"
CHECKPOINT_ABS="$(realpath "${CHECKPOINT}")"
OUT_DIR_ABS="$(realpath "${OUT_DIR}")"

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

CFG_OVERRIDES=()
case "${SPLIT}" in
  val)
    ANN_IN="${SARDET100K_ROOT}/Annotations/val.json"
    IMG_PREFIX="JPEGImages/val/"
    ;;
  test)
    ANN_IN="${SARDET100K_ROOT}/Annotations/test.json"
    IMG_PREFIX="JPEGImages/test/"
    ;;
  *)
    echo "Unknown --split: '${SPLIT}' (expected val|test)" >&2
    exit 1
    ;;
esac

if [[ "${FULL}" == "1" ]]; then
  CFG_OVERRIDES+=(
    "test_dataloader.dataset.ann_file=${ANN_IN}"
    "test_dataloader.dataset.data_prefix.img=${IMG_PREFIX}"
    "test_evaluator.ann_file=${ANN_IN}"
    "model.test_cfg.score_thr=0.0"
  )
else
  SUBSET_JSON="${OUT_DIR_ABS}/vis_subset_${SPLIT}_${NUM_IMAGES}_seed${SEED}.json"
  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/make_coco_subset.py" \
    --in-json "${ANN_IN}" \
    --out-json "${SUBSET_JSON}" \
    --num-images "${NUM_IMAGES}" \
    --seed "${SEED}"

  CFG_OVERRIDES+=(
    "test_dataloader.dataset.ann_file=${SUBSET_JSON}"
    "test_dataloader.dataset.data_prefix.img=${IMG_PREFIX}"
    "test_evaluator.ann_file=${SUBSET_JSON}"
    "model.test_cfg.score_thr=0.0"
  )
fi

TEST_CMD=(
  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/visualization/mmdet_test_export.py"
  --config "${CONFIG_ABS}"
  --checkpoint "${CHECKPOINT_ABS}"
  --work-dir "${OUT_DIR_ABS}"
  --out-metrics "${OUT_DIR_ABS}/metrics.json"
  --out-pkl "${OUT_DIR_ABS}/predictions.pkl"
  --show-dir vis
)
if [[ "${#CFG_OVERRIDES[@]}" -gt 0 ]]; then
  TEST_CMD+=(--cfg-options "${CFG_OVERRIDES[@]}")
fi

"${TEST_CMD[@]}" 2>&1 | tee "${OUT_DIR_ABS}/visualize.log"

echo "OK: ${OUT_DIR_ABS}"
