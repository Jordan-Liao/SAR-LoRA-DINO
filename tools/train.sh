#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-sar_lora_dino}"

usage() {
  cat <<'EOF'
Usage:
  bash tools/train.sh <config.py> --work-dir <dir> [--resume <ckpt.pth>] [--cfg-options k=v ...]
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

CONFIG="$1"
shift

conda run -n "${ENV_NAME}" python tools/train.py "${CONFIG}" "$@"

