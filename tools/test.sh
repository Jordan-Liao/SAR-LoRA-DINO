#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-sar_lora_dino}"

usage() {
  cat <<'EOF'
Usage:
  bash tools/test.sh --config <config.py> --checkpoint <ckpt.pth> --work-dir <dir> --out-json <metrics.json> [--cfg-options k=v ...]
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python tools/test_to_json.py "$@"

