#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${REPO_ROOT}/configs/sardet100k/smoke/retinanet_r50_sardet_smoke.py"
WORK_DIR="${REPO_ROOT}/artifacts/work_dirs/sardet_smoke"

bash "${REPO_ROOT}/scripts/run_sardet_smoke_cfg.sh" \
  --config "${CONFIG}" \
  --work-dir "${WORK_DIR}"
