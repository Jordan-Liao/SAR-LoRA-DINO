#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
This repo uses MMDetection inference/visualization scripts under `visualization/`.

Examples:
  bash visualization/visualize_sardet.sh --config <cfg.py> --checkpoint <ckpt.pth> --out-dir artifacts/visualizations/painted
  OUT_ROOT=artifacts/visualizations/VR bash visualization/export_sardet_vr.sh --name <name> --config <cfg.py> --checkpoint <ckpt.pth>
EOF

