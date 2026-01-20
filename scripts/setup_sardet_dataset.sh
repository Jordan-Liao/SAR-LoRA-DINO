#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_LINK="${REPO_ROOT}/data/SARDet_100K"
ALT_DATA_LINK="${REPO_ROOT}/data/sardet100k"

is_valid_root() {
  local root="$1"
  [[ -n "${root}" ]] || return 1
  [[ -f "${root}/Annotations/train.json" ]] || return 1
  [[ -f "${root}/Annotations/val.json" ]] || return 1
  [[ -f "${root}/Annotations/test.json" ]] || return 1
  [[ -d "${root}/JPEGImages" ]] || return 1
  return 0
}

pick_root() {
  local candidates=(
    "${SARDET100K_ROOT:-}"
    "${ALT_DATA_LINK}"
    "${DATA_LINK}"
    "${HOME}/datasets/SARDet_100K/SARDet_100K"
    "${HOME}/datasets/SARDet_100K"
    "${HOME}/data/SARDet_100K/SARDet_100K"
    "${HOME}/data/SARDet_100K"
    "${HOME}/SARDet_100K"
  )

  local c
  for c in "${candidates[@]}"; do
    if is_valid_root "${c}"; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

ROOT="$(pick_root || true)"
if [[ -z "${ROOT}" ]]; then
  echo "Could not find SARDet_100K dataset root." >&2
  echo "Set SARDET100K_ROOT=/path/to/SARDet_100K (must contain Annotations/ and JPEGImages/)." >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/data"

if [[ -e "${ALT_DATA_LINK}" ]]; then
  echo "OK: ${ALT_DATA_LINK} already exists."
else
  ln -s "${ROOT}" "${ALT_DATA_LINK}"
  echo "Linked: ${ALT_DATA_LINK} -> ${ROOT}"
fi

echo "Suggested env:"
echo "  export SARDET100K_ROOT=\"${ALT_DATA_LINK}\""
