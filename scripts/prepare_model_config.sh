#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: prepare_model_config.sh /path/to/model_dir" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$1"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Missing ${MODEL_DIR}/config.json" >&2
  exit 1
fi

cp "${MODEL_DIR}/config.json" "${MODEL_DIR}/config.json.before_pangu_v2_moe_adapter"
python "${ROOT}/scripts/normalize_config.py" \
  "${MODEL_DIR}/config.json.before_pangu_v2_moe_adapter" \
  "${MODEL_DIR}/config.json"

echo "Normalized ${MODEL_DIR}/config.json"
echo "Backup: ${MODEL_DIR}/config.json.before_pangu_v2_moe_adapter"

