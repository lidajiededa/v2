#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install -e "${ROOT}"

echo "Installed editable vllm-pangu-v2-moe from ${ROOT}"

