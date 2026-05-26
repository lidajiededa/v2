#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install -e "${ROOT}/third_party/vllm-0.18.0"
python -m pip install -e "${ROOT}/third_party/vllm-ascend-0.18.0"
python -m pip install -e "${ROOT}"

echo "Installed editable vLLM, vllm-ascend, and vllm-pangu-v2-moe from ${ROOT}"

