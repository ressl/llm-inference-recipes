#!/bin/bash
set -euo pipefail
python3 /opt/recipe/gpu_preflight.py
exec vllm serve "$@"
