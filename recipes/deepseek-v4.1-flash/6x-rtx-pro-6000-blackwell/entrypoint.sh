#!/bin/bash
set -euo pipefail
if [[ "${RECIPE_DSV41_FP4_CACHE:-0}" == 1 && ! -f /opt/recipe/fp4-enabled ]]; then
    echo "The fp4 profile requires an image built with ENABLE_FP4_CACHE=1" >&2
    exit 1
fi
if [[ -f /opt/recipe/fp4-enabled && "${RECIPE_DSV41_FP4_CACHE:-0}" != 1 ]]; then
    echo "This FP4 image requires the fp4 profile" >&2
    exit 1
fi
python3 /opt/recipe/gpu_preflight.py
exec vllm serve "$@"
