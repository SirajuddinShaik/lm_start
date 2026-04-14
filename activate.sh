#!/bin/bash
# Activate the model's virtual environment
source "$(dirname "$0")/.venv/bin/activate"
echo "Activated venv for $(basename $(dirname "$0"))"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
