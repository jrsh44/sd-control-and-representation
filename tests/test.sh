#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
cd "$(dirname "$0")/.."
exec .venv/bin/pytest "$@"
