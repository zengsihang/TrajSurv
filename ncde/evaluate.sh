#!/bin/bash

# Evaluation script for NCDESurv model
# Usage: ./evaluate.sh <output_dir> [--use_abs_surv]

OUTPUT_DIR=${1:-"wandb/run/files"}
USE_ABS_SURV=${2:-""}

if [ -n "$USE_ABS_SURV" ]; then
    python evaluate.py --output_dir "$OUTPUT_DIR" --use_abs_surv
else
    python evaluate.py --output_dir "$OUTPUT_DIR"
fi
