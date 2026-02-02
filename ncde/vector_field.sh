#!/bin/bash

# Vector field analysis script
# Usage: ./vector_field.sh <output_dir> <data_path>

OUTPUT_DIR=${1:-"wandb/run/files"}
DATA_PATH=${2:-"../data/mimiciii_preprocessed_final.csv"}

python vector_field.py \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$DATA_PATH"
