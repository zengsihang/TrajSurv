#!/bin/bash

# Clustering analysis script for latent state analysis
# Usage: ./clustering.sh <output_dir> <data_path> <sofa_path>

OUTPUT_DIR=${1:-"wandb/run/files"}
DATA_PATH=${2:-"../data/mimiciii_preprocessed_final.csv"}
SOFA_PATH=${3:-"../data/mimic_sofa_scores_detail_40.csv"}
HADM_ID=${4:-182878}
SOFA_NUM=${5:--1}

# Run all latent state analysis
python clustering_latent_state.py \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$DATA_PATH" \
    --sofa_path "$SOFA_PATH" \
    --which_time all \
    --hadm_id "$HADM_ID" \
    --sofa_num "$SOFA_NUM"

# Run last time point analysis
python clustering_latent_state.py \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$DATA_PATH" \
    --sofa_path "$SOFA_PATH" \
    --which_time last \
    --hadm_id "$HADM_ID" \
    --sofa_num "$SOFA_NUM"

# Run trajectory analysis
python clustering_latent_state.py \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$DATA_PATH" \
    --sofa_path "$SOFA_PATH" \
    --which_time alltrajectory \
    --hadm_id "$HADM_ID" \
    --sofa_num "$SOFA_NUM"

# Run correlation analysis (requires existed data)
python clustering_latent_state.py \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$DATA_PATH" \
    --sofa_path "$SOFA_PATH" \
    --which_time allcorr \
    --hadm_id "$HADM_ID" \
    --sofa_num "$SOFA_NUM" \
    --existed_data
