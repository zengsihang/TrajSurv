#!/bin/bash

# Training script for NCDESurv model
# Usage: ./train.sh <data_path> [additional_args]

DATA_PATH=${1:-"../data/mimiciii_preprocessed_final.csv"}

python main.py \
    --data_path "$DATA_PATH" \
    --hidden_channels 64 \
    --k 4 \
    --dist Weibull \
    --lr 3e-4 \
    --batch_size 64 \
    --device cuda:0 \
    --head cox \
    --save_interval 5 \
    --layer_num 3 \
    --nonlinear \
    --pairwise \
    --seed 99 \
    --contrastive \
    --contrastive_weight 1.0 \
    --separate \
    --time_decay \
    --no_ffn \
    --weight_decay_temp 30.0 \
    --sofa_derivative_coeff 20.0 \
    --train_ratio 0.7 \
    --valid_ratio 0.1
