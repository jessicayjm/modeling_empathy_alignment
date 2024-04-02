#!/bin/bash

mkdir -p runs

python train.py \
    --train_data \
    --dev_data \
    --test_data \
    --model_name distilroberta-base \
    --labels Pleasantness Anticipated\ Effort Certainty Objective\ Experience Self-Other\ Agency Situational\ Control Advice \
    --num_epoch 50 \
    --batch_size 8 \
    --lr 1e-6 \
    --loss binary_cross_entropy_loss \
    --pred_threshold 0.3 \
    --pos_weight 5 \
    --optimizer AdamW \
    --m 1 \
    --lmbda 0.8 \
    --shuffle \
    --target_num_neighbors 0 \
    --target_context_pos succeeding \
    --observer_num_neighbors 0 \
    --observer_context_pos succeeding \
    --metrics total_accuracy total_recall f1_score \
    --random_seed 0 \
    --device cuda \
    --cuda 3
    # --train_to_alignment \
    # --dev_to_alignment \
    # --test_to_alignment \

# python baseline.py \
#     --test_data \
#     --labels Pleasantness Anticipated\ Effort Certainty Objective\ Experience Self-Other\ Agency Situational\ Control Advice \
#     --batch_size 8 \
#     --tokenizer roberta-large \
#     --target_num_neighbors 0 \
#     --target_context_pos succeeding \
#     --observer_num_neighbors 0 \
#     --observer_context_pos succeeding \
#     --metrics total_accuracy total_recall f1_score \
#     --random_seed 0 \
#     --device cuda \
#     --cuda 1