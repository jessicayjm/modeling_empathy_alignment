#!/bin/bash

mkdir -p runs


# train
python train.py \
    --train_data ${} \
    --dev_data ${} \
    --test_data ${} \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --tokenizer sentence-transformers/all-mpnet-base-v2 \
    --labels Pleasantness Anticipated\ Effort Certainty Objective\ Experience Self-Other\ Agency Situational\ Control Advice \
    --num_epoch 30 \
    --batch_size 1 \
    --lr 1e-6 \
    --loss mse_loss \
    --pred_threshold 0.3 \
    --optimizer AdamW \
    --shuffle \
    --target_num_neighbors 0 \
    --target_context_pos succeeding \
    --observer_num_neighbors 0 \
    --observer_context_pos succeeding \
    --metrics total_accuracy total_recall total_precision f1_score \
    --random_seed 0 \
    --device cuda \
    --cuda 0 \
    --train_to_alignment \
    --dev_to_alignment \
    --test_to_alignment \


# inference
python train.py \
    --test_data ${} \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --tokenizer sentence-transformers/all-mpnet-base-v2 \
    --load_checkpoint ${} \
    --evaluate_only \
    --labels Pleasantness Anticipated\ Effort Certainty Objective\ Experience Self-Other\ Agency Situational\ Control Advice \
    --batch_size 2 \
    --pred_threshold 0.3 \
    --target_num_neighbors 0 \
    --target_context_pos succeeding \
    --observer_num_neighbors 0 \
    --observer_context_pos succeeding \
    --metrics total_accuracy total_recall total_precision f1_score \
    --random_seed 0 \
    --device cuda \
    --cuda 0 \
    --test_to_alignment



