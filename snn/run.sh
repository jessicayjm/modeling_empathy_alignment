#!/bin/bash

mkdir -p runs

batch_size=8
python train.py \
    --train_data \
    --dev_data \
    --test_data \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --tokenizer sentence-transformers/all-mpnet-base-v2 \
    --load_checkpoin \
    --evaluate_only \
    --labels Pleasantness Anticipated\ Effort Certainty Objective\ Experience Self-Other\ Agency Situational\ Control Advice \
    --num_epoch 300 \
    --batch_size ${batch_size} \
    --lr 1e-7 \
    --loss mse_loss \
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
    --metrics total_accuracy \
    --random_seed 0 \
    --device cpu \
    --test_to_alignment \
    --output_alignments_path \
    --load_processed_dataset




