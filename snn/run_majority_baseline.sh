#!/bin/bash

mkdir -p runs

python majority_baseline.py \
    --test_data \
    --labels Pleasantness Anticipated\ Effort Certainty Objective\ Experience Self-Other\ Agency Situational\ Control Advice \
    --batch_size 8 \
    --tokenizer roberta-large \
    --target_num_neighbors 0 \
    --target_context_pos succeeding \
    --observer_num_neighbors 0 \
    --observer_context_pos succeeding \
    --metrics total_accuracy total_recall total_precision f1_score \
    --random_seed 0 \
    --device cuda \
    --cuda 1