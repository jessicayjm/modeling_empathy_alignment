#!/bin/bash

mkdir -p runs

python quick_preprocess_at_inference_time.py \
    --test_data ${} \
    --tokenizer sentence-transformers/all-mpnet-base-v2 \
    --labels Pleasantness Anticipated\ Effort Certainty Objective\ Experience Self-Other\ Agency Situational\ Control Advice \
    --target_num_neighbors 0 \
    --target_context_pos succeeding \
    --observer_num_neighbors 0 \
    --observer_context_pos succeeding \
    --dataset_save_path ${}