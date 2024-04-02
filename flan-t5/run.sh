#!/bin/bash

mkdir -p runs

python main.py \
    --train_data \
    --dev_data \
    --test_data \
    --model_name google/flan-t5-small \
    --tokenizer google/flan-t5-small \
    --num_epoch 50 \
    --batch_size 32 \
    --optimizer AdamW \
    --shuffle \
    --lr 0.00001 \
    --momentum 0.8 \
    --process_data multilabel_only \
    --include_target \
    --include_observer \
    --text_prefix Multilabel\ classification \
    --train_to_span \
    --val_to_span \
    --test_to_span \
    --pred_num_beams 5 \
    --pred_max_length 80 \
    --pred_repetition_penalty 2.5 \
    --pred_length_penalty 1.0 \
    --show_dataset_stats \
    --random_seed 0 \
    --cuda 1