#!/bin/bash

mkdir -p runs

# BERT-based model
# python train.py \
#     --train_data ${} \
#     --dev_data ${} \
#     --test_data ${} \
#     --model_name bert-large-uncased \
#     --num_epoch 200 \
#     --batch_size 32 \
#     --loss simple_cross_entropy_loss \
#     --optimizer AdamW \
#     --shuffle \
#     --lr 1e-6 \
#     --momentum 0.8 \
#     --lmbda 0 \
#     --metrics total_accuracy macro_f1 macro_recall macro_precision per_label_recall per_label_precision \
#     --process_data feed_single_sent \
#     --pred_threshold 0.5 \
#     --random_seed 0 \
#     --cuda 0

# python train.py \
#     --train_data ${} \
#     --dev_data ${} \
#     --test_data ${} \
#     --model_name roberta-large \
#     --tokenizer roberta-large \
#     --num_epoch 200 \
#     --batch_size 32 \
#     --loss simple_cross_entropy_loss \
#     --optimizer AdamW \
#     --shuffle \
#     --lr 1e-6 \
#     --momentum 0.8 \
#     --lmbda 0 \
#     --metrics total_accuracy macro_f1 macro_recall macro_precision per_label_recall per_label_precision \
#     --process_data feed_single_sent \
#     --pred_threshold 0.5 \
#     --random_seed 0 \
#     --cuda 0

# python train.py \
#     --train_data ${} \
#     --dev_data ${} \
#     --test_data ${} \
#     --model_name SpanBERT/spanbert-large-cased \
#     --num_epoch 200 \
#     --batch_size 32 \
#     --loss simple_cross_entropy_loss \
#     --optimizer AdamW \
#     --shuffle \
#     --lr 1e-6 \
#     --momentum 0.8 \
#     --lmbda 0 \
#     --metrics total_accuracy macro_f1 macro_recall macro_precision per_label_recall per_label_precision \
#     --process_data feed_single_sent \
#     --pred_threshold 0.5 \
#     --random_seed 0 \
#     --cuda 0

# python train.py \
#     --train_data ${} \
#     --dev_data ${} \
#     --test_data ${} \
#     --model_name microsoft/deberta-v3-large \
#     --tokenizer microsoft/deberta-v3-large \
#     --num_epoch 200 \
#     --batch_size 16 \
#     --loss simple_cross_entropy_loss \
#     --optimizer AdamW \
#     --shuffle \
#     --lr 1e-6 \
#     --momentum 0.8 \
#     --lmbda 0 \
#     --metrics total_accuracy macro_f1 macro_recall macro_precision per_label_recall per_label_precision \
#     --process_data feed_single_sent \
#     --pred_threshold 0.5 \
#     --random_seed 0 \
#     --cuda 0

# python train.py \
#     --train_data ${} \
#     --dev_data ${} \
#     --test_data ${} \
#     --load_checkpoint \
#     --model_name sentence-transformers/all-MiniLM-L6-v2 \
#     --tokenizer sentence-transformers/all-MiniLM-L6-v2 \
#     --num_epoch 200 \
#     --batch_size 16 \
#     --loss simple_cross_entropy_loss \
#     --optimizer AdamW \
#     --shuffle \
#     --lr 1e-6 \
#     --momentum 0.8 \
#     --lmbda 0 \
#     --metrics total_accuracy macro_f1 macro_recall macro_precision per_label_recall per_label_precision \
#     --process_data feed_single_sent \
#     --pred_threshold 0.5 \
#     --random_seed 0 \
#     --cuda 0
