#!/bin/bash

mkdir -p runs

# training
# python train.py \
#     --train_data ${} \
#     --dev_data ${} \
#     --test_data ${} \
#     --prompt_file ./prompts/prompt1 \
#     --model_family bert \
#     --model_name bert-large-uncased \
#     --num_epoch 200 \
#     --batch_size 16 \
#     --loss simple_cross_entropy_loss \
#     --optimizer AdamW \
#     --shuffle \
#     --lr 1e-6 \
#     --momentum 0.8 \
#     --lmbda 0 \
#     --max_seq_length 512 \
#     --decoder_max_length 3 \
#     --truncate_method head \
#     --metrics total_accuracy macro_f1 per_label_recall per_label_precision \
#     --train_to_span \
#     --dev_to_span \
#     --test_to_span \
#     --random_seed 0 \
#     --cuda 1

# python train.py \
#     --train_data ${} \
#     --dev_data ${} \
#     --test_data ${} \
#     --prompt_file ./prompts/prompt1 \
#     --model_family roberta \
#     --model_name roberta-large \
#     --tokenizer roberta-large \
#     --load_trained_mode \
#     --num_epoch 200 \
#     --batch_size 8 \
#     --loss simple_cross_entropy_loss \
#     --optimizer AdamW \
#     --shuffle \
#     --lr 1e-7 \
#     --momentum 0.8 \
#     --lmbda 0 \
#     --max_seq_length 512 \
#     --decoder_max_length 3 \
#     --truncate_method head \
#     --metrics total_accuracy macro_f1 per_label_recall per_label_precision \
#     --train_to_span \
#     --dev_to_span \
#     --test_to_span \
#     --random_seed 0 \
#     --cuda 0

# python train.py \
#     --train_data ${} \
#     --dev_data ${} \
#     --test_data ${} \
#     --prompt_file ./prompts/prompt1 \
#     --model_family t5 \
#     --model_name t5-large \
#     --num_epoch 200 \
#     --batch_size 8 \
#     --loss simple_cross_entropy_loss \
#     --optimizer AdamW \
#     --shuffle \
#     --lr 1e-6 \
#     --momentum 0.8 \
#     --lmbda 0 \
#     --max_seq_length 512 \
#     --decoder_max_length 3 \
#     --truncate_method head \
#     --metrics total_accuracy macro_f1 per_label_recall per_label_precision \
#     --train_to_span \
#     --dev_to_span \
#     --test_to_span \
#     --random_seed 0 \
#     --cuda 0

# test
# python train.py \
#     --test_data ${} \
#     --prompt_file ./prompts/prompt1 \
#     --model_family bert \
#     --model_name bert-large-uncased \
#     --load_checkpoint ${} \
#     --batch_size 2\
#     --max_seq_length 512 \
#     --decoder_max_length 3 \
#     --truncate_method head \
#     --metrics total_accuracy macro_f1 macro_recall macro_precision per_label_recall per_label_precision \
#     --evaluate_only \
#     --random_seed 0 \
#     --cuda 0

# python train.py \
#     --test_data ${} \
#     --prompt_file ./prompts/prompt1 \
#     --model_family roberta \
#     --model_name roberta-large \
#     --load_checkpoint ${} \
#     --batch_size 2\
#     --max_seq_length 512 \
#     --decoder_max_length 3 \
#     --truncate_method head \
#     --metrics total_accuracy macro_f1 macro_recall macro_precision per_label_recall per_label_precision \
#     --evaluate_only \
#     --random_seed 0 \
#     --cuda 0

# python train.py \
#     --test_data ${} \
#     --prompt_file ./prompts/prompt1 \
#     --model_family t5 \
#     --model_name t5-large \
#     --load_checkpoint ${} \
#     --batch_size 2\
#     --max_seq_length 512 \
#     --decoder_max_length 3 \
#     --truncate_method head \
#     --metrics total_accuracy macro_f1 macro_recall macro_precision per_label_recall per_label_precision \
#     --evaluate_only \
#     --random_seed 0 \
#     --cuda 0

