#!/bin/bash

mkdir -p runs

python baseline_sim.py \
    --train_data ${} \
    --dev_data ${} \
    --test_data ${} \
    --labels Pleasantness Anticipated\ Effort Certainty Objective\ Experience Self-Other\ Agency Situational\ Control Advice \
    --metrics total_accuracy total_recall total_precision f1_score \
    --random_seed 0 \
    --device cuda \
    --cuda 1