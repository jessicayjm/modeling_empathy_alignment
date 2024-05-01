#!/bin/bash

python preprocess.py \
    --test_data ${} \
    --cased \
    --labels Pleasantness Anticipated\ Effort Certainty Objective\ Experience Self-Other\ Agency Situational\ Control Advice Trope \
    --include_target \
    --single_label \
    --show_dataset_stats \
    --saving_folder ${}
