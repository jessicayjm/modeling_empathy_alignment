# Modeling Empathetic Alignment in Conversation

This repo contains all code for paper [Modeling Empathetic Alignment in Conversation]().

#### General Work-flow

Two models are performed for the final prediction of whether two appraisal aligned or not:

1. Appraisal prediction
2. Alignment prediction given appraisals

#### Input Format

The input format is the same for both tasks.

The input file is a `json` file. 

Easier to prepare the data by creating a `pd.Dataframe` with the following labels and convert it to `json` file:

| id   | target_id | observer_id | parent_id | subreddit | target_text | observer_text | distress_score | condolence_score | empathy_score | full_text | spans                      | alignments                                                   |
| ---- | --------- | ----------- | --------- | --------- | ----------- | ------------- | -------------- | ---------------- | ------------- | --------- | -------------------------- | ------------------------------------------------------------ |
| int  | string    | string      | string    | string    | string      | string        | float          | float            | float         | string    | [start, end, label], ... ] | [[(target_start, target_end), (observer_start, observer_end)], ...] |

Note:  For `spans` and `alignments` entry, `start` and `end` are set relative to `full_text`  which contains prefixes for `target_text` and `observer_text` with the format of:

`target:\n\n${target_text}\n\nobserver:\n\n${observer_text}`

#### Appraisal Prediction

##### Preprocess data

Command is in `preprocess/run.sh`.

The output file is a `pickle` file that can be directly used for training appraisal models.

##### Appraisal models

###### Baselines

Commands are in `baseline/run.sh`.

Note: For `majority_baseline`, `Self-other agency` is always predicted.

###### BERT models

Commands are in `bert/run.sh`. Modify for your own use (data paths, hyper-parameters etc.)

All available arguments are at the top of `bert/train.py`.

`evaluation_only` not available.

###### OpenPrompt models

Commands are in `openprompt/run.sh`. Modify for your own use (data paths, hyper-parameters etc.)

All available arguments are at the top of `openprompt/train.py`.

`evaluation_only` available.

#### Alignment Prediction

##### Preprocess data

You can:

1. Directly use the `json` input format in training script OR;

2. Preprocess the data with `snn/run_quick_preprocess.sh` and specify the argument `--load_processed_dataset` in the training script.

##### Train alignment models

Commands are in `snn/run.sh`. Modify for your own use (data paths, hyper-parameters etc.)

All available arguments are at the top of `snn/train.py`.

`evaluation_only` available.

#### Citation

