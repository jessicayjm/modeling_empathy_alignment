import os
import re
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import TQDMProgressBar

import sys
sys.path.append('..')
from variables import labels, metrics, loss_funcs
from utils import save_checkpoint, prepare_saving
# from utils import map_tokens2spans, merge_spans, combine_sents
# from utils import predict, get_metrics
from dataset import MultilabelOnlyDataset, MultilabelwithSpansDataset

from transformers import AutoTokenizer
from train import FLAN_T5

# parsers
parser = argparse.ArgumentParser(description='T5 models training')

# model specifications
parser.add_argument('--train_data', type=str)
parser.add_argument('--dev_data', type=str)
parser.add_argument('--test_data', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--optimizer', type=str, choices=['AdamW', 'SGD'], default='AdamW')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--tokenizer', type=str)
parser.add_argument('--process_data', type=str, choices=['multilabel_only']) # 'multilabel_with_spans' code not completed
parser.add_argument('--include_target', action='store_true', default=False)
parser.add_argument('--include_observer', action='store_true', default=False)
parser.add_argument('--text_prefix', type=str)
parser.add_argument('--labels', nargs='+', choices=labels, default=labels)
parser.add_argument('--metrics', nargs='+', choices=metrics, default=metrics)
parser.add_argument('--num_sanity_val_steps', type=int, default=0, help="Sanity check runs n validation batches before starting the training routine. Set it to -1 to run all batches in all validation dataloaders")
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
# parser.add_argument('--num_devices', type=int, default=1)

# model prediction
parser.add_argument('--pred_num_beams', type=int, default=5)
parser.add_argument('--pred_max_length', type=int, default=80)
parser.add_argument('--pred_repetition_penalty', type=float, default=2.5)
parser.add_argument('--pred_length_penalty', type=float, default=1.0)

# checkpoints
parser.add_argument('--checkpoint_save_path', type=str, default='checkpoints')
parser.add_argument('--load_checkpoint', type=str, default=None, help='need full path from project root directory') 

# output format specification
parser.add_argument('--train_to_span', action='store_true', default=False)
parser.add_argument('--val_to_span', action='store_true', default=False)
parser.add_argument('--test_to_span', action='store_true', default=False)
parser.add_argument('--output_spans_path', type=str, default='output_spans')

# loggings
parser.add_argument('--show_dataset_stats', action='store_true', default=True)
parser.add_argument('--saving_folder', type=str, default='./runs')

# cuda
parser.add_argument('--cuda', nargs='+', choices=[0,1,2], type=int)

args = parser.parse_args()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _main():
    now, root_save_path = prepare_saving(args.saving_folder, args.model_name.replace("/", "_"), args.checkpoint_save_path, args.output_spans_path)
    logfile_path = f'{root_save_path}/output.log'
    logging.basicConfig(filename=logfile_path, level=logging.INFO)
    
    seed_everything(args.random_seed, workers=True)
    
    start_time = datetime.now()

    # load in datafile
    train_df = pd.read_json(args.train_data)
    val_df = pd.read_json(args.dev_data)
    test_df = pd.read_json(args.test_data)

    # testing purpose
    # train_df = train_df.head(32)
    # val_df = val_df.head(8)
    # test_df = test_df.head(8)

    dataset_params = {
        'tokenizer': AutoTokenizer.from_pretrained(args.tokenizer),
        'process_data': args.process_data,
        'include_target': args.include_target,
        'include_observer': args.include_observer,
        'labels': args.labels,
        'show_dataset_stats': args.show_dataset_stats,
        'text_prefix': args.text_prefix
    }
    if dataset_params['process_data'] == 'multilabel_only':
        trainset = MultilabelOnlyDataset(train_df, **dataset_params)
        valset = MultilabelOnlyDataset(val_df, **dataset_params)
        testset = MultilabelOnlyDataset(test_df, **dataset_params)
    elif dataset_params['process_data'] == 'multilabel_with_spans':
        trainset = MultilabelwithSpansDataset(train_df, **dataset_params)
        valset = MultilabelwithSpansDataset(val_df, **dataset_params)
        testset = MultilabelwithSpansDataset(test_df, **dataset_params)
    data = {
        'trainset': trainset,
        'valset': valset,
        'testset': testset
    }
    loader_params = {
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
        'num_workers': 48,
        'persistent_workers': True
    }
    model_params = {
        'model_name': args.model_name,
        'optimizer': args.optimizer,
        'lr': args.lr,
        'momentum': args.momentum,
        'include_target': args.include_target,
        'include_observer': args.include_observer,
        'process_data': args.process_data,
        'metrics': args.metrics,
        'pred_num_beams': args.pred_num_beams,
        'pred_max_length': args.pred_max_length,
        'pred_repetition_penalty': args.pred_repetition_penalty,
        'pred_length_penalty': args.pred_length_penalty,
        'device': args.device
    }
    output_params = {
        'train_to_span': args.train_to_span,
        'val_to_span': args.val_to_span,
        'test_to_span':args.test_to_span,
        'output_spans_path': f'{root_save_path}/{args.output_spans_path}',
        'logfile_path': logfile_path
    }
    all_args = {
        'dataset_params': dataset_params,
        'loader_params': loader_params,
        'model_params': model_params,
        'output_params': output_params
    }

    # saving the model configurations
    with open(f'{root_save_path}/config.json', 'w') as model_config_file:
        json.dump(args.__dict__, model_config_file, indent=4)

    tqdmBar = TQDMProgressBar()
    train_params = dict(
        accelerator='gpu' if args.device == 'cuda' else 'cpu',
        # devices=[int(i) for i in args.cuda],
        gpus=1,
        # devices=3,
        auto_select_gpus=True,
        max_epochs=args.num_epoch,
        default_root_dir=f'{root_save_path}/{args.checkpoint_save_path}',
        num_sanity_val_steps=args.num_sanity_val_steps,
        deterministic=True,
        # strategy="deepspeed_stage_3", # if enable this, disable plugins
        precision="bf16",
        callbacks=[tqdmBar],
        plugins=DDPPlugin(find_unused_parameters=False)
    )
    trainer = pl.Trainer(**train_params)
    model = FLAN_T5(data, **all_args)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    
    end_time = datetime.now()
    logging.info(f'total running time: {end_time - start_time}')

if __name__ == '__main__':
    _main()
