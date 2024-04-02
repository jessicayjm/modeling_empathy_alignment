import os
import re
import math
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
import torch.optim as optim

from transformers import AutoTokenizer

from dataset import AlignmentDataset, AlignmentDatasetFullTarget
from model import SNN

import sys
sys.path.append('..')
from variables import labels, metrics, loss_funcs
from utils import save_checkpoint, prepare_saving
from utils import check_early_stopping, map_tokens2spans, merge_spans, combine_sents
from utils import contrastive_loss
from utils import predict, get_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='Siamese Network for alignment detection')

    parser.add_argument('--test_data', type=str)
    parser.add_argument('--tokenizer', type=str, default='bert-base-cased')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--target_num_neighbors', type=int, default=0)
    parser.add_argument('--target_context_pos', choices=['preceding', 'succeeding', 'surround', None], type=str, default=None)
    parser.add_argument('--observer_num_neighbors', type=int, default=0)
    parser.add_argument('--observer_context_pos', choices=['preceding', 'succeeding', 'surround', None], type=str, default=None)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--labels', nargs='+', choices=labels, default=labels)
    parser.add_argument('--metrics', nargs='+', choices=metrics, default=metrics)
    parser.add_argument('--random_seed', type=int, default=None)

    parser.add_argument('--saving_folder', type=str, default='./runs')
    parser.add_argument('--show_dataset_stats', action='store_true', default=True)
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpoints')
    parser.add_argument('--output_alignments_path', type=str, default='output_alignments')

    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device=='cuda':
        torch.cuda.set_device(args.cuda)

    now, root_save_path = prepare_saving(args.saving_folder, "baseline", args.checkpoint_save_path, args.output_alignments_path)
    logfile_path = f'{root_save_path}/output.log'
    logging.basicConfig(filename=logfile_path, level=logging.INFO)

    if args.random_seed != None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available() and args.device=='cuda':
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.benchmark = False
    
    return args, root_save_path

def evaluate(test_loader, **kwargs):
    
    all_predictions = None
    all_labels = None
    metrics = {}

    
    for data in tqdm(test_loader):
        _,_,_,_,test_labels = data
      
        test_labels = test_labels.to(kwargs['device'])

        all_labels = test_labels if all_labels == None else torch.cat((all_labels, test_labels), 0)

    # all_predictions = torch.ones_like(all_labels)  
    # all_predictions = torch.randint(2, size=all_labels.shape)

    metrics = get_metrics(all_labels, all_predictions, id2labels=None, metrics=kwargs["metrics"], process_data=None)
    logging.info(f"baseline test metrics: {metrics}")  

def _main():
    start_time = datetime.now()

    args, root_save_path = get_args()

    # saving the model configurations
    with open(f'{root_save_path}/config.json', 'w') as model_config_file:
        json.dump(args.__dict__, model_config_file, indent=4)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    dataset_params = {
        'tokenizer': tokenizer,
        'label_names': args.labels,
        'target_num_neighbors': args.target_num_neighbors,
        'target_context_pos': args.target_context_pos,
        'observer_num_neighbors': args.observer_num_neighbors,
        'observer_context_pos': args.observer_context_pos,
        'load_processed_dataset': False,
        'save_path': None
    }
    test_dataset_params = dataset_params
    test_dataset_params['to_alignment'] = False
    dev_test_loader_params = {
        'batch_size': args.batch_size,
    }

    # load in datafile
    test_df = pd.read_json(args.test_data)
    testset = AlignmentDatasetFullTarget(test_df, **test_dataset_params)
    test_loader = torch.utils.data.DataLoader(testset, **dev_test_loader_params)

    if args.show_dataset_stats:
        logging.info('TEST DATA')
        testset.get_stats()

    model_params = {
        'metrics': args.metrics,
        'device': args.device
    }
    
    evaluate(test_loader, **model_params)

if __name__ == '__main__':
    _main()