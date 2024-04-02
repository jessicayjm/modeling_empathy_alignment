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

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator

from dataset import AlignmentDataset

import sys
sys.path.append('..')
from variables import labels, metrics, loss_funcs
from utils import save_checkpoint, prepare_saving
from utils import check_early_stopping, map_tokens2spans, merge_spans, combine_sents
from utils import contrastive_loss, mse_loss, simple_cross_entropy_loss, binary_cross_entropy_loss
from utils import predict, get_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='Siamese Network for alignment detection')

    # model specifications
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--loss', type=str, choices=loss_funcs, default="contrastive_loss")
    parser.add_argument('--pos_weight', type=float, default=None)
    parser.add_argument('--m', type=float, help="hyparameter to control contrastive loss")
    parser.add_argument('--lmbda', type=float, help="hyparameter to control the contribution of contrastive loss")
    parser.add_argument('--optimizer', type=str, choices=['AdamW', 'SGD'], default='AdamW')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--target_num_neighbors', type=int, default=0)
    parser.add_argument('--target_context_pos', choices=['preceding', 'succeeding', 'surround', None], type=str, default=None)
    parser.add_argument('--observer_num_neighbors', type=int, default=0)
    parser.add_argument('--observer_context_pos', choices=['preceding', 'succeeding', 'surround', None], type=str, default=None)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--labels', nargs='+', choices=labels, default=labels)
    parser.add_argument('--metrics', nargs='+', choices=metrics, default=metrics)
    parser.add_argument('--pred_threshold', type=float, default=0.5)
    parser.add_argument('--random_seed', type=int, default=None)

    # checkpoints
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='need full path from project root directory') 

    # output format specification
    parser.add_argument('--train_to_alignment', action='store_true', default=False)
    parser.add_argument('--dev_to_alignment', action='store_true', default=False)
    parser.add_argument('--test_to_alignment', action='store_true', default=False)
    parser.add_argument('--evaluate_only', action='store_true', default=False)
    parser.add_argument('--output_alignments_path', type=str, default='output_alignments')

    # loggings
    parser.add_argument('--show_dataset_stats', action='store_true', default=True)
    parser.add_argument('--saving_folder', type=str, default='./runs')

    # cuda
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device=='cuda':
        torch.cuda.set_device(args.cuda)

    now, root_save_path = prepare_saving(args.saving_folder, args.model_name.replace("/", "_"), args.checkpoint_save_path, args.output_alignments_path)
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


def _main():
    start_time = datetime.now()

    args, root_save_path = get_args()

    # saving the model configurations
    with open(f'{root_save_path}/config.json', 'w') as model_config_file:
        json.dump(args.__dict__, model_config_file, indent=4)

    dataset_params = {
        'label_names': args.labels,
        'target_num_neighbors': args.target_num_neighbors,
        'target_context_pos': args.target_context_pos,
        'observer_num_neighbors': args.observer_num_neighbors,
        'observer_context_pos': args.observer_context_pos
    }
    train_dataset_params = dataset_params
    train_dataset_params['to_alignment'] = args.train_to_alignment
    dev_dataset_params = dataset_params
    dev_dataset_params['to_alignment'] = args.dev_to_alignment
    test_dataset_params = dataset_params
    test_dataset_params['to_alignment'] = args.test_to_alignment
    
    train_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
    }
    dev_test_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
    }

    # load in datafile
    train_df = pd.read_json(args.train_data)
    dev_df = pd.read_json(args.dev_data)
    test_df = pd.read_json(args.test_data)

    trainset = AlignmentDataset(train_df, **train_dataset_params)
    train_examples = trainset.get_examples()
    train_loader = torch.utils.data.DataLoader(train_examples, **train_loader_params)
    devset = AlignmentDataset(dev_df, **dev_dataset_params)
    dev_examples = devset.get_examples()
    dev_loader = torch.utils.data.DataLoader(dev_examples, **dev_test_loader_params)
    testset = AlignmentDataset(test_df, **test_dataset_params)
    test_examples = testset.get_examples()
    test_loader = torch.utils.data.DataLoader(test_examples, **dev_test_loader_params)

    if args.show_dataset_stats:
        logging.info('TRAIN DATA')
        trainset.get_stats()
        logging.info('DEV DATA')
        devset.get_stats()
        logging.info('TEST DATA')
        testset.get_stats()
    
    model = CrossEncoder(args.model_name, num_labels=2)
    evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_examples, name='dev')
    warmup_steps = math.ceil(len(train_loader) * args.num_epoch * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    if args.optimizer == 'AdamW':
        optimizer_class = optim.AdamW
        optimizer_args = {"lr": args.lr}
    elif args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        optimizer_args = {"lr": args.lr, "momentum": args.momentum}
    else:
        raise(f'No such optimizer: {args.optimizer}')

    if args.loss == "contrastive_loss":
        loss = contrastive_loss
    elif args.loss == "mse_loss":
        loss = mse_loss
    elif args.loss == "simple_cross_entropy_loss":
        loss = simple_cross_entropy_loss
    elif args.loss == "binary_cross_entropy_loss":
        loss = binary_cross_entropy_loss
    else: assert("loss not supported")

    
    # starting training
    model.fit(train_dataloader=train_loader,
            evaluator=evaluator,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_args,
            epochs=args.num_epoch,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            output_path=root_save_path,
            show_progress_bar=True)
    end_time = datetime.now()
    logging.info(f'total running time: {end_time - start_time}')

if __name__ == '__main__':
    _main()