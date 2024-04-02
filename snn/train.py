import os
import re
import copy
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
from utils import check_early_stopping
from utils import contrastive_loss, mse_loss, simple_cross_entropy_loss, binary_cross_entropy_loss
from utils import predict, get_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='Siamese Network for alignment detection')

    # model specifications
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--dev_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--load_processed_dataset', type=str, default=None)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--early_stopping_cutoff', type=int, default=10)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--loss', type=str, choices=loss_funcs, default="contrastive_loss")
    parser.add_argument('--pos_weight', type=float, default=None)
    parser.add_argument('--m', type=float, help="hyparameter to control contrastive loss")
    parser.add_argument('--lmbda', type=float, help="hyparameter to control the contribution of contrastive loss")
    parser.add_argument('--optimizer', type=str, choices=['AdamW', 'SGD'], default='AdamW')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--tokenizer', type=str, default='bert-base-cased')
    parser.add_argument('--target_num_neighbors', type=int, default=0)
    parser.add_argument('--target_context_pos', choices=['preceding', 'succeeding', 'surround', None], type=str, default=None)
    parser.add_argument('--observer_num_neighbors', type=int, default=0)
    parser.add_argument('--observer_context_pos', choices=['preceding', 'succeeding', 'surround', None], type=str, default=None)
    parser.add_argument('--use_full_target', action="store_true", default=False)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--labels', nargs='+', choices=labels, default=labels)
    parser.add_argument('--metrics', nargs='+', choices=metrics, default=metrics)
    parser.add_argument('--pred_threshold', type=float, default=0.5)
    parser.add_argument('--full_target_model_output', type=str, default=None, help="the path of full_target_output with observer span start indices")
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
    parser.add_argument('--dataset_save_path', type=str, default=None)

    # cuda
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--use_deepspeed', action='store_true', default=False)

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device=='cuda' and args.cuda!=-1:
        torch.cuda.set_device(args.cuda)

    # now, root_save_path = prepare_saving(args.saving_folder, args.model_name.replace("/", "_"), args.checkpoint_save_path, args.output_alignments_path)
    # logfile_path = f'{root_save_path}/output.log'
    # logging.basicConfig(filename=logfile_path, level=logging.INFO)

    if args.random_seed != None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available() and args.device=='cuda':
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.benchmark = False
    
    # return args, root_save_path
    return args, None


def train(train_loader, model, loss_func, optimizer, alignments_output=None, **kwargs):
    model.train()
    
    train_loss = 0
    all_predictions = None
    all_labels = None
    metrics = {}

    if kwargs['train_to_alignment']:
        full_train_alignments = None
        full_train_mappings = None

    for data in tqdm(train_loader):
        if kwargs['train_to_alignment']:
            train_target_input_ids, \
            train_target_attention_mask, \
            train_observer_input_ids, \
            train_observer_attention_mask, \
            train_labels, \
            train_ori_alignments, \
            train_text_mappings = data

            full_train_alignments = train_ori_alignments if full_train_alignments == None else torch.cat((full_train_alignments, train_ori_alignments), 0)
            full_train_mappings = train_text_mappings if full_train_mappings == None else torch.cat((full_train_mappings, train_text_mappings), 0)
        else:
            train_target_input_ids, \
            train_target_attention_mask, \
            train_observer_input_ids, \
            train_observer_attention_mask, \
            train_labels = data
        # transfer to device
        train_target_input_ids = train_target_input_ids.to(kwargs['device'])
        train_target_attention_mask = train_target_attention_mask.to(kwargs['device'])
        train_observer_input_ids = train_observer_input_ids.to(kwargs['device'])
        train_observer_attention_mask = train_observer_attention_mask.to(kwargs['device'])
        train_labels = train_labels.to(kwargs['device'])

        all_labels = train_labels if all_labels == None else torch.cat((all_labels, train_labels), 0)
        
        optimizer.zero_grad()
        outputs = model(train_target_input_ids, \
                        train_target_attention_mask, \
                        train_observer_input_ids, \
                        train_observer_attention_mask)

        loss = loss_func(outputs, train_labels, **kwargs)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        predictions = predict(outputs, process_data="feed_alignments", pred_threshold=kwargs["pred_threshold"])
        all_predictions = predictions if all_predictions == None else torch.cat((all_predictions, predictions), 0)

    if kwargs['train_to_alignment']:
        pred_alignments_index = all_predictions.nonzero().flatten()
        full_train_alignments = full_train_alignments.tolist()
        full_train_mappings = full_train_mappings.tolist()
        for i in pred_alignments_index:
            if isinstance(full_train_mappings[i], tuple):
                text_id = full_train_mappings[i]
            else:
                text_id = int(full_train_mappings[i])
            if text_id not in alignments_output.keys():
                alignments_output[text_id] = [full_train_alignments[i]]
            else:
                alignments_output[text_id].append(full_train_alignments[i])

    metrics = get_metrics(all_labels, all_predictions, id2labels=None, metrics=kwargs["metrics"], process_data=None)
    # logging.info(f"train loss: {round(train_loss, 5)} \t train metrics: {metrics}")

# @resource_usage_monitor
def evaluate(test_loader, model, loss_func, dataset_name, alignments_output=None, **kwargs):
    model.eval()

    test_loss = 0
    all_predictions = None
    all_labels = None
    metrics = {}

    if kwargs[dataset_name+'_to_alignment']:
        full_test_alignments = None
        full_test_mappings = None
    
    for data in tqdm(test_loader):
        if kwargs[dataset_name+'_to_alignment']:
            test_target_input_ids, \
            test_target_attention_mask, \
            test_observer_input_ids, \
            test_observer_attention_mask, \
            test_labels, \
            test_ori_alignments, \
            test_text_mappings = data

            full_test_alignments = test_ori_alignments if full_test_alignments == None else torch.cat((full_test_alignments, test_ori_alignments), 0)
            full_test_mappings = test_text_mappings if full_test_mappings == None else torch.cat((full_test_mappings, test_text_mappings), 0)
        else:
            test_target_input_ids, \
            test_target_attention_mask, \
            test_observer_input_ids, \
            test_observer_attention_mask, \
            test_labels = data
        
        # transfer to device
        test_target_input_ids = test_target_input_ids.to(kwargs['device'])
        test_target_attention_mask = test_target_attention_mask.to(kwargs['device'])
        test_observer_input_ids = test_observer_input_ids.to(kwargs['device'])
        test_observer_attention_mask = test_observer_attention_mask.to(kwargs['device'])
        test_labels = test_labels.to(kwargs['device'])

        all_labels = test_labels if all_labels == None else torch.cat((all_labels, test_labels), 0)

        outputs = model(test_target_input_ids, \
                        test_target_attention_mask, \
                        test_observer_input_ids, \
                        test_observer_attention_mask)
        loss = loss_func(outputs, test_labels, **kwargs)

        test_loss += loss.item()
        predictions = predict(outputs, process_data="feed_alignments", pred_threshold=kwargs["pred_threshold"])
        all_predictions = predictions if all_predictions == None else torch.cat((all_predictions, predictions), 0)

    if kwargs[dataset_name+'_to_alignment']:
        pred_alignments_index = all_predictions.nonzero().flatten()
        full_test_alignments = full_test_alignments.tolist()
        full_test_mappings = full_test_mappings.tolist()
        for i in pred_alignments_index:
            if isinstance(full_test_mappings[i], list):
                text_id = [str(ti) for ti in full_test_mappings[i]]
                text_id = " ".join(text_id)
            else:
                text_id = str(int(full_test_mappings[i]))
            # filter the model output given full target model output
            if kwargs['full_target_model_output']:
                if full_test_alignments[i][2] not in kwargs['full_target_model_output'][text_id]:
                    # set prediction to 0
                    all_predictions[i] = 0
                    continue
            if text_id not in alignments_output.keys():
                alignments_output[text_id] = [full_test_alignments[i]]
            else:
                alignments_output[text_id].append(full_test_alignments[i])

    metrics = get_metrics(all_labels, all_predictions, id2labels=None, metrics=kwargs["metrics"], process_data=None)
    # logging.info(f"{dataset_name} loss: {round(test_loss, 5)} \t {dataset_name} metrics: {metrics}")  

    return test_loss

# @resource_usage_monitor
def _main():
    start_time = datetime.now()

    args, root_save_path = get_args()

    if args.full_target_model_output and not args.test_to_alignment:
        assert("must specify test_to_alignment to use full_target_model_output")

    # saving the model configurations
    # with open(f'{root_save_path}/config.json', 'w') as model_config_file:
    #     json.dump(args.__dict__, model_config_file, indent=4)


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) if not args.load_processed_dataset else None
    
    dataset_params = {
        'tokenizer': tokenizer,
        'label_names': args.labels,
        'target_num_neighbors': args.target_num_neighbors,
        'target_context_pos': args.target_context_pos,
        'observer_num_neighbors': args.observer_num_neighbors,
        'observer_context_pos': args.observer_context_pos
    }
    if args.evaluate_only:
        test_dataset_params = copy.deepcopy(dataset_params)
        test_dataset_params['to_alignment'] = args.test_to_alignment
        
        test_dataset_params['save_path'] = f'{args.dataset_save_path}' if args.dataset_save_path else None
        test_dataset_params['load_processed_dataset'] = f'{args.load_processed_dataset}' if args.load_processed_dataset else None

        dev_test_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
        }

        test_df = pd.read_json(args.test_data) if not args.load_processed_dataset else None

        if args.use_full_target:
            testset = AlignmentDatasetFullTarget(test_df, **test_dataset_params)
        else:
            testset = AlignmentDataset(test_df, **test_dataset_params)
        
        test_loader = torch.utils.data.DataLoader(testset, **dev_test_loader_params)

        if args.show_dataset_stats and not args.load_processed_dataset:
            # logging.info('TEST DATA')
            testset.get_stats()
        
        if args.dataset_save_path:
            testset.save_dataset()
            return

    else:
        train_dataset_params = copy.deepcopy(dataset_params)
        train_dataset_params['to_alignment'] = args.train_to_alignment
        dev_dataset_params = copy.deepcopy(dataset_params)
        dev_dataset_params['to_alignment'] = args.dev_to_alignment
        test_dataset_params = copy.deepcopy(dataset_params)
        test_dataset_params['to_alignment'] = args.test_to_alignment

        train_dataset_params['save_path'] = f'{args.dataset_save_path}/train.pt' if args.dataset_save_path else None
        dev_dataset_params['save_path'] = f'{args.dataset_save_path}/dev.pt'  if args.dataset_save_path else None
        test_dataset_params['save_path'] = f'{args.dataset_save_path}/test.pt'  if args.dataset_save_path else None

        train_dataset_params['load_processed_dataset'] = f'{args.load_processed_dataset}/train.pt' if args.load_processed_dataset else None
        dev_dataset_params['load_processed_dataset'] = f'{args.load_processed_dataset}/dev.pt' if args.load_processed_dataset else None
        test_dataset_params['load_processed_dataset'] = f'{args.load_processed_dataset}/test.pt' if args.load_processed_dataset else None

        train_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': args.shuffle,
        }
        dev_test_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
        }

        # load in datafile
        train_df = pd.read_json(args.train_data) if not args.load_processed_dataset else None
        dev_df = pd.read_json(args.dev_data) if not args.load_processed_dataset else None
        test_df = pd.read_json(args.test_data) if not args.load_processed_dataset else None
        
        if args.use_full_target:
            trainset = AlignmentDatasetFullTarget(train_df, **train_dataset_params)
        else:
            trainset = AlignmentDataset(train_df, **train_dataset_params)
        train_loader = torch.utils.data.DataLoader(trainset, **train_loader_params)
        if args.use_full_target:
            devset = AlignmentDatasetFullTarget(dev_df, **dev_dataset_params)
        else:
            devset = AlignmentDataset(dev_df, **dev_dataset_params)
        dev_loader = torch.utils.data.DataLoader(devset, **dev_test_loader_params)
        if args.use_full_target:
            testset = AlignmentDatasetFullTarget(test_df, **test_dataset_params)
        else:
            testset = AlignmentDataset(test_df, **test_dataset_params)
        test_loader = torch.utils.data.DataLoader(testset, **dev_test_loader_params)

        if args.show_dataset_stats and not args.load_processed_dataset:
            # logging.info('TRAIN DATA')
            trainset.get_stats()
            # logging.info('DEV DATA')
            devset.get_stats()
            # logging.info('TEST DATA')
            testset.get_stats()
        
        if args.dataset_save_path:
            trainset.save_dataset()
            devset.save_dataset()
            testset.save_dataset()
            return
    
    model = SNN(args.model_name).to(args.device)

    start_epoch = 0
    cur_min_dev_loss = math.inf
    cur_min_dev_loss_epoch = -1


    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
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

    if args.full_target_model_output:
        with open(args.full_target_model_output, 'r') as f:
            full_target_model_output = json.load(f)
    else:
        full_target_model_output = None
    model_params = {
        'm': args.m,
        'lmbda': args.lmbda,
        'pos_weight': torch.Tensor([args.pos_weight]).to(args.device),
        'metrics': args.metrics,
        'pred_threshold': args.pred_threshold,
        'train_to_alignment': args.train_to_alignment,
        'dev_to_alignment': args.dev_to_alignment,
        'test_to_alignment': args.test_to_alignment,
        'full_target_model_output': full_target_model_output,
        'device': args.device
    }

    if args.load_checkpoint:
        # logging.info(f'loading checkpoint from {args.load_checkpoint}')
        checkpoint = torch.load(args.load_checkpoint,map_location=f'cuda:{args.cuda}')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if args.use_deepspeed:
            import deepspeed
            deepspeed_config = {
                "kernel_inject": False,
                "dtype": "bf16",
                "enable_cuda_graph": False,
            }
            engine = deepspeed.init_inference(model=model, config=deepspeed_config)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        cur_min_dev_loss_epoch = start_epoch
        if args.evaluate_only:
            epoch = checkpoint['epoch']
            test_alignments_output = {} if args.test_to_alignment else None
            if args.use_deepspeed:
                _ = evaluate(test_loader, engine, loss, 'test', test_alignments_output, **model_params)
            else:
                _ = evaluate(test_loader, model, loss, 'test', test_alignments_output, **model_params)
            if args.test_to_alignment:
                with open(args.output_alignments_path, 'w') as test_alignment_out:
                    json.dump(test_alignments_output, test_alignment_out, indent=4)
            return
    
    # starting training
    for epoch in range(start_epoch, args.num_epoch):
        if check_early_stopping(cur_min_dev_loss_epoch, epoch, args.early_stopping_cutoff):
            # logging.info(f"early stop at epoch {epoch-1}")
            break
        # logging.info(f'start epoch {epoch}')

        train_alignments_output = {} if args.train_to_alignment else None
        dev_alignments_output = {} if args.dev_to_alignment else None
        test_alignments_output = {} if args.test_to_alignment else None

        train(train_loader, model, loss, optimizer, train_alignments_output, **model_params)
        dev_loss = evaluate(dev_loader, model, loss, 'dev', dev_alignments_output, **model_params)
        if dev_loss <= cur_min_dev_loss:
            cur_min_dev_loss = dev_loss
            cur_min_dev_loss_epoch = epoch
        _ = evaluate(test_loader, model, loss, 'test', test_alignments_output, **model_params)

        # save output train and test spans
        if args.train_to_alignment:
            with open(f'{root_save_path}/{args.output_alignments_path}/epoch{str("{:03d}".format(epoch))}_train.json', 'w') as train_alignment_out:
                json.dump(train_alignments_output, train_alignment_out)
        if args.dev_to_alignment:
            with open(f'{root_save_path}/{args.output_alignments_path}/epoch{str("{:03d}".format(epoch))}_dev.json', 'w') as dev_alignment_out:
                json.dump(dev_alignments_output, dev_alignment_out)
        if args.test_to_alignment:
            with open(f'{root_save_path}/{args.output_alignments_path}/epoch{str("{:03d}".format(epoch))}_test.json', 'w') as test_alignment_out:
                json.dump(test_alignments_output, test_alignment_out)
        
        save_checkpoint(epoch, \
                        model.state_dict(), \
                        optimizer.state_dict(), \
                        f'{root_save_path}/{args.checkpoint_save_path}')
    
    # logging.info(f"lowest dev loss at epoch {cur_min_dev_loss_epoch}")
    end_time = datetime.now()
    # logging.info(f'total running time: {end_time - start_time}')

if __name__ == '__main__':
    _main()
    