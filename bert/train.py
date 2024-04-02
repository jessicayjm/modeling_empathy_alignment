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

from transformers import BertTokenizerFast, RobertaTokenizerFast, AutoTokenizer
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

from dataset import BertSentDataset
from BERT import *

import sys
sys.path.append('..')
from variables import labels, metrics, loss_funcs
from utils import save_checkpoint, prepare_saving
from utils import check_early_stopping, map_tokens2spans, merge_spans, combine_sents
from utils import simple_cross_entropy_loss, penalize_label_loss, penalize_length_loss, bce_with_logits_loss
from utils import predict, get_metrics


def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='BERT models training')

    # model specifications
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--early_stopping_cutoff', type=int, default=10)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--loss', type=str, choices=loss_funcs)
    parser.add_argument('--optimizer', type=str, choices=['AdamW', 'SGD'], default='AdamW')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--penalize_labels', nargs='+', type=str, default=None)
    parser.add_argument('--lmbda', nargs='+', type=float, default=0., help='hyperparameter for penalizing labels or span length')
    parser.add_argument('--tokenizer', type=str, default='bert-base-cased')
    parser.add_argument('--process_data', type=str, choices=['feed_token', 'feed_single_sent', 'feed_multi_sent'], default='feed_token')
    parser.add_argument('--pred_threshold', type=float, default=0.5, help='prediction threshold for multilabel classification')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--labels', nargs='+', choices=labels, default=labels)
    parser.add_argument('--metrics', nargs='+', choices=metrics, default=metrics)
    parser.add_argument('--random_seed', type=int, default=None)

    # checkpoints
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='need full path from project root directory') 

    # output format specification
    parser.add_argument('--train_to_span', action='store_true', default=False)
    parser.add_argument('--dev_to_span', action='store_true', default=False)
    parser.add_argument('--test_to_span', action='store_true', default=False)
    parser.add_argument('--output_spans_path', type=str, default='output_spans')

    # loggings
    parser.add_argument('--saving_folder', type=str, default='./runs')

    # cuda
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device=='cuda':
        torch.cuda.set_device(args.cuda)

    now, root_save_path = prepare_saving(args.saving_folder, args.model_name.replace("/", "_"), args.checkpoint_save_path, args.output_spans_path)
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


def train(train_loader, model, loss_func, optimizer, id2labels, span_output=None, **kwargs):
    model.train()
    
    train_loss = 0
    all_predictions = None
    all_labels = None
    metrics = {}

    full_train_texts = []
    full_train_mappings = None
    full_train_predictions = None

    for train_texts, train_input_ids, train_attention_mask, train_mapping, train_labels in tqdm(train_loader):
        # transfer to device
        train_input_ids = train_input_ids.to(kwargs['device'])
        train_attention_mask = train_attention_mask.to(kwargs['device'])
        train_mapping = train_mapping.to(kwargs['device'])
        train_labels = train_labels.to(kwargs['device'])
        
        optimizer.zero_grad()
        outputs = model(train_input_ids, att_mask=train_attention_mask)

        loss = loss_func(outputs, train_labels, **kwargs)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        predictions = predict(outputs, **kwargs)

        if kwargs['process_data'] != 'feed_multi_sent':
            all_predictions = predictions.reshape(-1) if all_predictions == None else torch.cat((all_predictions, predictions.reshape(-1)), 0)
            all_labels = train_labels.reshape(-1) if all_labels == None else torch.cat((all_labels, train_labels.reshape(-1)), 0)
        else:
            all_predictions = predictions if all_predictions == None else torch.cat((all_predictions, predictions), 0)
            all_labels = train_labels if all_labels == None else torch.cat((all_labels, train_labels), 0)

        if span_output != None:
            if kwargs['process_data'] == 'feed_token':
                spans = map_tokens2spans(predictions, train_mapping)
                span_output += merge_spans(train_texts, spans, id2labels)
            else:
                full_train_texts += train_texts
                full_train_mappings = train_mapping if full_train_mappings == None else torch.cat((full_train_mappings, train_mapping), 0)
                full_train_predictions = predictions if full_train_predictions == None else torch.cat((full_train_predictions, predictions), 0)
    
    
    if span_output != None and (kwargs['process_data'] == 'feed_single_sent' or kwargs['process_data'] == 'feed_multi_sent'):
        span_output += combine_sents(full_train_texts, full_train_mappings.to('cpu'), full_train_predictions.to('cpu'), id2labels, **kwargs)
    metrics = get_metrics(all_labels, all_predictions, id2labels, **kwargs)
    logging.info(f"train loss: {round(train_loss, 5)} \t train metrics: {metrics}")


def evaluate(test_loader, model, loss_func, id2labels, dataset_name, span_output=None, **kwargs):
    model.eval()

    test_loss = 0
    all_predictions = None
    all_labels = None
    metrics = {}

    full_test_texts = []
    full_test_mappings = None
    full_test_predictions = None

    for test_texts, test_input_ids, test_attention_mask, test_mapping, test_labels in tqdm(test_loader):
        # transfer to device
        test_input_ids = test_input_ids.to(kwargs['device'])
        test_attention_mask = test_attention_mask.to(kwargs['device'])
        test_mapping = test_mapping.to(kwargs['device'])
        test_labels = test_labels.to(kwargs['device'])
        
        outputs = model(test_input_ids, att_mask=test_attention_mask)

        loss = loss_func(outputs, test_labels, **kwargs)
        test_loss += loss.item()
        predictions = predict(outputs, **kwargs)
        
        if kwargs['process_data'] != 'feed_multi_sent':
            all_predictions = predictions.reshape(-1) if all_predictions == None else torch.cat((all_predictions, predictions.reshape(-1)), 0)
            all_labels = test_labels.reshape(-1) if all_labels == None else torch.cat((all_labels, test_labels.reshape(-1)), 0)
        else:
            all_predictions = predictions if all_predictions == None else torch.cat((all_predictions, predictions), 0)
            all_labels = test_labels if all_labels == None else torch.cat((all_labels, test_labels), 0)

        if span_output != None:
            if kwargs['process_data'] == 'feed_token':
                spans = map_tokens2spans(predictions, test_mapping)
                span_output += merge_spans(test_texts, spans, id2labels)
            else:
                full_test_texts += test_texts
                full_test_mappings = test_mapping if full_test_mappings == None else torch.cat((full_test_mappings, test_mapping), 0)
                full_test_predictions = predictions if full_test_predictions == None else torch.cat((full_test_predictions, predictions), 0)
    
    if span_output != None and (kwargs['process_data'] == 'feed_single_sent' or kwargs['process_data'] == 'feed_multi_sent'):
        span_output += combine_sents(full_test_texts, full_test_mappings.to('cpu'), full_test_predictions.to('cpu'), id2labels, **kwargs)
    metrics = get_metrics(all_labels, all_predictions, id2labels, **kwargs)
    logging.info(f"{dataset_name} loss: {round(test_loss, 5)} \t {dataset_name} metrics: {metrics}")  

    return test_loss

def _main():
    start_time = datetime.now()
    args, root_save_path = get_args()
    
    # load tokenizer
    if 'roberta' in args.model_name.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer)
    elif 'deberta' in args.model_name.lower():
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.tokenizer)
    elif 'minilm' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    
    dataset_params = {
        'tokenizer': tokenizer,
        'process_data': args.process_data,
        'multilabel': args.process_data == 'feed_multi_sent'
    }
    train_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
    }
    dev_test_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
    }

    # saving the model configurations
    with open(f'{root_save_path}/config.json', 'w') as model_config_file:
        json.dump(args.__dict__, model_config_file, indent=4)
    
    trainset = BertSentDataset(args.train_data, **dataset_params)
    train_loader = torch.utils.data.DataLoader(trainset, **train_loader_params)
    devset = BertSentDataset(args.dev_data, **dataset_params)
    dev_loader = torch.utils.data.DataLoader(devset, **dev_test_loader_params)
    testset = BertSentDataset(args.test_data, **dataset_params)
    test_loader = torch.utils.data.DataLoader(testset, **dev_test_loader_params)
    

    if args.process_data == 'feed_token':
        if 'roberta' in args.model_name.lower():
            model = RoBERTaTokenClassification(args.model_name, len(trainset.get_id2labels())).to(args.device)
        elif 'deberta' in args.model_name.lower():
            assert("feed token not supported for deberta")
        elif 'minilm' in args.model_name.lower():
            assert("feed token not supported for miniLM")
        else:
            model = BERTTokenClassification(args.model_name, len(trainset.get_id2labels())).to(args.device)
    elif args.process_data == 'feed_single_sent':
        if 'roberta' in args.model_name.lower():
            model = RoBERTaSentenceClassification(args.model_name, len(trainset.get_id2labels())).to(args.device)
        elif 'deberta' in args.model_name.lower():
            model = DeBERTaSentenceClassification(args.model_name, len(trainset.get_id2labels())).to(args.device)
        elif 'minilm' in args.model_name.lower():
            model = MiniLMSentenceClassification(args.model_name, len(trainset.get_id2labels())).to(args.device)
        else:
            model = BERTSentenceClassification(args.model_name, len(trainset.get_id2labels())).to(args.device)
    else:
        if 'roberta' in args.model_name.lower():
            model = RoBERTaMultilabelClassication(args.model_name, len(trainset.get_id2labels())).to(args.device)
        elif 'deberta' in args.model_name.lower():
            assert("multi-label-sent not supported for deberta")
        elif 'minilm' in args.model_name.lower():
            assert("feed token not supported for miniLM")
        else:
            model = BERTMultilabelClassication(args.model_name, len(trainset.get_id2labels())).to(args.device)

    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
    else:
        raise(f'No such optimizer: {args.optimizer}')

    if args.process_data != 'feed_multi_sent' and args.loss == 'simple_cross_entropy_loss':
        loss = simple_cross_entropy_loss
    elif args.process_data != 'feed_multi_sent' and args.loss == 'penalize_label_loss':
        loss = penalize_label_loss
    elif args.process_data != 'feed_multi_sent' and args.loss == 'penalize_length_loss':
        loss = penalize_length_loss
    elif args.process_data == 'feed_multi_sent' and args.loss == 'bce_with_logits_loss':
        loss = bce_with_logits_loss
    else:
        raise(f'Wrong loss specification. Check loss name or multilabel setting.')

    labels2id = trainset.get_labels2id()
    penalize_labels = [labels2id[i] for i in args.penalize_labels] if args.penalize_labels != None else []
    model_params = {
        'penalize_labels': penalize_labels,
        'lmbda': args.lmbda,
        'metrics': args.metrics,
        'process_data': args.process_data, # used for sentence-based text reconstruction
        'pred_threshold': args.pred_threshold, # used for multilabel prediction
        'device': args.device
    }

    start_epoch = 0
    cur_min_dev_loss = math.inf
    cur_min_dev_loss_epoch = -1
    if args.load_checkpoint:
        logging.info(f'loading checkpoint from {args.load_checkpoint}')
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        cur_min_dev_loss_epoch = start_epoch

    for epoch in range(start_epoch, args.num_epoch):
        if check_early_stopping(cur_min_dev_loss_epoch, epoch, args.early_stopping_cutoff):
            logging.info(f"early stop at epoch {epoch-1}")
            break
        logging.info(f'start epoch {epoch}')

        train_span_output = [] if args.train_to_span else None
        dev_span_output = [] if args.dev_to_span else None
        test_span_output = [] if args.test_to_span else None

        train(train_loader, model, loss, optimizer, trainset.get_id2labels(), train_span_output, **model_params)
        dev_loss = evaluate(dev_loader, model, loss, devset.get_id2labels(), 'dev', dev_span_output, **model_params)
        if dev_loss <= cur_min_dev_loss:
            cur_min_dev_loss = dev_loss
            cur_min_dev_loss_epoch = epoch
        _ = evaluate(test_loader, model, loss, testset.get_id2labels(), 'test', test_span_output, **model_params)

        # save output train and test spans
        if args.train_to_span:
            with open(f'{root_save_path}/{args.output_spans_path}/epoch{str("{:03d}".format(epoch))}_train.json', 'w') as train_span_out:
                json.dump(train_span_output, train_span_out)
        if args.dev_to_span:
            with open(f'{root_save_path}/{args.output_spans_path}/epoch{str("{:03d}".format(epoch))}_dev.json', 'w') as dev_span_out:
                json.dump(dev_span_output, dev_span_out)
        if args.test_to_span:
            with open(f'{root_save_path}/{args.output_spans_path}/epoch{str("{:03d}".format(epoch))}_test.json', 'w') as test_span_out:
                json.dump(test_span_output, test_span_out)
        
        save_checkpoint(epoch, \
                        model.state_dict(), \
                        optimizer.state_dict(), \
                        f'{root_save_path}/{args.checkpoint_save_path}')
    
    logging.info(f"lowest dev loss at epoch {cur_min_dev_loss_epoch}")
    end_time = datetime.now()
    logging.info(f'total running time: {end_time - start_time}')

if __name__ == '__main__':
    _main()