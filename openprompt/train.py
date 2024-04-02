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
import deepspeed

from transformers import BertConfig, BertModel, BertForMaskedLM, RobertaConfig, RobertaModel, RobertaForMaskedLM
from transformers import AutoModel, AutoTokenizer, RobertaTokenizer

import sys
sys.path.append('..')
from dataset import OpenpromptSentDataset
from variables import labels, metrics
from utils import predict, get_metrics
from utils import check_early_stopping, combine_sents
from utils import save_checkpoint, prepare_saving, get_template, get_promptdataloader
from utils import simple_cross_entropy_loss, penalize_label_loss, penalize_length_loss

from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.utils.reproduciblity import set_seed
from openprompt.plms.mlm import MLMTokenizerWrapper

def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='OpenPrompt models training')

    # model specifications
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--dev_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--prompt_file', type=str)
    parser.add_argument('--model_family', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--tokenizer', type=str, default="")
    parser.add_argument('--load_trained_model', type=str, default=None)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--early_stopping_cutoff', type=int, default=10)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--process_data', type=str, choices=['feed_single_sent', 'feed_multi_sent'], default='feed_single_sent')
    parser.add_argument('--loss', type=str, choices=['simple_cross_entropy_loss', 'penalize_no_label_loss', 'penalize_length_loss'], default='simple_cross_entropy_loss')
    parser.add_argument('--optimizer', type=str, choices=['AdamW', 'SGD'], default='AdamW')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--penalize_labels', nargs='+', type=str)
    parser.add_argument('--lmbda', nargs='+', type=float, default=0., help='hyperparameter for penalizing labels or span length')
    parser.add_argument('--freeze_plm', action='store_true', default=False, help='freeze the pretrained language model when training prompt model')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--labels', nargs='+', choices=labels, default=labels)
    parser.add_argument('--metrics', nargs='+', choices=metrics, default=metrics)
    parser.add_argument('--random_seed', type=int, default=None)

    # prompt dataloader specifications
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--decoder_max_length', type=int)
    parser.add_argument('--teacher_forcing', action='store_true', default=False)
    parser.add_argument('--predict_eos_token', action='store_true', default=False)
    parser.add_argument('--truncate_method', type=str, default='head')

    # checkpoints
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='need full path from project root directory') 

    # output format specification
    parser.add_argument('--train_to_span', action='store_true', default=False)
    parser.add_argument('--dev_to_span', action='store_true', default=False)
    parser.add_argument('--test_to_span', action='store_true', default=False)
    parser.add_argument('--evaluate_only', action='store_true', default=False)
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
        set_seed(args.random_seed)
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

    for inputs in tqdm(train_loader):
        inputs = inputs.to(kwargs['device'])
        optimizer.zero_grad()
        outputs = model(inputs)

        train_texts = list(inputs['guid'][0])
        train_mapping = torch.transpose(torch.stack(inputs['guid'][1]), 0, 1)
        train_labels = inputs['label']
        loss = loss_func(outputs, train_labels, **kwargs)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        predictions = predict(outputs, **kwargs)
        all_predictions = predictions.reshape(-1) if all_predictions == None else torch.cat((all_predictions, predictions.reshape(-1)), 0)
        all_labels = train_labels.reshape(-1) if all_labels == None else torch.cat((all_labels, train_labels.reshape(-1)), 0)
        if span_output != None:
            full_train_texts += train_texts
            full_train_mappings = train_mapping if full_train_mappings == None else torch.cat((full_train_mappings, train_mapping), 0)
            full_train_predictions = predictions if full_train_predictions == None else torch.cat((full_train_predictions, predictions), 0)
    
    if span_output != None:
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

    for inputs in tqdm(test_loader):
        inputs = inputs.to(kwargs['device'])
        outputs = model(inputs)

        test_texts = list(inputs['guid'][0])
        test_mapping = torch.transpose(torch.stack(inputs['guid'][1]), 0, 1)
        test_labels = inputs['label']
        loss = loss_func(outputs, test_labels)
        test_loss += loss.item()

        predictions = predict(outputs, **kwargs)
        all_predictions = predictions.reshape(-1) if all_predictions == None else torch.cat((all_predictions, predictions.reshape(-1)), 0)
        all_labels = test_labels.reshape(-1) if all_labels == None else torch.cat((all_labels, test_labels.reshape(-1)), 0)
        
        if span_output != None:
            full_test_texts += test_texts
            full_test_mappings = test_mapping if full_test_mappings == None else torch.cat((full_test_mappings, test_mapping), 0)
            full_test_predictions = predictions if full_test_predictions == None else torch.cat((full_test_predictions, predictions), 0)
    
    if span_output != None:
        span_output += combine_sents(full_test_texts, full_test_mappings.to('cpu'), full_test_predictions.to('cpu'), id2labels, **kwargs)
    metrics = get_metrics(all_labels, all_predictions, id2labels, **kwargs)
    logging.info(f"{dataset_name} loss: {round(test_loss, 5)} \t {dataset_name} metrics: {metrics}")
    return test_loss

def _main():
    start_time = datetime.now()

    args, root_save_path = get_args()

    dataset_params = {
        'labels': args.labels
    }

    # saving the model configurations
    with open(f'{root_save_path}/config.json', 'w') as model_config_file:
        json.dump(args.__dict__, model_config_file, indent=4)

    if not args.evaluate_only:
        assert(args.train_data != None and args.dev_data != None and args.test_data != None)
        trainset = OpenpromptSentDataset(args.train_data, **dataset_params)
        devset = OpenpromptSentDataset(args.dev_data, **dataset_params)
        testset = OpenpromptSentDataset(args.test_data, **dataset_params)

        train_examples = trainset.get_examples()
        dev_examples = devset.get_examples()
        test_examples = testset.get_examples()
    else:
        assert(args.test_data != None)
        testset = OpenpromptSentDataset(args.test_data, **dataset_params)
        test_examples = testset.get_examples()
    
    if args.load_trained_model:
        if 'roberta' in args.model_name:
            plm = RobertaForMaskedLM.from_pretrained(args.model_name)
            checkpoint = torch.load(args.load_trained_model)
            plm.load_state_dict(checkpoint['model_state_dict'],strict=True)
            tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer, already_has_special_tokens=True,return_special_tokens_mask=False)
            model_config = None
            WrapperClass = MLMTokenizerWrapper
        else:
            assert("Only support loading roberta")
    else:
        plm, tokenizer, model_config, WrapperClass = load_plm(args.model_family, args.model_name)
    template_text = get_template(args.prompt_file)
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

    if not args.evaluate_only:
        train_loader = get_promptdataloader(train_examples, template, tokenizer, WrapperClass, args, args.shuffle)
        dev_loader = get_promptdataloader(dev_examples, template, tokenizer, WrapperClass, args, False)
        test_loader = get_promptdataloader(test_examples, template, tokenizer, WrapperClass, args, False)
    else:
        test_loader = get_promptdataloader(test_examples, template, tokenizer, WrapperClass, args, False)


    verbalizer = ManualVerbalizer(tokenizer, num_classes=len(testset.get_id2labels()), label_words=testset.get_verbalizer_labels())
    prompt_model = PromptForClassification(plm=plm,template=template, verbalizer=verbalizer, freeze_plm=args.freeze_plm).to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr = args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(optimizer_grouped_parameters, lr = args.lr, momentum=args.momentum)
    else:
        raise(f'No such optimizer: {args.optimizer}')
    
    if args.loss == 'simple_cross_entropy_loss':
        loss = simple_cross_entropy_loss
    elif args.loss == 'penalize_label_loss':
        loss = penalize_label_loss
    elif args.loss == 'penalize_length_loss':
        loss = penalize_length_loss
    else:
        raise(f'No such loss: {args.loss}')
    
    labels2id = testset.get_labels2id()
    penalize_labels = [labels2id[i] for i in args.penalize_labels] if args.penalize_labels != None else []
    model_params = {
        'penalize_labels': penalize_labels,
        'lmbda': args.lmbda,
        'metrics': args.metrics,
        'process_data': args.process_data, # used for sentence-based text reconstruction
        'device': args.device
    }

    start_epoch = 0
    cur_min_dev_loss = math.inf
    cur_min_dev_loss_epoch = -1

    if args.load_checkpoint:
        logging.info(f'loading checkpoint from {args.load_checkpoint}')
        checkpoint = torch.load(args.load_checkpoint,map_location=f'cuda:{args.cuda}')
        # print(checkpoint['model_state_dict'].keys())
        prompt_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        deepspeed_config = {
            "kernel_inject": False,
            "dtype": "bf16",
            "enable_cuda_graph": False
        }
        engine = deepspeed.init_inference(model=prompt_model, config=deepspeed_config)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        cur_min_dev_loss_epoch = start_epoch
        if args.evaluate_only:
            epoch = checkpoint['epoch']
            test_span_output = [] if args.test_to_span else None
            _ = evaluate(test_loader, engine, loss, testset.get_id2labels(), 'test',test_span_output, **model_params)
            if args.test_to_span:
                with open(args.output_spans_path, 'w') as test_span_out:
                    json.dump(test_span_output, test_span_out, indent=4)
            return

    for epoch in range(start_epoch, args.num_epoch):
        if check_early_stopping(cur_min_dev_loss_epoch, epoch, args.early_stopping_cutoff):
            logging.info(f"early stop at epoch {epoch-1}")
            break
        logging.info(f'start epoch {epoch}')

        train_span_output = [] if args.train_to_span else None
        dev_span_output = [] if args.dev_to_span else None
        test_span_output = [] if args.test_to_span else None

        train(train_loader, prompt_model, loss, optimizer, trainset.get_id2labels(), train_span_output, **model_params)
        dev_loss = evaluate(dev_loader, prompt_model, loss, devset.get_id2labels(), 'dev', dev_span_output, **model_params)
        if dev_loss <= cur_min_dev_loss:
            cur_min_dev_loss = dev_loss
            cur_min_dev_loss_epoch = epoch
        _ = evaluate(test_loader, prompt_model, loss, testset.get_id2labels(), 'test', test_span_output, **model_params)

        # save output train and test spans
        if args.train_to_span:
            with open(f'{root_save_path}/{args.output_spans_path}/epoch{str("{:03d}".format(epoch))}_train.json', 'w') as train_span_out:
                json.dump(train_span_output, train_span_out, indent=4)
        if args.dev_to_span:
            with open(f'{root_save_path}/{args.output_spans_path}/epoch{str("{:03d}".format(epoch))}_dev.json', 'w') as dev_span_out:
                json.dump(dev_span_output, dev_span_out, indent=4)
        if args.test_to_span:
            with open(f'{root_save_path}/{args.output_spans_path}/epoch{str("{:03d}".format(epoch))}_test.json', 'w') as test_span_out:
                json.dump(test_span_output, test_span_out, indent=4)
        
        save_checkpoint(epoch, \
                        prompt_model.state_dict(), \
                        optimizer.state_dict(), \
                        f'{root_save_path}/{args.checkpoint_save_path}')
    
    logging.info(f"lowest dev loss at epoch {cur_min_dev_loss_epoch}")
    end_time = datetime.now()
    logging.info(f'total running time: {end_time - start_time}')


if __name__ == '__main__':
    _main()