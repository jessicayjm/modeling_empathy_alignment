import argparse
import os
import json
import time
import logging
import random
import re

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AutoModelForSeq2SeqLM

import sys
sys.path.append('..')
from utils import get_metrics
from utils import combine_sents


class FLAN_T5(pl.LightningModule):
    def __init__(self, data, **kwargs):
        super().__init__()
        self.config = kwargs
        self.trainset = data['trainset']
        self.valset = data['valset']
        self.testset = data['testset']

        self.dataset_params = kwargs['dataset_params']
        self.loader_params = kwargs['loader_params']
        self.model_params = kwargs['model_params']
        self.output_params = kwargs['output_params']
        self.labels2id = self.trainset.get_labels2id()
        self.id2labels = self.trainset.get_id2labels()
        self.tokenizer = self.dataset_params['tokenizer']

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_params['model_name'])

        logging.basicConfig(filename=kwargs['output_params']['logfile_path'], level=logging.INFO)

    '''
    override Trainer function
    define the forward pass
    '''
    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        lm_labels=None
        ):

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

    '''
    override Trainer function
    define the train DataLoader
    '''
    def train_dataloader(self):
        # loader_params = self.loader_params
        # if 'shuffle' in self.loader_params.keys(): loader_params['shuffle'] = False
        return torch.utils.data.DataLoader(self.trainset, **self.loader_params)

    '''
    override Trainer function
    define the validation DataLoader
    '''
    def val_dataloader(self):
        loader_params = self.loader_params
        if 'shuffle' in self.loader_params.keys(): loader_params['shuffle'] = False
        return torch.utils.data.DataLoader(self.valset, **self.loader_params)

    '''
    override Trainer function
    define the test DataLoader
    '''
    def test_dataloader(self):    
        loader_params = self.loader_params
        if 'shuffle' in self.loader_params.keys(): loader_params['shuffle'] = False     
        return torch.utils.data.DataLoader(self.testset, **self.loader_params)

    '''
    override Trainer function
    set the optimizer
    '''
    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        # optimizer_parameters = [
        #     {
        #         "params": [
        #             p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay": 0.001,
        #     },
        #     {
        #         "params": [
        #             p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay": 0.0,
        #     },
        # ]
        if self.model_params['optimizer'] == 'AdamW':
            # return optim.AdamW(optimizer_parameters, lr = self.model_params['lr'])
            return optim.AdamW(self.model.parameters(),lr = self.model_params['lr'])
            # return FusedAdam(optimizer_parameters, lr = self.model_params['lr'])
        elif self.model_params['optimizer'] == 'SGD':
            return optim.SGD(self.model.parameters(),lr = self.model_params['lr'], momentum=self.model_params['momentum'])
            # return optim.SGD(optimizer_parameters, lr = self.model_params['lr'], momentum=self.model_params['momentum'])
        else:
            raise(f'No such optimizer: {self.model_params["optimizer"]}')
    
    # merge dict object outputs
    def _merge_epoch_outputs(self, step_outputs):
        merged_output = {}
        for output in step_outputs:
            output_keys = output.keys()
            for k in output_keys:
                if k not in merged_output.keys():
                    merged_output[k] = output[k]
                    continue
                if k == 'loss':
                    merged_output[k] = torch.hstack((merged_output[k],output[k]))
                else: # k == 'data
                    for subkey in output[k].keys():
                        if subkey == 'texts': merged_output[k][subkey] += output[k][subkey]
                        else: merged_output[k][subkey] = torch.vstack((merged_output[k][subkey], output[k][subkey]))   
        return merged_output

    def _get_predictions(self, batch):
        generated_ids = self.model.generate(
            input_ids=batch['src_input_ids'],
            attention_mask=batch['src_attention_mask'],
            num_beams=self.model_params['pred_num_beams'],
            max_length=self.model_params['pred_max_length'],
            repetition_penalty=self.model_params['pred_repetition_penalty'],
            length_penalty=self.model_params['pred_length_penalty'],
            early_stopping=True,
            use_cache=True,
        )
        # map ids to texts
        pred_texts = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        # map texts to one-hot encoding for metric calculation
        pred_mat = torch.zeros((batch['src_input_ids'].shape[0], len(self.labels2id.keys()))).to(self.model_params['device'])
        for i, pt in enumerate(pred_texts):
            pt_labels = pt.split(',')
            pt_labels = [ptl.strip() for ptl in pt_labels]
            for ptl in pt_labels:
                if ptl in self.labels2id.keys():
                    pred_mat[i,self.labels2id[ptl]] = 1
        return pred_texts, pred_mat

    # def set_seed(self):
    #     random.seed(0)
    #     np.random.seed(0)
    #     torch.manual_seed(0)
    #     torch.cuda.manual_seed(0)
    #     torch.cuda.manual_seed_all(0)
    #     torch.backends.cudnn.deterministic=True
    #     torch.backends.cudnn.benchmark = True
    #     seed_everything(0, workers=True)
    
    # def on_fit_start(self): self.set_seed()
    # def on_train_start(self): self.set_seed()
    # def on_validation_start(self): self.set_seed()
    # def on_test_start(self): self.set_seed()

    '''
    override Trainer function
    code when training each step
    '''
    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['src_input_ids'], 
            attention_mask=batch['src_attention_mask'],
            labels=batch['tgt_input_ids'],
            decoder_attention_mask=batch['tgt_attention_mask'])
        loss = outputs[0]
        return {'loss': loss, 'data': batch}
    
    '''
    override Trainer function
    code run after one epoch is finished
    '''
    def training_epoch_end(self, training_step_outputs):
        outputs = self._merge_epoch_outputs(training_step_outputs)
        # calculate total loss
        train_loss = torch.mean(outputs['loss']).item()
        print(outputs['loss'])
        # evaluate performance
        # reduce the batch dimension
        data = outputs['data']
        pred_texts, pred_mat = self._get_predictions(data)
        metrics = get_metrics(data['one_hot_labels'], pred_mat, self.id2labels, **self.model_params)

        # map to spans
        if self.output_params['train_to_span']:
            train_span_output = combine_sents(data['texts'], data['mappings'].to('cpu'), pred_mat.to('cpu'), self.id2labels, **self.model_params)
            with open(f"{self.output_params['output_spans_path']}/epoch{str('{:03d}'.format(self.current_epoch))}_train.json", 'w') as train_span_out:
                json.dump(train_span_output, train_span_out, indent=4)
        
        logging.info(f"Epoch {self.current_epoch}")
        logging.info(f"train loss: {round(train_loss, 5)} \t train metrics: {metrics}")  

    '''
    override Trainer function
    optimzer update
    '''
    def optimizer_step(self, 
                        epoch,batch_idx, 
                        optimizer, 
                        optimizer_idx, 
                        optimizer_closure=None, 
                        on_tpu=None, 
                        using_native_amp=None, 
                        using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    '''
    override Trainer function
    '''
    def validation_step(self, batch, batch_idx):
        outputs = self.model(
                input_ids=batch['src_input_ids'], 
                attention_mask=batch['src_attention_mask'],
                labels=batch['tgt_input_ids'],
                decoder_attention_mask=batch['tgt_attention_mask'])
        loss = outputs[0]

        return {'loss': loss, 'data': batch}

    '''
    override Trainer function
    '''
    def validation_epoch_end(self, validation_step_outputs):
        outputs = self._merge_epoch_outputs(validation_step_outputs)

        # calculate total loss
        val_loss = torch.mean(outputs['loss']).item()

        # evaluate performance
        # reduce the batch dimension
        data = outputs['data']
        pred_texts, pred_mat = self._get_predictions(data)
        metrics = get_metrics(data['one_hot_labels'], pred_mat, self.id2labels, **self.model_params)
        # map to spans
        if self.output_params['val_to_span']:
            val_span_output = combine_sents(data['texts'], data['mappings'].to('cpu'), pred_mat.to('cpu'), self.id2labels, **self.model_params)
            with open(f"{self.output_params['output_spans_path']}/epoch{str('{:03d}'.format(self.current_epoch))}_val.json", 'w') as val_span_out:
                json.dump(val_span_output, val_span_out, indent=4)
        
        logging.info(f"val loss: {round(val_loss, 5)} \t val metrics: {metrics}") 
    
    '''
    override Trainer function
    '''
    def test_step(self, batch, batch_idx):
        outputs = self.model(
                input_ids=batch['src_input_ids'], 
                attention_mask=batch['src_attention_mask'],
                labels=batch['tgt_input_ids'],
                decoder_attention_mask=batch['tgt_attention_mask'])
        loss = outputs[0]

        return {'loss': loss, 'data': batch}

    '''
    override Trainer function
    '''
    def test_epoch_end(self, test_step_outputs):
        outputs = self._merge_epoch_outputs(test_step_outputs)

        # calculate total loss
        test_loss = torch.mean(outputs['loss']).item()

        # evaluate performance
        # reduce the batch dimension
        data = outputs['data']
        pred_texts, pred_mat = self._get_predictions(data)
        metrics = get_metrics(data['one_hot_labels'], pred_mat, self.id2labels, **self.model_params)

        # map to spans
        if self.output_params['test_to_span']:
            test_span_output = combine_sents(data['texts'], data['mappings'].to('cpu'), pred_mat.to('cpu'), self.id2labels, **self.model_params)
            with open(f"{self.output_params['output_spans_path']}/epoch{str('{:03d}'.format(self.current_epoch))}_test.json", 'w') as test_span_out:
                json.dump(test_span_output, test_span_out, indent=4)
        
        logging.info(f"test loss: {round(test_loss, 5)} \t test metrics: {metrics}")