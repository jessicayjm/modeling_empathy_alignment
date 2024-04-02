import re
import time
import pickle
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

class BertSentDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, **kwargs):
        super().__init__()
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
        self.id2labels = data['label_dict']
        self.labels2id = { value : key for key, value in self.id2labels.items()}
        self.texts = data['texts']
        self.tokenizer = kwargs['tokenizer']

        self.input_ids, \
        self.attention_masks = self._tokenize()
        self.labels = torch.Tensor(data['labels']).type(torch.LongTensor)
        self.offset_mappings = torch.Tensor(data['offset_mappings']).type(torch.LongTensor)           
        
    def get_labels2id(self):
        return self.labels2id

    def get_id2labels(self):
        return self.id2labels

    def _tokenize(self):
        text_encodings = self.tokenizer(self.texts, padding=True, truncation=True)
        input_ids = text_encodings['input_ids']
        attention_masks = text_encodings['attention_mask']
        return torch.Tensor(input_ids).type(torch.LongTensor), \
               torch.Tensor(attention_masks).type(torch.LongTensor)
               

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], \
               self.input_ids[idx], \
               self.attention_masks[idx], \
               self.offset_mappings[idx], \
               self.labels[idx]