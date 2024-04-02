import pickle
from collections import defaultdict
from openprompt.data_utils import InputExample

import numpy as np
import torch 

class OpenpromptSentDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, **kwargs):
        super().__init__()
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
        self.id2labels = data['label_dict']
        self.labels2id = { value : key for key, value in self.id2labels.items()}
        self.texts = data['texts']
        self.labels = data['labels']
        self.offset_mappings = data['offset_mappings']
        self.examples = self._create_input_examples()    
        
    def get_labels2id(self):
        return self.labels2id

    def get_id2labels(self):
        return self.id2labels
    
    def get_verbalizer_labels(self):
        return [[label] for label in self.labels2id.keys()]
    
    def _create_input_examples(self):
        examples = []
        for i in range(len(self.texts)):
            examples.append(InputExample(guid = [self.texts[i], self.offset_mappings[i]], text_a = self.texts[i], label=self.labels[i]))
        return examples
    
    def get_examples(self):
        return self.examples