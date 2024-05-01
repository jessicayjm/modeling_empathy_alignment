import os
import json
import pickle
import random
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import prepare_saving
from utils import get_metrics
from variables import labels

# parsers
parser = argparse.ArgumentParser(description='majority baseline')

# model specifications
parser.add_argument('--test_data', type=str)
parser.add_argument('--random_seed', type=int, default=None)

parser.add_argument('--saving_folder', type=str, default='./runs')

args = parser.parse_args()

_, root_save_path = prepare_saving(args.saving_folder, 'majority_baseline', None, None)
logfile_path = f'{root_save_path}/output.log'
logging.basicConfig(filename=logfile_path, level=logging.INFO)

if args.random_seed != None:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

# always predict self-other agency
def predict(output_shape):
    predictions = torch.zeros(output_shape)+5
    return predictions

def _main():
    # load in datafile
    with open(args.test_data, 'rb') as f:
        data = pickle.load(f)
    
    id2labels = data['label_dict']
    metrics = ["total_accuracy", \
            "macro_recall", \
            "macro_precision", \
            "macro_f1", \
            "per_label_recall", \
            "per_label_precision"]
    test_labels = torch.Tensor(data['labels'])      
    predictions = predict(test_labels.shape)

    metrics = get_metrics(predictions, test_labels, id2labels, metrics=metrics, process_data="feed_single_sent")
    logging.info(metrics)
   

if __name__ == '__main__':
    _main()
