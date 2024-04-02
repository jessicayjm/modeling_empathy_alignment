import os
import re
import json
import logging
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from variables import labels, metrics
from utils import get_metrics

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='baseline with only similarity considered')

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--labels', nargs='+', choices=labels, default=labels)
    parser.add_argument('--metrics', nargs='+', choices=metrics, default=metrics)
    parser.add_argument('--random_seed', type=int, default=None)

    parser.add_argument('--saving_folder', type=str, help='folder to save preprocessed data') 
    parser.add_argument('--load_folder', type=str, default=None, help='folder to load preprocessed_data')

    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()
    
    return args


class PairedDataset():
    def __init__(self, df, label_names, save_path=None, load_path=None):
        self.df = df

        self.target_prefix = 'target:\n\n'
        self.observer_prefix = '\n\nobserver:\n\n'
        self.texts = self.df['full_text']
        self.df['target_len'] = self.df['target_text'].str.len() + len(self.target_prefix) + len(self.observer_prefix)
    
        self.label_names = label_names
        self.save_path = save_path

        if load_path:
            with open(load_path, 'r') as f:
                data = json.load(f)
            self.targets = data['targets']
            self.observers = data['observers']
            self.similarities = data['similarities']
            self.overlaps = data['overlaps']
            self.labels = data['labels']
        else:
            self.encode_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

            self.targets, \
            self.observers, \
            self.similarities, \
            self.overlaps, \
            self.labels = self._create_pairs()

    def get_word_dataset(self):
        return np.array(self.overlaps).reshape(-1, 1) , np.array(self.labels)
    
    def get_sent_sim_dataset(self):
        return np.array(self.similarities).reshape(-1, 1) , np.array(self.labels)

    def save_dataset(self):
        if not self.save_path:
            assert("save path is None")
        save_data = {
            "targets": self.targets,
            "observers": self.observers,
            "similarities": self.similarities,
            "overlaps": self.overlaps,
            "labels": self.labels,
        }
        print(len(self.targets), len(self.observers), len(self.labels))
        with open(self.save_path, 'w') as f:
            json.dump(save_data, f, indent=4)
    
    def _create_span(self, span_pos, text):
        span_start = span_pos[0]
        span_end = span_pos[1]
        return (text[span_start:span_end], span_pos[2]) 

    def get_words(self, text):
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = cleaned_text.lower().split()
        return set(words)

    def calculate_word_overlap(self, target_words, observer_words):
        overlap = target_words.intersection(observer_words)
        if len(target_words) == 0 or len(observer_words) == 0:
            return 0
        return len(overlap)/(len(target_words)*len(observer_words))
        

    def _create_pairs(self):
        def process_one_instance(row):
            # alignment pairs
            alignment_pairs = [(align[0][0], align[0][1], align[1][0], align[1][1]) for align in row["alignments"]]
            # get target spans and observer spans
            target_spans = []
            observer_spans = []
            for ann in row['spans']:
                if ann[2] not in self.label_names: continue
                if ann[1] <= row['target_len']:
                    # modify the target start and end to be relative to the start of target text
                    target_spans.append([ann[0]-len(self.target_prefix), ann[1]-len(self.target_prefix), ann[2], ann[3]])
                else:
                    # modify the observer start and end to be relative to the start of observer text
                    observer_spans.append([ann[0]-row['target_len'], ann[1]-row['target_len'], ann[2], ann[3]])
            # permute the target and observer spans
            output_labels = []
            for target_span in target_spans:
                for observer_span in observer_spans:
                    # highly unlikely case
                    if target_span[2] == "Advice" and observer_span[2] == "Pleasantness" \
                        or target_span[2] == "Advice" and observer_span[2] == "Objective Experience" \
                        or target_span[2] == "Anticipated Effort" and observer_span[2] == "Objective Experience": 
                        continue
                    if (target_span[0]+len(self.target_prefix), target_span[1]+len(self.target_prefix), observer_span[0]+row['target_len'], observer_span[1]+row['target_len']) in alignment_pairs:
                        output_labels.append(1)
                    else:
                        output_labels.append(0)    
            output_targets = []
            output_observers = []
            output_similarities = []
            output_overlaps = []
            target_spans_w_context = [self._create_span(t[:3], row['target_text']) for t in target_spans]
            observer_spans_w_context = [self._create_span(t[:3], row['observer_text']) for t in observer_spans]
            
            # encode with sentence-transformer
            target_embeddings = torch.from_numpy(self.encode_model.encode([t[0] for t in target_spans_w_context]))
            observer_embeddings = torch.from_numpy(self.encode_model.encode([o[0] for o in observer_spans_w_context]))

            target_words = [self.get_words(t[0]) for t in target_spans_w_context]
            observer_words = [self.get_words(o[0]) for o in observer_spans_w_context]

            for t_idx, target_span in enumerate(target_spans_w_context):
                for o_idx, observer_span in enumerate(observer_spans_w_context):
                    # highly unlikely case
                    if target_span[1] == "Advice" and observer_span[1] == "Pleasantness" \
                        or target_span[1] == "Advice" and observer_span[1] == "Objective Experience" \
                        or target_span[1] == "Anticipated Effort" and observer_span[1] == "Objective Experience": 
                        continue
                    output_targets.append(target_span[0])
                    output_observers.append(observer_span[0])
                    output_similarities.append(F.cosine_similarity(target_embeddings[t_idx], observer_embeddings[o_idx], dim=0).item())
                    output_overlaps.append(self.calculate_word_overlap(target_words[t_idx], observer_words[o_idx]))
            # sanity check
            assert(len(output_targets)==len(output_observers))
            return output_targets, output_observers, output_similarities, output_overlaps, output_labels

        targets = []
        observers = []
        similarities = []
        overlaps = []
        labels = []
        for row_id, row in self.df.iterrows():
            output_targets, output_observers, output_similarities, output_overlaps, output_labels = process_one_instance(row)
                
            targets += output_targets
            observers += output_observers
            similarities += output_similarities
            overlaps += output_overlaps
            labels += output_labels

        return targets, observers, similarities, overlaps, labels


def evaluate(all_labels, all_predictions, metrics):
    cal_metrics = get_metrics(all_labels, all_predictions, id2labels=None, metrics=metrics, process_data=None)
    print(cal_metrics)  

def _main():
    args = get_args()

    train_df = pd.read_json(args.train_data)
    dev_df = pd.read_json(args.dev_data)
    test_df = pd.read_json(args.test_data)

    if args.saving_folder:
        train_dataset = PairedDataset(train_df, args.labels, save_path=args.saving_folder+'/train_sim.json')
        dev_dataset = PairedDataset(dev_df, args.labels, save_path=args.saving_folder+'/dev_sim.json')
        test_dataset = PairedDataset(test_df, args.labels, save_path=args.saving_folder+'/test_sim.json')
        train_dataset.save_dataset()
        dev_dataset.save_dataset()
        test_dataset.save_dataset()
        return
    elif args.load_folder:
        train_dataset = PairedDataset(train_df, args.labels, load_path=args.load_folder+'/train_sim.json')
        dev_dataset = PairedDataset(dev_df, args.labels, load_path=args.load_folder+'/dev_sim.json')
        test_dataset = PairedDataset(test_df, args.labels, load_path=args.load_folder+'/test_sim.json')
    else: 
        assert("no save path or load path")

    # use word overlap
    train_word, train_word_labels = train_dataset.get_word_dataset()
    dev_word, dev_word_labels = dev_dataset.get_word_dataset()
    test_word, test_word_labels = test_dataset.get_word_dataset()
    clf_word = LogisticRegression(random_state=0).fit(train_word, train_word_labels)
    dev_word_pred = clf_word.predict(dev_word)
    test_word_pred = clf_word.predict(test_word)
    print("word overlap")
    print("dev")
    evaluate(torch.from_numpy(dev_word_labels), torch.from_numpy(dev_word_pred), args.metrics)
    print("test")
    evaluate(torch.from_numpy(test_word_labels), torch.from_numpy(test_word_pred), args.metrics)

    # use sentence transformer 
    train_sim, train_sim_labels = train_dataset.get_sent_sim_dataset()
    dev_sim, dev_sim_labels = dev_dataset.get_sent_sim_dataset()
    test_sim, test_sim_labels = test_dataset.get_sent_sim_dataset()
    clf_sim = LogisticRegression(random_state=0).fit(train_sim, train_sim_labels)
    dev_sim_pred = clf_sim.predict(dev_sim)
    test_sim_pred = clf_sim.predict(test_sim)
    print(dev_sim_pred.shape)
    print((test_sim_pred!=0).sum())
    print("sentence transformer similarity")
    print("dev")
    evaluate(torch.from_numpy(dev_sim_labels), torch.from_numpy(dev_sim_pred), args.metrics)
    print("test")
    evaluate(torch.from_numpy(test_sim_labels), torch.from_numpy(test_sim_pred), args.metrics)

if __name__ == '__main__':
    _main()