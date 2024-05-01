import os
import re
import json
import pickle
import numpy as np
import time
import logging
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('../..')
from variables import labels

def get_args():
    parser = argparse.ArgumentParser(description='BERT models training')

    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--dev_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--cased', action='store_true', default=False)
    parser.add_argument('--labels', nargs='+', choices=labels, default=labels)
    parser.add_argument('--include_target', action='store_true', default=False)
    parser.add_argument('--include_observer', action='store_true', default=False)
    parser.add_argument('--show_dataset_stats', action='store_true', default=True)
    parser.add_argument('--single_label', action='store_true')
    parser.add_argument('--multi_label', action='store_true')
    parser.add_argument('--save_full_data', action='store_true', default=False)
    parser.add_argument('--saving_folder', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.saving_folder):
        os.makedirs(args.saving_folder)
    
    if args.single_label:
        if not os.path.exists(f"{args.saving_folder}/single_label_sentence"):
            os.makedirs(f"{args.saving_folder}/single_label_sentence")
    
    if args.multi_label:
        if not os.path.exists(f"{args.saving_folder}/multi_label_sentence"):
            os.makedirs(f"{args.saving_folder}/multi_label_sentence")
    
    with open(f'{args.saving_folder}/config.json', 'w') as config_file:
        json.dump(args.__dict__, config_file, indent=4)

    return args

class BaseDataset():
    def __init__(self, df, **kwargs):
        self.df = df
        self.include_target = kwargs['include_target']
        self.include_observer = kwargs['include_observer']
        self.label_names = kwargs['labels'] if kwargs['multilabel'] else ['No Label'] + kwargs['labels']

        self.target_prefix = 'target:\n\n'
        self.observer_prefix = '\n\nobserver:\n\n'

        self.labels2id, self.id2labels = self._create_label_dict()
        self.save_dir = kwargs['save_dir']
        self.split = kwargs['split']
        self.cased = kwargs['cased']
        self.save_full_data = kwargs['save_full_data']

    def _create_label_dict(self):
        if len(set(self.label_names)) != len(self.label_names):
            raise Exception('Duplicate labels not allowed.')
        return {label: id for id, label in enumerate(self.label_names)}, \
                {id: label for id, label in enumerate(self.label_names)}


class SentenceClassificationDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.multilabel = kwargs['multilabel'] if 'multilabel' in kwargs.keys() else None
        self.num_sents_multiple_labels = 0
        self.all_info_per_instance = []
        self.offset_mappings = [] # mapping of sentences to full text
        self.texts, self.annotations, self.labels = self._process_annotations()

    def get_stats(self):
        with open(f"{self.save_dir}/{self.split}_stats.log", "w") as stats_f:
            stats_f.write('############## Sentence Dataset Stats ##############\n')
            # num of texts
            stats_f.write(f'number of sentences: {len(self.texts)}\n')
            stats_f.write(f'number of sentences that contains multiple labels: {self.num_sents_multiple_labels}\n')
            # num of labels for each class after tokenization
            stats_f.write("num of labels for each class\n")
            for key in self.labels2id.keys():
                if self.multilabel: num_label = (np.array(self.labels)[:,self.labels2id[key]]).sum()
                else: num_label = (np.array(self.labels) == self.labels2id[key]).sum()
                stats_f.write(f'{key}: {num_label}\n')
            stats_f.write('#####################################################\n')

    def save_dataset(self):
        save_obj = {
            "label_dict": self.id2labels,
            "texts": self.texts,
            "labels": self.labels,
            "offset_mappings": self.offset_mappings
        }
        if not self.multilabel:
            with open(f"{self.save_dir}/single_label_sentence_data_{self.split}.pkl", 'wb') as fp:
                pickle.dump(save_obj, fp)
            if self.save_full_data:
                with open(f"{self.save_dir}/single_label_sentence_data_{self.split}_full_info.pkl", 'wb') as fp:
                    pickle.dump(self.all_info_per_instance, fp)
        else:
            with open(f"{self.save_dir}/multi_label_sentence_data_{self.split}.pkl", 'wb') as fp:
                pickle.dump(save_obj, fp)
            if self.save_full_data:
                with open(f"{self.save_dir}/multi_label_sentence_data_{self.split}_full_info.pkl", 'wb') as fp:
                    pickle.dump(self.all_info_per_instance, fp)
    

    def _get_sentence_split_pos(self, text):
        par_index = [_.end() for _ in re.finditer(r'[\n\.?!]+', text)]
        # get the position of punctuations that separate sentences
        segment_pos = []
        prev_idx = 0
        # clear the segment positions if two punctuations are too close
        for i in par_index:
            if (i-prev_idx) <= 3:
                if len(segment_pos) != 0:
                    segment_pos[-1] = i
                    prev_idx = i
                continue
            segment_pos.append(i)
            prev_idx = i
        if len(segment_pos) == 0:
            segment_pos.append(len(text))
        elif len(text)-segment_pos[-1]>3:
            segment_pos.append(len(text))
        else: 
            segment_pos[-1] = len(text)
        return segment_pos


    '''
    if only include_target or include_observer is set True, 
    split the text and recalculate the span_positions as it is specified with 
    'target:\n\n' and '\n\nobserver:\n\n'
    '''
    def _process_annotations(self):
        def process_one_text(text, type_include, text_idx, sent_pos, target_len):
            mapping = []
            sentences = []
            annotations = []
            sent_labels = []
            split_pos = self._get_sentence_split_pos(text)
            if type_include == 'target':
                preprocess_annotations = sorted([[ann[0]-len(self.target_prefix), ann[1]-len(self.target_prefix), self.labels2id[ann[2]]] \
                    for ann in row['spans'] if ann[1] <= row['target_len'] and ann[2] in self.labels2id.keys()], key=lambda x: x[0])
            elif type_include == 'observer':
                preprocess_annotations = sorted([[ann[0]-row['target_len'], ann[1]-row['target_len'], self.labels2id[ann[2]]] \
                    for ann in row['spans'] if ann[0] >= row['target_len'] - 2 and ann[2] in self.labels2id.keys()],key=lambda x: x[0])
                    # -2 to avoid '\n\n' before observer is included in label
            # split text and annotations by sentences
            prev_idx = 0
            for sent_idx, cur_idx in enumerate(split_pos):
                sentences.append(text[prev_idx:cur_idx])
                mapping.append([text_idx, sent_pos + sent_idx, target_len]) #[text_idx, pos_in_text, target_length]
                annotations_tmp = []
                for ann in preprocess_annotations:
                    if ann[1] <= prev_idx or ann[0] >= cur_idx:
                        continue
                    new_ann = [max(ann[0], prev_idx)-prev_idx,
                                min(ann[1], cur_idx)-prev_idx, 
                                ann[2]]
                    annotations_tmp.append(new_ann)
                annotations.append(annotations_tmp)
                prev_idx = cur_idx

            # combine labels
            # the label of one sentence is determined by the dominant label of all annotations
            for anns in annotations:
                if self.multilabel:
                    labels_per_text = [0 for _ in range(len(self.label_names))]
                    for ann in anns:
                        labels_per_text[ann[2]] = 1
                    self.num_sents_multiple_labels += sum(labels_per_text) > 1
                    sent_labels.append(labels_per_text)
                else:
                    labels_length_count = [0 for _ in range(len(self.label_names))]
                    for ann in anns:
                        labels_length_count[ann[2]] += ann[1]-ann[0]
                    self.num_sents_multiple_labels += np.count_nonzero(labels_length_count) > 1
                    if sum(labels_length_count) == 0: sent_labels.append(self.labels2id['No Label'])
                    else: sent_labels.append(labels_length_count.index(max(labels_length_count)))
            return mapping, sentences, annotations, sent_labels
    
        sentences = []
        annotations = []
        sent_labels = []
        
        self.df['target_len'] = self.df['target_text'].str.len() + len(self.target_prefix) + len(self.observer_prefix)
        for row_idx, row in self.df.iterrows():
            if "seq_id" in row.keys():
                text_idx = row["seq_id"]
            else:
                text_idx = row_idx
            if self.save_full_data:
                per_text_full_info = {
                    'id': row['id'],
                    'subreddit': row['subreddit'],
                    'parent_id': row['parent_id'],
                    'target_id': row['target_id'],
                    'target_text': row['target_text'],
                    'observer_id': row['observer_id'],
                    'observer_text': row['observer_text'],
                    'full_text': row['full_text'],
                    'distress_score': row['distress_score'],
                    'condolence_score': row['condolence_score'],
                    'empathy_score': row['empathy_score'],
                    "label_dict": self.id2labels,
                    'spans': row['spans']
                }
            sent_pos = 0
            if self.include_target:
                if self.cased:
                    maps, sents, anns, lbls = process_one_text(row['target_text'], 'target', text_idx, sent_pos, row['target_len'])
                else:
                    maps, sents, anns, lbls = process_one_text(row['target_text'].lower(), 'target', text_idx, sent_pos, row['target_len'])
                self.offset_mappings += maps
                sentences += sents
                annotations += anns
                sent_labels += lbls
                sent_pos += len(sents)
                if self.save_full_data:
                    per_text_full_info['target_sents'] = sents
                    per_text_full_info['target_sent_annotations'] = anns
                    per_text_full_info['target_sent_offset_mappings'] = maps
                    per_text_full_info['target_sent_labels'] = lbls
            if self.include_observer:
                if self.cased:
                    maps, sents, anns, lbls = process_one_text(row['observer_text'], 'observer', text_idx, sent_pos, row['target_len'])
                else:
                    maps, sents, anns, lbls = process_one_text(row['observer_text'].lower(), 'observer', text_idx, sent_pos, row['target_len'])
                self.offset_mappings += maps
                sentences += sents
                annotations += anns
                sent_labels += lbls
                if self.save_full_data:
                    per_text_full_info['observer_sents'] = sents
                    per_text_full_info['observer_sent_annotations'] = anns
                    per_text_full_info['observer_sent_offset_mappings'] = maps
                    per_text_full_info['observer_sent_labels'] = lbls
            if self.save_full_data:
                self.all_info_per_instance.append(per_text_full_info)
        return sentences, annotations, sent_labels

def process(args, filename, data_split):
    print(f"start {filename}")
    df = pd.read_json(filename)

    # sentence-level dataset, single label
    if args.single_label:
        print("process single label")
        ssent_dataset_params = {
            'include_target': args.include_target,
            'include_observer': args.include_observer,
            'labels': args.labels,
            'multilabel': False,
            'cased': args.cased,
            'save_full_data': args.save_full_data,
            'save_dir': f"{args.saving_folder}/single_label_sentence"
        }
        ssent_dataset = SentenceClassificationDataset(df, **ssent_dataset_params, split=data_split)
        if args.show_dataset_stats:
            ssent_dataset.get_stats()
        ssent_dataset.save_dataset()
        print("single label saved")

    # sentence-level dataset, multi-label
    if args.multi_label:
        print("process multi label")
        msent_dataset_params = {
            'include_target': args.include_target,
            'include_observer': args.include_observer,
            'labels': args.labels,
            'multilabel': True,
            'cased': args.cased,
            'save_full_data': args.save_full_data,
            'save_dir': f"{args.saving_folder}/multi_label_sentence"
        }
        msent_dataset = SentenceClassificationDataset(df, **msent_dataset_params, split=data_split)
        if args.show_dataset_stats:
            msent_dataset.get_stats()
        msent_dataset.save_dataset()
        print("multi label save")


if __name__ == '__main__':
    args = get_args()
    print("start preprocess")
    if args.train_data == None and args.dev_data == None and args.test_data == None:
        assert("No data provided")
    if args.train_data != None:
        process(args, args.train_data, 'train')
    if args.dev_data != None:
        process(args, args.dev_data, 'dev')
    if args.test_data != None:
        process(args, args.test_data, 'test')
    print("preprocess finished")