import re
from bisect import bisect
import torch
import re
import logging
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer
from dataset import AlignmentDataset


import sys
sys.path.append('..')
from variables import labels


def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='Siamese Network for alignment detection')

    # model specifications
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default='bert-base-cased')
    parser.add_argument('--target_num_neighbors', type=int, default=0)
    parser.add_argument('--target_context_pos', choices=['preceding', 'succeeding', 'surround', None], type=str, default=None)
    parser.add_argument('--observer_num_neighbors', type=int, default=0)
    parser.add_argument('--observer_context_pos', choices=['preceding', 'succeeding', 'surround', None], type=str, default=None)
    parser.add_argument('--labels', nargs='+', choices=labels, default=labels)

    parser.add_argument('--dataset_save_path', type=str, default=None)

    args = parser.parse_args()
    return args



class BaseDataset():
    def __init__(self, df, **kwargs):
        self.df = df

        self.target_prefix = 'target:\n\n'
        self.observer_prefix = '\n\nobserver:\n\n'
        self.texts = self.df['full_text']
        self.df['target_len'] = self.df['target_text'].str.len() + len(self.target_prefix) + len(self.observer_prefix)
    
        self.target_context_pos = kwargs['target_context_pos']
        self.target_num_neighbors = kwargs['target_num_neighbors']
        self.observer_context_pos = kwargs['observer_context_pos']
        self.observer_num_neighbors = kwargs['observer_num_neighbors']
        self.label_names = kwargs['label_names']
        self.tokenizer = kwargs['tokenizer']

        self.save_path = kwargs["save_path"]

    def get_texts(self):
        return self.texts


class AlignmentDataset(BaseDataset):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.target_spans_input_ids, \
        self.target_spans_attention_mask, \
        self.observer_spans_input_ids, \
        self.observer_spans_attention_mask, \
        self.labels, \
        self.ori_alignments, \
        self.text_mappings = self._create_pairs_tokenized()


    def save_dataset(self):
        if not self.save_path:
            assert("save path is None")
        save_data = {
            "target_spans_input_ids": self.target_spans_input_ids,
            "target_spans_attention_mask": self.target_spans_attention_mask,
            "observer_spans_input_ids": self.observer_spans_input_ids,
            "observer_spans_attention_mask": self.observer_spans_attention_mask,
            "labels": self.labels,
            "ori_alignments": self.ori_alignments,
            "text_mappings": self.text_mappings
        }
        torch.save(save_data, self.save_path)           

    '''split the text into sentences to provide context'''
    def _get_sentence_split_pos(self, text):
        par_index = [_.end() for _ in re.finditer(r'[\n\.?!]+', text)]
        # get the position of punctuations that separate sentences
        segment_pos = [0]
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
    Create the context for the current span.
    Note that target and observer are considered separately, i.e. provide the position
    relative to target_text or observer_text instead of full_text(target:\n\n...observer:\n\n)
    Input:
    span_pos: the start and end position of the current span
    text: the text where the span is extracted from
    segment_pos: the sentence segment positions of the text, this list should be in ascending order
    context_pos: the position of context to consider. 
        Options: 
            preceding: only consider the sentences before the span
            succeeding: only consider the sentences after the span
            surround: consider the sentences both before and after the span
    num_neighbors: the numebr of sentences to add as context
    Return:
    context (str): the context for the current span
    '''
    def _create_span_w_context(self, span_pos, text, segment_pos, context_pos, num_neighbors):
        span_start = span_pos[0]
        span_end = span_pos[1]
        if num_neighbors == 0:
            return (text[span_start:span_end], span_pos[2]) 
        pos_span_start_idx = bisect(segment_pos, span_start)-1 # -1 becasue bisect gives the postion to insert this value
        pos_span_end_idx = min(bisect(segment_pos, span_end), len(segment_pos)-1)-1 # this is same as bisect_right
        # extract the sentence 
        min_pos_context_idx = max(pos_span_start_idx - num_neighbors, 0)
        max_pos_context_idx = min(pos_span_end_idx + num_neighbors, len(segment_pos)-1)
        if context_pos == "preceding":
            return (text[segment_pos[min_pos_context_idx]:span_end], span_pos[2]) 
        elif context_pos == "succeeding":
            return (text[span_start: segment_pos[max_pos_context_idx]], span_pos[2]) 
        elif context_pos == "surround":
            return (text[segment_pos[min_pos_context_idx]: segment_pos[max_pos_context_idx]], span_pos[2])
        else:
            assert("Invalid context position")
    
    '''
    create all possible pairs of spans with context if specified
    Return:
    target_spans_input_ids (torch.LongTensor)
    target_spans_attention_mask (torch.LongTensor)
    observer_spans_input_ids (torch.LongTensor)
    observer_spans_attention_mask (torch.LongTensor)
    labels (torch.LongTensor)
    text_mappings (torch.LongTensor): text_idx in order to find where the span is
    '''
    def _create_pairs_tokenized(self):
        def process_one_instance(row):
            # alignment pairs
            alignment_pairs = [(align[0][0], align[0][1], align[1][0], align[1][1]) for align in row["alignments"]]
            # get target spans and observer spans
            output_ori_alignments = []
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
                    output_ori_alignments.append([target_span[0]+len(self.target_prefix), target_span[1]+len(self.target_prefix), observer_span[0]+row['target_len'], observer_span[1]+row['target_len']])
                    if (target_span[0]+len(self.target_prefix), target_span[1]+len(self.target_prefix), observer_span[0]+row['target_len'], observer_span[1]+row['target_len']) in alignment_pairs:
                        output_labels.append(1)
                    else:
                        output_labels.append(0)    
            output_targets = []
            output_observers = []
            segment_pos_target = self._get_sentence_split_pos(row['target_text'])
            segment_pos_observer = self._get_sentence_split_pos(row['observer_text'])
            target_spans_w_context = [self._create_span_w_context(t[:3], row['target_text'], segment_pos_target, self.target_context_pos, self.target_num_neighbors) for t in target_spans]
            observer_spans_w_context = [self._create_span_w_context(t[:3], row['observer_text'], segment_pos_observer, self.observer_context_pos, self.observer_num_neighbors) for t in observer_spans]
            for target_span in target_spans_w_context:
                for observer_span in observer_spans_w_context:
                    # highly unlikely case
                    if target_span[1] == "Advice" and observer_span[1] == "Pleasantness" \
                        or target_span[1] == "Advice" and observer_span[1] == "Objective Experience" \
                        or target_span[1] == "Anticipated Effort" and observer_span[1] == "Objective Experience": 
                        continue
                    output_targets.append(target_span[0])
                    output_observers.append(observer_span[0])
            # sanity check
            assert(len(output_targets)==len(output_observers))
            assert(len(output_ori_alignments)==len(output_targets))
            return output_targets, output_observers, output_labels, output_ori_alignments

        ori_alignments = []
        targets = []
        observers = []
        labels = []
        text_mappings = [] # for mapping the alignment back
        for row_idx, row in self.df.iterrows():
            if "target_seq_id" and "observer_seq_id" in row.keys():
                text_idx = [row["target_seq_id"],row["observer_seq_id"]]
            else:
                text_idx = row_idx
            output_targets, output_observers, output_labels, output_ori_alignments = process_one_instance(row)
            ori_alignments += output_ori_alignments
            text_mappings += [text_idx] * len(output_targets)
                
            targets += output_targets
            observers += output_observers
            labels += output_labels

        # print(text_mappings[12200:12225])
        # exit(0)    
        # tokenize
        target_spans_encodings = self.tokenizer(targets, padding=True, truncation=True)
        target_spans_input_ids = target_spans_encodings['input_ids']
        target_spans_attention_mask = target_spans_encodings['attention_mask']
        observer_spans_encodings = self.tokenizer(observers, padding=True, truncation=True)
        observer_spans_input_ids = observer_spans_encodings['input_ids']
        observer_spans_attention_mask = observer_spans_encodings['attention_mask']
        return torch.LongTensor(target_spans_input_ids), \
                torch.LongTensor(target_spans_attention_mask), \
                torch.LongTensor(observer_spans_input_ids), \
                torch.LongTensor(observer_spans_attention_mask), \
                torch.Tensor(labels).type(torch.FloatTensor), \
                torch.LongTensor(ori_alignments), \
                torch.LongTensor(text_mappings)


def _main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    dataset_params = {
        'tokenizer': tokenizer,
        'label_names': args.labels,
        'target_num_neighbors': args.target_num_neighbors,
        'target_context_pos': args.target_context_pos,
        'observer_num_neighbors': args.observer_num_neighbors,
        'observer_context_pos': args.observer_context_pos,
        'save_path': args.dataset_save_path
    }
    test_df = pd.read_json(args.test_data)
    testset = AlignmentDataset(test_df, **dataset_params)
    testset.save_dataset()

if __name__ == '__main__':
    _main()
