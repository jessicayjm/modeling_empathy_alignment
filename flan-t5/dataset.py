import re
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

class DatasetInfo():
    def __init__(self, **kwargs):
        self.include_target = kwargs['include_target']
        self.include_observer = kwargs['include_observer']
        self.label_names = kwargs['labels']
        self.labels2id, self.id2labels = self._create_label_dict()

        self.target_prefix = 'target:\n\n'
        self.observer_prefix = '\n\nobserver:\n\n'

    def get_labels2id(self):
        return self.labels2id

    def get_id2labels(self):
        return self.id2labels

    def _create_label_dict(self):
        if len(set(self.label_names)) != len(self.label_names):
            raise Exception('Duplicate labels not allowed.')
        return {label: id for id, label in enumerate(self.label_names)}, \
                {id: label for id, label in enumerate(self.label_names)}

    def get_labels(self):
        return self.label_names

class BaseDataset(DatasetInfo):
    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.raw_sentences, self.annotations, self.mappings = self._process_annotations()
        self.complete_labels = self._get_labels()
        # sentences: processed cleaner text
        # complete_labels: [[span_text1, label_text1], ...]
        # raw_sentences: filtered non-empty sentences, uncleaned
        # annotations: annotations based on raw_sentences [[start1, end1, label_id1], ...]
        # mappinga: mappings for convert sentences back to texts
        self.sentences, self.complete_labels, self.raw_sentences, self.annotations, self.mappings = self._clean_raw_sentences()

        self.text_prefix = kwargs['text_prefix'] # added before the text input specifying the task type
        self.texts = None
        self.labels = None
        self.tokenizer = kwargs['tokenizer']

        self.num_sent_no_label = 0
        self.num_sents_multiple_labels = 0
        self.total_num_of_labels = 0
        self.sent_count_per_label = defaultdict(int)

    def get_stats(self):
        logging.info('############## Dataset Stats ##############')
        # num of texts
        logging.info(f'number of sentences: {len(self.texts)}')
        logging.info(f'average length of sentences: {sum([len(t) for t in self.texts])/len(self.texts)} chars')
        logging.info(f'number of sentences with no labels: {self.num_sent_no_label}')
        logging.info(f'number of sentences that contains multiple labels: {self.num_sents_multiple_labels}')
        logging.info(f'average labels for each sentence: {self.total_num_of_labels/len(self.texts)}')
        # num of texts for each label
        for key in self.sent_count_per_label.keys():
            logging.info(f'num of sentences has {key}: {self.sent_count_per_label[key]}')
        logging.info('###########################################')
    
    def _clean_sentence(self, sent):
        sent = sent.strip()
        sent = re.compile("[.;:!\'?,\"()\[\]]").sub("", sent)
        sent = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)").sub("", sent)
        return sent

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
            text = text.lower()
            mapping = []
            sentences = []
            annotations = []
            split_pos = self._get_sentence_split_pos(text)
            if type_include == 'target':
                preprocess_annotations = sorted([[ann[0]-len(self.target_prefix), ann[1]-len(self.target_prefix), ann[2]] \
                    for ann in row['annotations'] if ann[1] <= row['target_len'] and ann[2] in self.label_names], key=lambda x: x[0])
            elif type_include == 'observer':
                preprocess_annotations = sorted([[ann[0]-row['target_len'], ann[1]-row['target_len'], ann[2]] \
                    for ann in row['annotations'] if ann[0] >= row['target_len'] - 2 and ann[2] in self.label_names],key=lambda x: x[0])
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
            return mapping, sentences, annotations
    
        sentences = []
        annotations = []
        mappings = []
        self.df['target_len'] = self.df['target'].apply(lambda x: len(x))
        self.df['target_len'] += len(self.target_prefix) + len(self.observer_prefix)
        # self.df['target_len'] = self.df['target'].str.len() + len(self.target_prefix) + len(self.observer_prefix)
        for text_idx, row in self.df.iterrows():
            sent_pos = 0
            if self.include_target:
                maps, sents, anns = process_one_text(row['target'], 'target', text_idx, sent_pos, row['target_len'])
                mappings += maps
                sentences += sents
                annotations += anns
                sent_pos += len(sents)
            if self.include_observer:
                maps, sents, anns = process_one_text(row['observer'], 'observer', text_idx, sent_pos, row['target_len'])
                mappings += maps
                sentences += sents
                annotations += anns
        return sentences, annotations, mappings

    def _get_labels(self):
        # the label of one sentence is determined by the dominant label of all annotations
        def combine_labels(idx, anns):
            labels_per_text = [0 for _ in range(len(self.label_names))]
            labels_with_spans = [] # format: [[span1, label1], [span2, label2],...]
            for ann in anns:
                labels_with_spans.append([self._clean_sentence(self.raw_sentences[idx][ann[0]:ann[1]]), ann[2]])
            return labels_with_spans
        labels = [combine_labels(idx, anns) for idx, anns in enumerate(self.annotations)]
        return labels
    
    def _clean_raw_sentences(self):
        sentences = [self._clean_sentence(sent) for sent in self.raw_sentences]
        new_sents = []
        new_complete_labels = []
        new_raw_sentences = []
        new_annotaitons = []
        new_mappings = []
        for idx, sent in enumerate(sentences):
            if sent != '':
                new_sents.append(sent)
                new_complete_labels.append(self.complete_labels[idx])
                new_raw_sentences.append(self.raw_sentences[idx])
                new_annotaitons.append(self.annotations[idx])
                new_mappings.append(self.mappings[idx])
        return new_sents, new_complete_labels, new_raw_sentences, new_annotaitons, new_mappings

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels
    
    def _generate_labels(self):
        raise NotImplementedError
    
    def _generate_texts(self):
        raise NotImplementedError
    
    
class MultilabelOnlyDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.labels, self.one_hot_labels = self._generate_labels()
        self.texts = self._generate_texts()
        self.src_tokenized, self.tgt_tokenized = self._tokenize()

        if kwargs['show_dataset_stats']:
            self.get_stats()
    
    def _generate_labels(self):
        labels = []
        one_hot_labels = []
        for sent_labels in self.complete_labels:
            one_hot = [0 for _ in range(len(self.labels2id.keys()))]
            if len(sent_labels) == 0:
                labels.append('No Label')
                self.num_sent_no_label += 1
            else:
                unique_labels = set([l[1] for l in sent_labels])
                labels.append(','.join(list(unique_labels))+' </s>')
                for l in unique_labels:
                    one_hot[self.labels2id[l]] = 1
                    self.sent_count_per_label[l] += 1
            self.num_sents_multiple_labels += sum(one_hot) > 1
            self.total_num_of_labels += sum(one_hot)
            one_hot_labels.append(one_hot)

        return labels, one_hot_labels
    
    def _generate_texts(self):
        texts = [self.text_prefix + ': ' + s for s in self.sentences]
        return texts
    
    def _tokenize(self):
        src_tokenized = self.tokenizer(
            self.texts, 
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        tgt_tokenized = self.tokenizer(
            self.labels, 
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        return src_tokenized, tgt_tokenized

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        src_input_ids = self.src_tokenized['input_ids'][idx]
        src_attention_mask = self.src_tokenized['attention_mask'][idx]
        tgt_input_ids = self.tgt_tokenized['input_ids'][idx]
        tgt_attention_mask = self.tgt_tokenized['attention_mask'][idx]
        return {
            'src_input_ids': torch.Tensor(src_input_ids).type(torch.LongTensor),
            'src_attention_mask': torch.Tensor(src_attention_mask).type(torch.LongTensor),
            'tgt_input_ids': torch.Tensor(tgt_input_ids).type(torch.LongTensor),
            'tgt_attention_mask': torch.Tensor(tgt_attention_mask).type(torch.LongTensor),
            'texts': self.raw_sentences[idx],
            'one_hot_labels': torch.Tensor(self.one_hot_labels[idx]).type(torch.LongTensor),
            'mappings': torch.Tensor(self.mappings[idx]).type(torch.LongTensor)
        }
        

class MultilabelwithSpansDataset(BaseDataset):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.labels = self._generate_labels()
        self.texts = self._generate_texts()
        self.src_tokenized, self.tgt_tokenized = self._tokenize()
    
    def _generate_labels(self):
        labels = []
        for sent_labels in self.complete_labels:
            labels.append('Labels:'+'\t'.join([f'"{l[0]} -> {l[1]}' for l in sent_labels])+' </s>')
        return labels
    
    def _generate_texts(self):
        texts = []
        texts.append([self.text_prefix + ':' + s for s in self.sentences])
        return texts
    
    def _tokenize(self):
        src_tokenized = self.tokenizer(
            self.texts, 
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        tgt_tokenized = self.tokenizer(
            self.labels, 
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        return src_tokenized, tgt_tokenized

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        src_input_ids = self.src_tokenized['input_ids'][idx]
        src_attention_mask = self.src_tokenized['attention_mask'][idx]
        tgt_input_ids = self.tgt_tokenized['input_ids'][idx]
        tgt_attention_mask = self.tgt_tokenized['attention_mask'][idx]
        return {
            'src_input_ids': torch.Tensor(src_input_ids).type(torch.LongTensor),
            'src_attention_mask': torch.Tensor(src_attention_mask).type(torch.LongTensor),
            'tgt_input_ids': torch.Tensor(tgt_input_ids).type(torch.LongTensor),
            'tgt_attention_mask': torch.Tensor(tgt_attention_mask).type(torch.LongTensor),
            'mapping': torch.Tensor(self.mappings).type(torch.LongTensor)
        }