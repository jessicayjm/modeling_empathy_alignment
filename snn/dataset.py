import re
from bisect import bisect
import logging
import torch

class BaseDataset():
    def __init__(self, df, **kwargs):            
        self.load_processed_dataset = kwargs["load_processed_dataset"]

        if not self.load_processed_dataset:
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
        self.to_alignment = kwargs['to_alignment']

        self.save_path = kwargs["save_path"]
        

    def get_texts(self):
        return self.texts


class AlignmentDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        if self.to_alignment:
            if self.load_processed_dataset:
                dataset = torch.load(kwargs["load_processed_dataset"])
                self.target_spans_input_ids = dataset["target_spans_input_ids"]
                self.target_spans_attention_mask = dataset["target_spans_attention_mask"]
                self.observer_spans_input_ids = dataset["observer_spans_input_ids"]
                self.observer_spans_attention_mask = dataset["observer_spans_attention_mask"]
                self.labels = dataset["labels"]
                self.ori_alignments = dataset["ori_alignments"]
                self.text_mappings = dataset["text_mappings"]
            else:
                self.target_spans_input_ids, \
                self.target_spans_attention_mask, \
                self.observer_spans_input_ids, \
                self.observer_spans_attention_mask, \
                self.labels, \
                self.ori_alignments, \
                self.text_mappings = self._create_pairs_tokenized()
        else:
            if self.load_processed_dataset:
                dataset = torch.load(kwargs["load_processed_dataset"])
                self.target_spans_input_ids = torch.LongTensor(dataset["target_spans_input_ids"])
                self.target_spans_attention_mask = torch.LongTensor(dataset["target_spans_attention_mask"])
                self.observer_spans_input_ids = torch.LongTensor(dataset["observer_spans_input_ids"])
                self.observer_spans_attention_mask = torch.LongTensor(dataset["observer_spans_attention_mask"])
                self.labels = torch.FloatTensor(dataset["labels"])
            else:
                self.target_spans_input_ids, \
                self.target_spans_attention_mask, \
                self.observer_spans_input_ids, \
                self.observer_spans_attention_mask, \
                self.labels = self._create_pairs_tokenized()

    def get_stats(self):
        if not self.load_processed_dataset:
            logging.info('############## Dataset Stats ##############')
            logging.info(f'number of texts: {len(self.texts)}')
            logging.info(f'total number of alignments: {int(self.labels.sum().item())}')
            logging.info(f'total number of pairs: {self.labels.shape[0]}')
            logging.info('###########################################')

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
        print("dataset saved.")
           

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
            if self.to_alignment:
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
                    if self.to_alignment:
                        # relative to full_text
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
            if self.to_alignment:
                assert(len(output_ori_alignments)==len(output_targets))
            if self.to_alignment:
                return output_targets, output_observers, output_labels, output_ori_alignments
            return output_targets, output_observers, output_labels

        if self.to_alignment:
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
            if self.to_alignment:
                output_targets, output_observers, output_labels, output_ori_alignments = process_one_instance(row)
                ori_alignments += output_ori_alignments
                text_mappings += [text_idx] * len(output_targets)
            else:
                output_targets, output_observers, output_labels = process_one_instance(row)
            targets += output_targets
            observers += output_observers
            labels += output_labels
            
        # tokenize
        target_spans_encodings = self.tokenizer(targets, padding=True, truncation=True)
        target_spans_input_ids = target_spans_encodings['input_ids']
        target_spans_attention_mask = target_spans_encodings['attention_mask']
        observer_spans_encodings = self.tokenizer(observers, padding=True, truncation=True)
        observer_spans_input_ids = observer_spans_encodings['input_ids']
        observer_spans_attention_mask = observer_spans_encodings['attention_mask']
        if self.to_alignment:
            return torch.LongTensor(target_spans_input_ids), \
                torch.LongTensor(target_spans_attention_mask), \
                torch.LongTensor(observer_spans_input_ids), \
                torch.LongTensor(observer_spans_attention_mask), \
                torch.Tensor(labels).type(torch.FloatTensor), \
                torch.LongTensor(ori_alignments), \
                torch.LongTensor(text_mappings)
        return torch.LongTensor(target_spans_input_ids), \
                torch.LongTensor(target_spans_attention_mask), \
                torch.LongTensor(observer_spans_input_ids), \
                torch.LongTensor(observer_spans_attention_mask), \
                torch.Tensor(labels).type(torch.FloatTensor)
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if self.to_alignment:
            return self.target_spans_input_ids[idx], \
               self.target_spans_attention_mask[idx], \
               self.observer_spans_input_ids[idx], \
               self.observer_spans_attention_mask[idx], \
               self.labels[idx], \
               self.ori_alignments[idx], \
               self.text_mappings[idx]
        return self.target_spans_input_ids[idx], \
               self.target_spans_attention_mask[idx], \
               self.observer_spans_input_ids[idx], \
               self.observer_spans_attention_mask[idx], \
               self.labels[idx]


class AlignmentDatasetFullTarget(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        if self.to_alignment:
            self.target_spans_input_ids, \
            self.target_spans_attention_mask, \
            self.observer_spans_input_ids, \
            self.observer_spans_attention_mask, \
            self.labels, \
            self.ori_alignments, \
            self.text_mappings = self._create_pairs()
        else:
            self.target_spans_input_ids, \
            self.target_spans_attention_mask, \
            self.observer_spans_input_ids, \
            self.observer_spans_attention_mask, \
            self.labels = self._create_pairs()

    def get_stats(self):
        logging.info('############## Dataset Stats ##############')
        logging.info(f'number of texts: {len(self.texts)}')
        logging.info(f'total number of alignments: {int(self.labels.sum().item())}')
        logging.info(f'total number of pairs: {self.labels.shape[0]}')
        logging.info('###########################################')

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
    
    def _create_pairs(self):
        def process_one_instance(row):
            # alignment pairs
            aligned_observers = set([(align[1][0], align[1][1]) for align in row["alignments"]])
            # get target spans and observer spans
            if self.to_alignment:
                output_ori_alignments = []
            observer_spans = []
            for ann in row['spans']:
                if ann[2] not in self.label_names or ann[1] <= row['target_len']: continue
                # modify the observer start and end to be relative to the start of observer text
                observer_spans.append([ann[0]-row['target_len'], ann[1]-row['target_len'], ann[2], ann[3]])
            
            output_targets = []
            output_observers = []
            output_labels = []
            for observer_span in observer_spans:
                output_targets.append(row["target_text"])
                output_observers.append(observer_span[3])
                if self.to_alignment:
                    # relative to full_text
                    # only store the start of observer span is enough to identify it
                    output_ori_alignments.append(observer_span[0]+row['target_len'])
                if (observer_span[0]+row['target_len'], observer_span[1]+row['target_len']) in aligned_observers:
                    output_labels.append(1)
                else:
                    output_labels.append(0)                   
            # sanity check
            assert(len(output_targets)==len(output_observers))
            if self.to_alignment:
                assert(len(output_ori_alignments)==len(output_targets))
            if self.to_alignment:
                return output_targets, output_observers, output_labels, output_ori_alignments
            return output_targets, output_observers, output_labels

        if self.to_alignment:
            ori_alignments = []
        targets = []
        observers = []
        labels = []
        text_mappings = [] # for mapping the alignment back
        for text_idx, row in self.df.iterrows():
            if self.to_alignment:
                output_targets, output_observers, output_labels, output_ori_alignments = process_one_instance(row)
                ori_alignments += output_ori_alignments
                text_mappings += [text_idx] * len(output_targets)
            else:
                output_targets, output_observers, output_labels = process_one_instance(row)
            targets += output_targets
            observers += output_observers
            labels += output_labels   
        # tokenize
        target_spans_encodings = self.tokenizer(targets, padding=True, truncation=True)
        target_spans_input_ids = target_spans_encodings['input_ids']
        target_spans_attention_mask = target_spans_encodings['attention_mask']
        observer_spans_encodings = self.tokenizer(observers, padding=True, truncation=True)
        observer_spans_input_ids = observer_spans_encodings['input_ids']
        observer_spans_attention_mask = observer_spans_encodings['attention_mask']
        if self.to_alignment:
            return torch.Tensor(target_spans_input_ids).type(torch.LongTensor), \
                    torch.Tensor(target_spans_attention_mask).type(torch.LongTensor), \
                    torch.Tensor(observer_spans_input_ids).type(torch.LongTensor), \
                    torch.Tensor(observer_spans_attention_mask).type(torch.LongTensor), \
                    torch.Tensor(labels).type(torch.FloatTensor), \
                    torch.Tensor(ori_alignments).type(torch.IntTensor), \
                    torch.Tensor(text_mappings).type(torch.IntTensor)
        return torch.Tensor(target_spans_input_ids).type(torch.LongTensor), \
               torch.Tensor(target_spans_attention_mask).type(torch.LongTensor), \
               torch.Tensor(observer_spans_input_ids).type(torch.LongTensor), \
               torch.Tensor(observer_spans_attention_mask).type(torch.LongTensor), \
               torch.Tensor(labels).type(torch.FloatTensor)
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if self.to_alignment:
            return self.target_spans_input_ids[idx], \
               self.target_spans_attention_mask[idx], \
               self.observer_spans_input_ids[idx], \
               self.observer_spans_attention_mask[idx], \
               self.labels[idx], \
               self.ori_alignments[idx], \
               self.text_mappings[idx]
        return self.target_spans_input_ids[idx], \
               self.target_spans_attention_mask[idx], \
               self.observer_spans_input_ids[idx], \
               self.observer_spans_attention_mask[idx], \
               self.labels[idx]