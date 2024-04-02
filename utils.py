import os
import re
import numpy as np
from datetime import datetime
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, roc_auc_score

def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, save_folder_path):
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    save_path = f'{save_folder_path}/checkpoint{str(epoch)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            }, save_path)

def prepare_saving(saving_folder, model_name, checkpoint_save_path, output_spans_path, include_time=True):
    now = datetime.now()
    path = saving_folder
    if model_name != None:
        path = f'{path}/{model_name}'
    if include_time:
        path = f'{path}_{now.strftime("%Y-%m-%d_%H-%M-%S")}'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(f'{path}/{checkpoint_save_path}') and checkpoint_save_path != None:
        os.makedirs(f'{path}/{checkpoint_save_path}')
    if not os.path.exists(f'{path}/{output_spans_path}') and output_spans_path != None:
        os.makedirs(f'{path}/{output_spans_path}')
    return now, path


# predict from logits
def predict(outputs, **kwargs):
    if kwargs['process_data'] == 'feed_multi_sent':
        sigmoid = nn.Sigmoid()
        return sigmoid(outputs) >= kwargs['pred_threshold']
    elif kwargs['process_data'] == 'feed_alignments':
        return (outputs > kwargs["pred_threshold"]).reshape(-1)
    else:
        return torch.argmax(outputs, dim=1)
        
# check if should early stop the training 
# if the last {cutoff} losses are always increasing
def check_early_stopping(lowest_loss_epoch, current_epoch, cutoff):
    return (current_epoch-lowest_loss_epoch) > cutoff


''' reconstruction from prediction '''
# pair the output predictions with tokens
def map_tokens2spans(predictions, offset_mapping):
    N, num_tokens = predictions.shape
    spans = []
    for i in range(N):
        span = []
        for nt in range(num_tokens):
            if offset_mapping[i, nt].sum() == 0: # special tokens or padding
                continue
            span.append([offset_mapping[i, nt, 0].item(), offset_mapping[i, nt, 1].item(), predictions[i, nt].item()])
        spans.append(span)            
    return spans

# merge consecutive spans with the same label
def merge_spans(texts, spans, id2labels):
    N = len(spans)
    output = []
    for i in range(N):
        span_start = -1
        span_end = -1
        span_label = None
        merged_spans = []
        for span in spans[i]:
            if span_start == -1:
                span_start, span_end, span_label = span
            else:
                # check if the next token share the same label and is consecutive or split by space and punctuations
                if span[2] == span_label and span[0] == span_end or \
                    span[2] == span_label and re.match(r'^[ _.,!?"\'/$]+', texts[i][span_end: span[0]]):
                    span_end = span[1]
                else:
                    merged_spans.append([span_start, span_end, id2labels[span_label]])
                    span_start, span_end, span_label = span  
        merged_spans.append([span_start, span_end, id2labels[span_label]])
        merged_spans = [span for span in merged_spans if span[2] != 'No Label']
        output.append({
            'text': texts[i],
            'spans': merged_spans
        })           
    return output


def combine_sents(texts, mappings, predictions, id2labels, **kwargs):
    output = []
    mappings_tmp = mappings.numpy()
    ordered_indices = np.lexsort((mappings_tmp[:,1], mappings_tmp[:,0]))
    ordered_indices = torch.LongTensor(ordered_indices) # find the order of the sentences
    ordered_mappings = mappings[ordered_indices]
    ordered_texts = np.array(texts)[ordered_indices].tolist()
    ordered_predictions = predictions[ordered_indices]
    target_prefix = 'target:\n\n'
    observer_prefix = '\n\nobserver:\n\n'
    prev_text = -1
    text = None
    spans = None
    for i, order in enumerate(ordered_mappings):
        if order[0] != prev_text: # a new text
            if prev_text != -1:
                output.append({
                    'seq_id': prev_text.item(),
                    'text': text,
                    'spans': spans
                }) 
            text = ""
            spans = []
            prev_text = order[0]
        elif len(text) + len(observer_prefix) == order[2]: # observer part
            text += observer_prefix
        if kwargs['process_data'] == 'feed_single_sent':
            if id2labels[ordered_predictions[i].item()] != 'No Label':
                spans.append([len(text), len(text) + len(ordered_texts[i]), id2labels[ordered_predictions[i].item()]])
        elif kwargs['process_data'] == 'feed_multi_sent' or kwargs['process_data'] == 'multilabel_only':
            if ordered_predictions[i].sum() != 0:
                spans.append([len(text), len(text) + len(ordered_texts[i]), [id2labels[idx] for idx, pred in enumerate(ordered_predictions[i]) if pred]])
        text += ordered_texts[i]
    output.append({
        'seq_id': prev_text.item(),
        'text': text,
        'spans': spans
    }) # append the last text
    return output


# define different losses
'''
only cross entropy
'''
def simple_cross_entropy_loss(outputs, labels, **kwargs):
    loss = nn.CrossEntropyLoss()
    return loss(outputs, labels)


'''
binary cross entropy loss, can pass weights
'''
def binary_cross_entropy_loss(outputs, labels, **kwargs):
    loss = nn.BCEWithLogitsLoss(pos_weight=kwargs["pos_weight"])
    return loss(outputs, labels)

'''
cross entropy + penalize specified labels
'''
def penalize_label_loss(outputs, labels, **kwargs):
    # sanity check
    assert len(kwargs['penalize_labels']) == len(kwargs['lmbda'])
    labels_mask = labels.ne(-100)
    predictions = torch.argmax(outputs, dim=1)
    selected_predictions = torch.masked_select(predictions, labels_mask)
    selected_labels = torch.masked_select(labels, labels_mask)
    
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(outputs, labels)
    for penalized_label, lmbda in zip(kwargs['penalize_labels'], kwargs['lmbda']):
        if lmbda > 0:
            loss += lmbda* ((selected_predictions==penalized_label).sum())/torch.numel(selected_labels)
        elif lmbda < 0:
            loss -= lmbda* (torch.numel(selected_labels)-(selected_predictions==penalized_label).sum())/torch.numel(selected_labels)
    return loss

'''
cross entropy + encourage longer spans
'''
def penalize_length_loss(outputs, labels, **kwargs):
    lmbda = kwargs['lmbda']
    
    predictions = torch.argmax(outputs, dim=1)
    next_token = torch.full(predictions.shape, -100).to(kwargs['device'])
    next_token[:,:-1] = predictions[:, 1:]

    num_tokens = labels.ne(-100).sum(1)
    num_spans = (next_token == predictions).sum(1)

    span_loss = (num_spans / num_tokens).sum() / outputs.shape[0]

    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(outputs, labels) + lmbda * span_loss
    return loss


'''
loss for alignment
contrastive loss and classification loss
'''
def contrastive_loss(embed1, embed2, out, label, **kwargs):
    ce_loss = nn.BCEWithLogitsLoss(pos_weight=kwargs["pos_weight"])
    classification_loss = ce_loss(out, label)
    dist = torch.nn.functional.pairwise_distance(embed1, embed2)
    contrastive_loss = (1-label)*torch.pow(dist, 2) \
                        + label*torch.pow(torch.clamp(kwargs["m"]-dist, min=0.0), 2)
    contrastive_loss = torch.mean(contrastive_loss)
    loss = kwargs["lmbda"]*contrastive_loss + (1-kwargs["lmbda"])*classification_loss
    loss = loss / label.shape[0]
    return loss

'''
loss for alignment
'''
def mse_loss(outputs, labels, **kwargs):
    loss = nn.MSELoss(reduction="mean")
    return loss(outputs, labels)


'''
For multilabel classification: BCEWithLogitsLoss (Sigmoid+BCE)
'''
def bce_with_logits_loss(outputs, labels, **kwargs):
    loss = nn.BCEWithLogitsLoss()
    return loss(outputs.float(), labels.float())


# define different evaluation metrics
def get_metrics(labels, predictions, id2labels, **kwargs):
    if kwargs['process_data'] == 'feed_token':
        labels_mask = labels.ne(-100)
        selected_predictions = torch.masked_select(predictions, labels_mask)
        selected_labels = torch.masked_select(labels, labels_mask)
    else:
        selected_predictions = predictions
        selected_labels = labels
    metric_rst = {}
    if 'total_accuracy' in kwargs['metrics']:
        metric_rst['total_accuracy'] = get_total_accuracy(selected_labels, selected_predictions)
    if 'total_recall' in kwargs['metrics']:
        metric_rst['total_recall'] = get_total_recall(selected_labels, selected_predictions)
    if 'total_precision' in kwargs['metrics']:
        metric_rst['total_precision'] = get_total_precision(selected_labels, selected_predictions)
    if 'macro_recall' in kwargs['metrics']:
        metric_rst['macro_recall'] = get_macro_recall(selected_labels, selected_predictions)
    if 'macro_precision' in kwargs['metrics']:
        metric_rst['macro_precision'] = get_macro_precision(selected_labels, selected_predictions)
    if 'macro_f1' in kwargs['metrics']:
        metric_rst['macro_f1'] = get_macro_f1(selected_labels, selected_predictions)
    if 'f1_score' in kwargs['metrics']:
        metric_rst['f1_score'] = get_f1(selected_labels, selected_predictions)
    if 'auroc_score' in kwargs['metrics']:
        metric_rst['auroc'] = get_auroc(selected_labels, selected_predictions)
    if 'per_label_recall' in kwargs['metrics']:
        metric_rst['per_label_recall'] = get_per_label_recall(selected_labels, selected_predictions, id2labels)
    if 'per_label_precision' in kwargs['metrics']:
        metric_rst['per_label_precision'] = get_per_label_precision(selected_labels, selected_predictions, id2labels)
    return metric_rst

def get_total_accuracy(labels, predictions):
    # labels_mask = labels.ne(-100)
    # num_total = labels_mask.sum()
    # num_correct = torch.masked_select(predictions == labels, labels_mask).sum()
    # return round((num_correct/num_total).item(), 5)
    accuracy = accuracy_score(labels.cpu(), predictions.cpu())
    return round(accuracy, 5)

def get_total_recall(labels, predictions):
    recall = recall_score(labels.cpu(), predictions.cpu())
    return round(recall, 5)

def get_total_precision(labels, predictions):
    precision = precision_score(labels.cpu(), predictions.cpu())
    return round(precision, 5)

def get_macro_recall(labels, predictions):
    recall = recall_score(labels.cpu(), predictions.cpu(), average='macro')
    return round(recall, 5)

def get_macro_precision(labels, predictions):
    precision = precision_score(labels.cpu(), predictions.cpu(), average='macro')
    return round(precision, 5)

def get_macro_f1(labels, predictions):
    f1 = f1_score(labels.cpu(), predictions.cpu(), average='macro')
    return round(f1,5)

def get_f1(labels, predictions):
    f1 = f1_score(labels.cpu(), predictions.cpu())
    return round(f1,5)

def get_auroc(labels, predictions):
    auroc = roc_auc_score(labels.cpu(), predictions.cpu())
    return round(auroc,5)

def get_per_label_recall(labels, predictions, id2labels):
    # metric_dict = {}
    # ids = id2labels.keys()
    # for i in ids:
    #     # if i == 0: continue # ignore No Label
    #     labels_mask = labels.eq(i)
    #     num_total = labels_mask.sum()
    #     num_correct = torch.masked_select(predictions, labels_mask).eq(i).sum()
    #     metric_dict[id2labels[i]] = round((num_correct/num_total).item(), 5)
    # return metric_dict
    metric_dict = {}
    recalls = recall_score(labels.cpu(),predictions.cpu(), average=None)
    for idx, score in enumerate(recalls):
        metric_dict[id2labels[idx]] = round(score,5)
    return metric_dict

def get_per_label_precision(labels, predictions, id2labels):
    # metric_dict = {}
    # ids = id2labels.keys()
    # selected_predictions = torch.masked_select(predictions, labels.ne(-100))
    # for i in ids:
    #     # if i == 0: continue # ignore No Label
    #     labels_mask = labels.eq(i)
    #     num_total = selected_predictions.eq(i).sum()
    #     num_correct = torch.masked_select(predictions, labels_mask).eq(i).sum()
    #     metric_dict[id2labels[i]] = round((num_correct/num_total).item(), 5)
    # return metric_dict
    metric_dict = {}
    precisions = precision_score(labels.cpu(),predictions.cpu(), average=None)
    for idx, score in enumerate(precisions):
        metric_dict[id2labels[idx]] = round(score,5)
    return metric_dict


def get_template(filename):
    # load in prompt template
    with open(filename) as f:
        prompt = f.readlines()
    # clear all comments
    prompt = [p for p in prompt if not (len(p)>=8 and p[:4]=='<---' and p[-5:]=='--->\n')]
    prompt = ''.join(prompt)
    return prompt

def get_promptdataloader(dataset, template, tokenizer, WrapperClass, args, shuffle):
    from openprompt import PromptDataLoader
    return PromptDataLoader(dataset=dataset, 
                            template=template, 
                            tokenizer=tokenizer,
                            tokenizer_wrapper_class=WrapperClass,
                            max_seq_length=args.max_seq_length,
                            decoder_max_length=args.decoder_max_length,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            teacher_forcing=args.teacher_forcing,
                            predict_eos_token=args.predict_eos_token,
                            truncate_method=args.truncate_method)