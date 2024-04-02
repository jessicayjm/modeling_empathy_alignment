import torch
from torch import nn
from transformers import BertModel, RobertaModel, AutoModel

class BERTTokenClassification(nn.Module):

    def __init__(self, bert_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name, num_labels=num_labels)
        self.hidden_size = self.bert.config.to_dict()['hidden_size']
        self.linear = nn.Linear(self.hidden_size, num_labels)
    
    def forward(self, input_ids,att_mask=None):
        out = self.bert(input_ids, att_mask)
        out = out.last_hidden_state
        out = self.linear(out)
        out = out.transpose(1,2)
        return out


class BERTSentenceClassification(nn.Module):

    def __init__(self, bert_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name, num_labels=num_labels)
        self.hidden_size = self.bert.config.to_dict()['hidden_size']
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.hidden_size, num_labels)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, input_ids,att_mask=None):
        out = self.bert(input_ids, att_mask)
        out = out.pooler_output
        out = self.dropout(out)
        out = self.linear(out)
        out = self.leaky_relu(out)
        return out


class BERTMultilabelClassication(nn.Module):
    
    def __init__(self, bert_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name, num_labels=num_labels)
        self.hidden_size = self.bert.config.to_dict()['hidden_size']
        self.linear = nn.Linear(self.hidden_size, num_labels)
    
    def forward(self, input_ids,att_mask=None):
        out = self.bert(input_ids, att_mask)
        out = out.pooler_output
        out = self.linear(out)
        return out

class RoBERTaTokenClassification(nn.Module):

    def __init__(self, bert_name, num_labels):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(bert_name, num_labels=num_labels)
        self.hidden_size = self.bert.config.to_dict()['hidden_size']
        self.linear = nn.Linear(self.hidden_size, num_labels)
        #self.softmax = nn.Softmax(2)
    
    def forward(self, input_ids,att_mask=None):
        #print('input ids', input_ids.shape)
        out = self.bert(input_ids, att_mask)
        out = out.last_hidden_state
        print('last hidden state', out.shape)
        out = self.linear(out)
        #print('after linear', out.shape)
        #out = self.softmax(out)
        out = out.transpose(1,2)
        #print('after softmax', out.shape)
        return out


class RoBERTaSentenceClassification(nn.Module):

    def __init__(self, bert_name, num_labels):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(bert_name, num_labels=num_labels)
        self.hidden_size = self.bert.config.to_dict()['hidden_size']
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.hidden_size, num_labels)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, input_ids,att_mask=None):
        out = self.bert(input_ids, att_mask)
        out = out.pooler_output
        out = self.dropout(out)
        out = self.linear(out)
        out = self.leaky_relu(out)
        return out


class RoBERTaMultilabelClassication(nn.Module):
    
    def __init__(self, bert_name, num_labels):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(bert_name, num_labels=num_labels)
        self.hidden_size = self.bert.config.to_dict()['hidden_size']
        self.linear = nn.Linear(self.hidden_size, num_labels)
    
    def forward(self, input_ids,att_mask=None):
        out = self.bert(input_ids, att_mask)
        out = out.pooler_output
        out = self.linear(out)
        return out


class MeanPooling(nn.Module):
    
    def __init__(self):
        super(MeanPooling, self).__init__()  
    
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class DeBERTaSentenceClassification(nn.Module):

    def __init__(self, bert_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name, num_labels=num_labels)
        self.hidden_size = self.bert.config.to_dict()['hidden_size']
        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.hidden_size, num_labels)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, input_ids,att_mask=None):
        out = self.bert(input_ids, att_mask)
        out = self.pooler(out.last_hidden_state, att_mask)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.leaky_relu(out)
        return out

class MiniLMSentenceClassification(nn.Module):

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, num_labels=num_labels)
        self.hidden_size = self.model.config.to_dict()['hidden_size']
        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.hidden_size, num_labels)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, input_ids,att_mask=None):
        out = self.model(input_ids, att_mask)
        out = out.pooler_output
        out = self.dropout(out)
        out = self.linear(out)
        out = self.leaky_relu(out)
        return out
