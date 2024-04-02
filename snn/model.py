import torch
from torch import nn
from transformers import AutoModel

# class MeanPooling(nn.Module):
    
#     def __init__(self):
#         super(MeanPooling, self).__init__()  
    
#     def forward(self, last_hidden_state, attention_mask):
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = input_mask_expanded.sum(1)
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
#         return mean_embeddings

# class SNN(nn.Module):

#     def __init__(self, model_name):
#         super().__init__()
#         self.model_name = model_name
#         self.embedding_model = AutoModel.from_pretrained(model_name)
#         self.hidden_size = self.embedding_model.config.to_dict()['hidden_size']
#         # self.pooler = MeanPooling()
#         self.fc = nn.Sequential(
#             nn.Linear(self.hidden_size * 2, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 1),
#         )
#         self.sigmoid = nn.Sigmoid()

#     def embed(self, x_input_ids, x_attention_mask):
#         out = self.embedding_model(x_input_ids, x_attention_mask)
#         # if 'deberta' in self.model_name:
#         #     out = self.pooler(out.last_hidden_state, x_attention_mask)
#         # else:
#         out = out[1]
#         return out
    
#     def forward(self, x1_input_ids, x1_attention_mask, x2_input_ids, x2_attention_mask):
#         embed1 = self.embed(x1_input_ids, x1_attention_mask)
#         embed2 = self.embed(x2_input_ids, x2_attention_mask)
#         out = torch.cat((embed1, embed2), 1)
#         out = self.fc(out)
#         out = self.sigmoid(out)
#         return embed1, embed2, out

class SNN(nn.Module): 
    def __init__(self, model_name):
        super(SNN,self).__init__()
        self.model = AutoModel.from_pretrained(model_name).to("cuda").train()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-4)
        
    def mean_pooling(self, token_embeddings, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids_a, attention_a, input_ids_b, attention_b): 
        #encode sentence and get mean pooled sentence representation 
        encoding1 = self.model(input_ids_a, attention_mask=attention_a)[0] #all token embeddings
        encoding2 = self.model(input_ids_b, attention_mask=attention_b)[0]
        
        meanPooled1 = self.mean_pooling(encoding1, attention_a)
        meanPooled2 = self.mean_pooling(encoding2, attention_b)
        
        pred = self.cos(meanPooled1, meanPooled2)
        return pred
