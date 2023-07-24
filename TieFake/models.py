import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from transformers import BertForSequenceClassification
import numpy as np
from attention import MultiHeadAttention

class FusionModel(Module):

    def __init__(self, resnest_model, bert_model):
        super(FusionModel, self).__init__()

        assert isinstance(bert_model, BertForSequenceClassification)

        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        resnest_feature_size = resnest_model.fc.in_features
        self._resnest = resnest_model
        self._resnest.fc = nn.Linear(resnest_feature_size, 30)
        for param in self._resnest.parameters():
            param.requires_grad = False

        self._bert = bert_model.bert
        self._bert.eval()
        for param in self._bert.parameters():
            param.requires_grad = False
        self._attention_T_to_Ti = MultiHeadAttention(32,32,32,1)
        self._attention_Ti_to_T = MultiHeadAttention(32,32,32,1)
        self.linear = nn.Linear(32+30+38+32+32, 1)

    def forward(self, batch_in:dict):

        batch_in = {x: batch_in[x].to(next(self.parameters()).device) for x in batch_in}
        bert_output = self._bert(batch_in['bert_input_id_title'].squeeze(1), attention_mask=batch_in['bert_attention_mask_title'].squeeze(1))
        title_vector = bert_output.pooler_output
        bert_output = self._bert(batch_in['bert_input_id_title'].squeeze(1), attention_mask=batch_in['bert_attention_mask_title'].squeeze(1))
        text_vector = bert_output.pooler_output
        emo_feature = np.float32(batch_in['bert_input_emo'].cpu())
        emo_feature = torch.from_numpy(emo_feature).squeeze(1)
        emo_feature=emo_feature.to(self.device)
        resnest_feature = self._resnet(batch_in['image'])
        att_T_Ti,_=self._attention_T_to_Ti(text_vector,title_vector)
        att_Ti_T,_=self._attention_Ti_to_T(title_vector,text_vector)
        return self.linear(torch.cat((text_vector,emo_feature,resnest_feature,att_T_Ti,att_Ti_T), dim=1))