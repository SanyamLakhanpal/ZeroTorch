import nltk
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter('ignore')


# Model
class BERT_BASE_UNCASED(nn.Module):
    def __init__(self):
        super(BERT_BASE_UNCASED, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('../input/bert-base-uncased')
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.drop(output)
        output = self.fc(output)
        output = self.out(output)
        return output


class BERT_BASE_CASED(nn.Module):
    def __init__(self):
        super(BERT_BASE_CASED, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('../input/bert-base-cased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.drop(output)
        output = self.out(output)
        return output