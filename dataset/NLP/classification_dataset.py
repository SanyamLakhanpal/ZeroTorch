import os
import gc
import platform
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import torch
import transformers
from transformers import AutoTokenizer, AutoModel, XLNetForSequenceClassification

import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter('ignore')


class Config:
    SCALER = GradScaler()
    NB_EPOCHS = 6
    LR = 3e-4
    MAX_LEN = 200
    TRAIN_BS = 16
    VALID_BS = 16
    _wandb_kernel = 'tanaym'
    MODEL_NAME = 'xlnet-large-cased'
    TOKENIZER = AutoTokenizer.from_pretrained('xlnet-large-cased')


# Convert the Config class to a dict for logging
config_dict = dict(vars(Config))
del [config_dict['__module__']]
del [config_dict['__dict__']]
del [config_dict['__weakref__']]
del [config_dict['__doc__']]


class CommonLitDataset(Dataset):
    def __init__(self, texts, targets=None, is_test=False):

        self.texts = texts
        if not is_test:
            self.content, self.wording = targets
        self.is_test = is_test
        self.tokenizer = Config.TOKENIZER
        self.max_len = Config.MAX_LEN
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        if self.is_test:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
            }
        else:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': self.targets
            }

