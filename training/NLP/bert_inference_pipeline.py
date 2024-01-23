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

class Config:
    MAX_LEN = 284
    TRAIN_BS = 12
    STATE_DIR = "../input/training-kfolds-vanilla-pytorch-bert-starter"
    BERT_MODEL = 'bert-base-uncased'
    FILE_NAME = '../input/commonlitreadabilityprize/test.csv'
    TOKENIZER = transformers.BertTokenizer.from_pretrained('../input/bert-base-uncased', do_lower_case=True)
    scaler = GradScaler()


class BERTDataset(Dataset):
    def __init__(self, review, target=None, is_test=False):
        self.review = review
        self.target = target
        self.is_test = is_test
        self.tokenizer = Config.TOKENIZER
        self.max_len = Config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = str(self.review[idx])
        review = ' '.join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
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
            targets = torch.tensor(self.target[idx], dtype=torch.float)
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets
            }


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


@torch.no_grad()
def inference(model, states_list, test_dataloader, device=torch.device('cuda:0')):
    """
    Do inference for different model folds
    """
    model.eval()
    all_preds = []
    for state in states_list:
        print(f"State: {state}")
        state_dict = torch.load(state)
        model.load_state_dict(state_dict)
        model = model.to(device)

        # Clean
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        preds = []
        prog = tqdm(test_dataloader, total=len(test_dataloader))
        for data in prog:
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            ttis = data['token_type_ids'].to(DEVICE, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=ttis)
            preds.append(outputs.squeeze(-1).cpu().detach().numpy())

        all_preds.append(np.concatenate(preds))

        # Clean
        gc.collect()
        torch.cuda.empty_cache()

    return all_preds


# Inference Code
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("\n[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda:0')
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = torch.device('cpu')

    test_file = pd.read_csv(Config.FILE_NAME)

    test_data = BERTDataset(test_file['excerpt'].values, is_test=True)
    test_data = DataLoader(
        test_data,
        batch_size=Config.TRAIN_BS,
        shuffle=False
    )

    state_list = [os.path.join(Config.STATE_DIR, x) for x in os.listdir(Config.STATE_DIR) if x.endswith(".pt")]
    model = BERT_BASE_UNCASED()

    print("Doing Predictions for all folds")
    predictions = inference(model, state_list, test_data, device=DEVICE)

    final_predictions = pd.DataFrame(predictions).T.mean(axis=1).tolist()

    # Form the sample submission
    sub = pd.DataFrame()
    sub['id'] = test_file['id']
    sub['target'] = final_predictions

    sub.to_csv("submission.csv", index=None)
    sub.head()