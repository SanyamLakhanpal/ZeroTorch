import os
import gc
import sys
import random
import platform

import numpy as np
import pandas as pd
from rich import progress
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import mean_squared_error
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import wandb
import warnings
warnings.simplefilter('ignore')


def wandb_log(**kwargs):
    for k, v in kwargs.items():
        wandb.log({k: v})


def MCRMSE(y_trues, y_preds):
    """
    Credits to Y. Nakama for this function:
    https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train?scriptVersionId=104639699&cellId=10
    """
    y_trues = np.asarray(y_trues)
    y_preds = np.asarray(y_preds)
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        scores.append(rmse)
    mcrmse_score = np.mean(scores)
    return mcrmse_score


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)


Config = {
    'TRAIN_BS': 16,
    'VALID_BS': 16,
    'MODEL_NAME': 'roberta-large',
    'TOKENIZER': transformers.AutoTokenizer.from_pretrained('roberta-large', use_fast=True),
    'NUM_WORKERS': 8,
    'scaler': GradScaler(),
    'FILE_PATH': '../input/feedback-prize-english-language-learning/train.csv',
    'LOSS': 'SmoothL1Loss',
    'EVAL_METRIC': 'MCRMSE',
    'NB_EPOCHS': 5,
    'SPLITS': 5,
    'T_0': 20,
    'η_min': 1e-4,
    'fc_dropout': 0.2,
    'betas': (0.9, 0.999),
    'MAX_LEN': 200,
    'N_LABELS': 6,
    'LR': 2e-4,
    'competition': 'feedback_3',
    '_wandb_kernel': 'tanaym',
}
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wb_key = user_secrets.get_secret("WANDB_API_KEY")

wandb.login(key=wb_key)


# Start W&B logging
# W&B Login
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wb_key = user_secrets.get_secret("WANDB_API_KEY")

wandb.login(key=wb_key)

run = wandb.init(
    project='pytorch',
    config=Config,
    group='nlp',
    job_type='train',
)


class FeedBackDataset(Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['full_text'].values
        labels = self.data[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].values
        inputs = self._tokenize_texts(text[idx])
        if not self.is_test:
            targets = torch.tensor(labels[idx], dtype=torch.float)
            return inputs, targets
        return inputs

    def _tokenize_texts(self, text):
        inputs = Config['TOKENIZER'].encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=Config['MAX_LEN'],
            pad_to_max_length=True,
            truncation=True
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs


class FeedBackModel(nn.Module):
    def __init__(self):
        super(FeedBackModel, self).__init__()
        self.backbone = transformers.AutoModel.from_pretrained(Config['MODEL_NAME'])
        self.drop = nn.Dropout(0.3)

        if "large" in Config['MODEL_NAME']:
            self.fc = nn.Linear(1024, Config['N_LABELS'])
        elif "base" in Config['MODEL_NAME']:
            self.fc = nn.Linear(768, Config['N_LABELS'])

    def forward(self, input_dict):
        _, output = self.backbone(**input_dict, return_dict=False)
        output = self.drop(output)
        output = self.fc(output)
        return output

def yield_optimizer(model):
    """
    Returns optimizer for specific parameters
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.003,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return transformers.AdamW(optimizer_parameters, lr=Config['LR'])


class Trainer:
    def __init__(self, config, dataloaders, optimizer, model, loss_fns, scheduler, device="cuda:0"):
        self.train_loader, self.valid_loader = dataloaders
        self.train_loss_fn, self.valid_loss_fn = loss_fns
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.device = torch.device(device)
        self.config = config

    def train_one_epoch(self):
        """
        Trains the model for 1 epoch
        """
        self.model.train()
        train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        train_preds, train_targets = [], []

        for bnum, (inputs, targets) in train_pbar:
            for k, v in inputs.items():
                inputs[k] = self._convert_if_not_tensor(v, dtype="infer")

            targets = self._convert_if_not_tensor(targets, dtype=torch.float)

            with autocast(enabled=True):
                outputs = self.model(inputs)

                loss = self.train_loss_fn(outputs, targets)
                loss_itm = loss.item()

                wandb_log(
                    train_batch_loss=loss_itm
                )

                train_pbar.set_description('loss: {:.2f}'.format(loss_itm))

                Config['scaler'].scale(loss).backward()
                Config['scaler'].step(self.optimizer)
                Config['scaler'].update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            train_targets.extend(targets.cpu().detach().numpy().tolist())
            train_preds.extend(outputs.cpu().detach().numpy().tolist())

        # Tidy
        del outputs, targets, inputs, loss_itm, loss
        gc.collect()
        torch.cuda.empty_cache()

        return train_preds, train_targets

    @torch.no_grad()
    def valid_one_epoch(self):
        """
        Validates the model for 1 epoch
        """
        self.model.eval()
        valid_pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        valid_preds, valid_targets = [], []

        for idx, (inputs, targets) in valid_pbar:
            for k, v in inputs.items():
                inputs[k] = self._convert_if_not_tensor(v, dtype="infer")

            targets = self._convert_if_not_tensor(targets, dtype=torch.float)

            outputs = self.model(inputs)
            valid_loss = self.valid_loss_fn(outputs, targets)

            wandb_log(
                valid_batch_loss=valid_loss.item()
            )

            valid_pbar.set_description(desc=f"val_loss: {valid_loss.item():.4f}")

            valid_targets.extend(targets.cpu().detach().numpy().tolist())
            valid_preds.extend(outputs.cpu().detach().numpy().tolist())

        # Tidy
        del outputs, inputs, targets, valid_loss
        gc.collect()
        torch.cuda.empty_cache()

        return valid_preds, valid_targets

    def fit(self, epochs: int = 10, output_dir: str = "/kaggle/working/", custom_name: str = 'model.pth'):
        """
        Low-effort alternative for doing the complete training and validation process
        """
        best_loss = int(1e+7)
        best_preds = None
        for epx in range(epochs):
            print(f"{'=' * 20} Epoch: {epx + 1} / {epochs} {'=' * 20}")

            train_preds, train_targets = self.train_one_epoch()
            train_mcrmse = MCRMSE(train_targets, train_preds)
            print(f"Training MCRMSE: {train_mcrmse:.4f}")

            valid_preds, valid_targets = self.valid_one_epoch()
            valid_mcrmse = MCRMSE(valid_targets, valid_preds)
            print(f"Validation MCRMSE: {valid_mcrmse:.4f}")

            wandb_log(
                train_mcrmse=train_mcrmse,
                valid_mcrmse=valid_mcrmse
            )

            if valid_mcrmse < best_loss:
                best_loss = valid_mcrmse
                self.save_model(output_dir, custom_name)
                print(f"Saved model with validation MCRMSE: {best_loss:.4f}")

    def save_model(self, path, name, verbose=False):
        """
        Saves the model at the provided destination
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except:
            print("Errors encountered while making the output directory")

        torch.save(self.model.state_dict(), os.path.join(path, name))
        if verbose:
            print(f"Model Saved at: {os.path.join(path, name)}")

    def _convert_if_not_tensor(self, x, dtype):
        if dtype == "infer":
            dtype = x.dtype
        if self._tensor_check(x):
            return x.to(self.device, dtype=dtype)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)

    def _tensor_check(self, x):
        return isinstance(x, torch.Tensor)


# Training Code
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda:0')
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = torch.device('cpu')

    data = pd.read_csv(Config['FILE_PATH'])
    data = data.sample(frac=1).reset_index(drop=True)
    text = data[["full_text"]]
    labels = data[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]]
    per_fold_predictions = {}

    # Do Multilabel Stratified KFolds training and cross validation
    kf = MultilabelStratifiedKFold(n_splits=Config['SPLITS'], shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=text, y=labels.values)):
        # Train for only 1 fold, you can train it for more.
        #         if fold != 0:
        #             continue
        print(f"\n{'=' * 40} Fold: {fold} {'=' * 40}")

        train_data = data.loc[train_idx]
        valid_data = data.loc[valid_idx]

        train_set = FeedBackDataset(train_data)
        valid_set = FeedBackDataset(valid_data)

        train_loader = DataLoader(
            train_set,
            batch_size=Config['TRAIN_BS'],
            shuffle=True,
            num_workers=8
        )

        valid_loader = DataLoader(
            valid_set,
            batch_size=Config['VALID_BS'],
            shuffle=False,
            num_workers=8
        )

        model = FeedBackModel().to(DEVICE)
        nb_train_steps = int(len(train_data) / Config['TRAIN_BS'] * Config['NB_EPOCHS'])
        optimizer = yield_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=Config['T_0'],
            eta_min=Config['η_min']
        )
        train_loss_fn, valid_loss_fn = nn.SmoothL1Loss(), nn.SmoothL1Loss()

        wandb.watch(model, criterion=train_loss_fn)

        trainer = Trainer(
            config=Config,
            dataloaders=(train_loader, valid_loader),
            loss_fns=(train_loss_fn, valid_loss_fn),
            optimizer=optimizer,
            model=model,
            scheduler=scheduler,
        )

        best_pred = trainer.fit(
            epochs=Config['NB_EPOCHS'],
            custom_name=f"{Config['MODEL_NAME']}_fold_{fold}.bin"
        )
        model.cpu()

        per_fold_predictions[f"fold_{fold}"] = best_pred

        del best_pred, trainer, train_loss_fn, valid_loss_fn, model, optimizer, scheduler
        del train_data, valid_data, train_set, valid_set, train_loader, valid_loader, train_idx, valid_idx
        gc.collect()
        torch.cuda.empty_cache()

    best_predictions = pd.DataFrame()
    best_predictions['folds'] = list(per_fold_predictions.keys())
    best_predictions['predictions'] = list(per_fold_predictions.values())
    best_predictions.head()
    # Finish the logging run
    run.finish()