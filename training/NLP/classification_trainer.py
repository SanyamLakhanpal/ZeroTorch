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



def yield_loss(outputs, targets):
    """
    This is the loss function for this task
    """
    return torch.sqrt(nn.MSELoss()(outputs, targets))


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

        for bnum, inputs in train_pbar:
            ids = inputs['ids'].to(self.device, dtype=torch.long)
            mask = inputs['mask'].to(self.device, dtype=torch.long)
            ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
            content_tar = inputs['targets'][0].to(self.device, dtype=torch.float)
            wording_tar = inputs['targets'][1].to(self.device, dtype=torch.float)
            targets = torch.hstack((content_tar.view(-1, 1), wording_tar.view(-1, 1)))

            with autocast(enabled=True):
                outputs = self.model(input_ids=ids, attention_mask=mask, token_type_ids=ttis).logits

                loss = self.train_loss_fn(outputs, targets)
                loss_itm = loss.item()

                wandb_log(
                    train_batch_loss=loss_itm
                )

                train_pbar.set_description('loss: {:.2f}'.format(loss_itm))

                Config.SCALER.scale(loss).backward()
                Config.SCALER.step(self.optimizer)
                Config.SCALER.update()
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

        for bnum, inputs in valid_pbar:
            ids = inputs['ids'].to(self.device, dtype=torch.long)
            mask = inputs['mask'].to(self.device, dtype=torch.long)
            ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
            content_tar = inputs['targets'][0].to(self.device, dtype=torch.float)
            wording_tar = inputs['targets'][1].to(self.device, dtype=torch.float)
            targets = torch.hstack((content_tar.view(-1, 1), wording_tar.view(-1, 1)))

            outputs = self.model(input_ids=ids, attention_mask=mask, token_type_ids=ttis).logits
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
            train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
            print(f"Training RMSE: {train_rmse:.4f}")

            valid_preds, valid_targets = self.valid_one_epoch()
            valid_rmse = np.sqrt(mean_squared_error(valid_targets, valid_preds))
            print(f"Validation RMSE: {valid_rmse:.4f}")

            wandb_log(
                train_rmse=train_rmse,
                valid_rmse=valid_rmse
            )

            if valid_rmse < best_loss:
                best_loss = valid_rmse
                self.save_model(output_dir, custom_name)
                print(f"Saved model with validation RMSE: {best_loss:.4f}")

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
    return transformers.AdamW(optimizer_parameters, lr=Config.LR)
# To train model

# Training Code
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda:0')
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = torch.device('cpu')

    summaries_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv")
    prompts_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv")

    data = prompts_df.merge(summaries_df, on='prompt_id')
    data['text'] = data['prompt_question'] + ' ' + data['text']
    del summaries_df, prompts_df

    train_data = data.sample(frac=0.9).reset_index(drop=True)
    valid_data = data.sample(frac=0.1).reset_index(drop=True)

    train_set = CommonLitDataset(
        texts=train_data['text'].values,
        targets=(train_data['content'].values, train_data['wording'].values)
    )

    valid_set = CommonLitDataset(
        texts=valid_data['text'].values,
        targets=(valid_data['content'].values, valid_data['wording'].values)
    )

    train = DataLoader(
        train_set,
        batch_size=Config.TRAIN_BS,
        shuffle=True,
        num_workers=8
    )

    valid = DataLoader(
        valid_set,
        batch_size=Config.VALID_BS,
        shuffle=False,
        num_workers=8
    )

    model = XLNetForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=2).to(DEVICE)
    wandb.watch(model)
    nb_train_steps = int(len(train_data) / Config.TRAIN_BS * Config.NB_EPOCHS)
    optimizer = yield_optimizer(model)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=3,
        num_training_steps=nb_train_steps
    )

    trainer = Trainer(
        config=Config,
        dataloaders=(train, valid),
        optimizer=optimizer,
        model=model,
        loss_fns=(yield_loss, yield_loss),
        scheduler=scheduler,
        device='cuda:0'
    )
    trainer.fit(epochs=Config.NB_EPOCHS)