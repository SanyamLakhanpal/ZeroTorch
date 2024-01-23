import wandb
import torch
import tqdm
from PIL import Image

wandb.init(

    project="Celeba_experiments",
    name="Try_7",
    group="group_1",
    job_type="train",
    config=vars(args)
)

# Standard logging
wandb.log({"train_loss": train_loss, "joint_loss": joint_loss, "image_loss": image_loss, "attr_loss": attrs_loss})
wandb.log({"test_loss": test_loss})
wandb.log({f"{prefix}_joint_recon_image": image_bce.sum()})


# Wandb table logging with image

preview_table = wandb.Table(columns=['Id', 'Image', 'Subject Focus', 'Eyes', 'Face',
                                     'Near', 'Action', 'Accessory', 'Group', 'Collage',
                                     'Human', 'Occlusion', 'Info', 'Blur', 'Pawpularity'])

# Adding image in the table.
# Do not add TENSORS in this.
for i in tqdm(range(len(tmp_df))):
    row = tmp_df.loc[i]
    img = Image.open(row.file_path)
    preview_table.add_data(row['Id'],
                           wandb.Image(img),
                           row['Subject Focus'],
                           row['Eyes'],
                           row['Face'],
                           row['Near'],
                           row['Action'],
                           row['Accessory'],
                           row['Group'],
                           row['Collage'],
                           row['Human'],
                           row['Occlusion'],
                           row['Info'],
                           row['Blur'],
                           row['Pawpularity'])

wandb.log({'Visualization': preview_table})



wandb.finish()

# Wandb with config file

CONFIG = dict(
    seed = 42,
    model_name = 'tf_efficientnet_b4_ns',
    train_batch_size = 16,
    valid_batch_size = 32,
    img_size = 512,
    epochs = 5,
    learning_rate = 1e-4,
    scheduler = 'CosineAnnealingLR',
    min_lr = 1e-6,
    T_max = 100,
    T_0 = 25,
    warmup_epochs = 0,
    weight_decay = 1e-6,
    n_accumulate = 1,
    n_fold = 5,
    num_classes = 1,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    competition = 'PetFinder',
    _wandb_kernel = 'deb'
)



run = wandb.init(project='Pawpularity',
                 config=CONFIG,
                 job_type='Visualization',
                 group='Public_baseline',
                 anonymous='must')


# For Kaggle Usage

import wandb

try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("wandb_api")
    wandb.login(key=api_key)
    anony = None
except:
    anony = "must"
    print(
        'If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')



