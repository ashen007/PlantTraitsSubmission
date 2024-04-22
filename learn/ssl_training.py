import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.self_supervision_tresnet import FeatureLearnModule
from dataloader.imageInpaintData import ImageDataset
from dataloader.transformers import INPAINT_TRANSFORMER
from torchmetrics.regression import R2Score, MeanAbsoluteError
from train import Compile

df_train = pd.read_csv('../data/train.csv')
df_valid = pd.read_csv('../data/valid.csv')

train_dataloader = DataLoader(ImageDataset(df_train, INPAINT_TRANSFORMER),
                              shuffle=True, batch_size=64, drop_last=True)
oob_valid_dataloader = DataLoader(ImageDataset(df_valid, INPAINT_TRANSFORMER),
                                  shuffle=True, batch_size=64, drop_last=True)

if __name__ == '__main__':
    model = FeatureLearnModule()

    learner = Compile(model,
                      torch.nn.L1Loss,
                      torch.optim.AdamW,
                      3.75e-5,
                      1e-4,
                      25,
                      64,
                      train_dataloader,
                      f'./ckpts/best_ckpt_ssl_224.pth',
                      oob_valid_dataloader,
                      None)

    learner.fit()
