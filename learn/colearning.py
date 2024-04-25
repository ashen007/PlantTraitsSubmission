import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataloader.CLData import CLDataset
from dataloader.transformers import TRANSFORMER, TEST_TRANSFORMER
from noisy_train import Compile
from torchmetrics.regression import R2Score

df_train = pd.read_csv('../data/train.csv')
df_valid = pd.read_csv('../data/valid.csv')

train_dataloader = DataLoader(CLDataset(df_train, TRANSFORMER), shuffle=True, batch_size=16, drop_last=True)
oob_valid_dataloader = DataLoader(CLDataset(df_valid, TEST_TRANSFORMER), shuffle=True, batch_size=16, drop_last=True)


def full_batch_train():
    learner = Compile(3.75e-5,
                      10,
                      16,
                      train_dataloader,
                      f'./ckpts/best_ckpt_colearn_effvit_288.pth',
                      oob_valid_dataloader,
                      {'r2': R2Score(6).cuda()})

    learner.fit()


if __name__ == '__main__':
    full_batch_train()
