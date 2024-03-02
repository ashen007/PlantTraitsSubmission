import torch
from torch import nn
from torchmetrics.regression import R2Score, MeanAbsoluteError
from train import Compile
from loss import AltR2Loss
from models.effnet_mm import CustomEffnet
from make_dataloaders import get_train_dataloader, get_val_dataloader

train_dataloader = get_train_dataloader(32)
val_dataloader = get_val_dataloader(32)

complied = Compile(CustomEffnet(),
                   nn.MSELoss,
                   torch.optim.AdamW,
                   3e-4,
                   1e-2,
                   25,
                   32,
                   train_dataloader,
                   val_dataloader,
                   metrics={'r2': R2Score(6).cuda(),
                            'mae': MeanAbsoluteError().cuda()}
                   )

if __name__ == '__main__':
    complied.fit()
