import torch
from torch import nn
from torchmetrics.regression import R2Score, MeanAbsoluteError
from train import Compile
from loss import AltR2Loss
from models.effnet import CustomEffnet
from make_dataloaders import get_train_dataloader, get_val_dataloader

train_dataloader = get_train_dataloader(32)
val_dataloader = get_val_dataloader(32)

# load model
model = CustomEffnet()
state = torch.load('./best_checkpoint.pth')
model.load_state_dict(state['model_state_dict'])

complied = Compile(model,
                   nn.MSELoss,
                   torch.optim.AdamW,
                   3e-4,
                   1e-2,
                   10,
                   32,
                   train_dataloader,
                   val_dataloader,
                   metrics={'r2': R2Score(6).cuda(),
                            'mae': MeanAbsoluteError().cuda()}
                   )

if __name__ == '__main__':
    complied.fit()
