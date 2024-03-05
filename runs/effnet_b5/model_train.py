import torch
from torch import nn
from torchmetrics.regression import R2Score, MeanAbsoluteError
from train import Compile
from models.effnet_b1 import CustomEffnetVitB1
from make_dataloaders import get_train_dataloader, get_val_dataloader

train_dataloader = get_train_dataloader(32)
val_dataloader = get_val_dataloader(32)

# load model
model = CustomEffnetVitB1()
# state = torch.load('./best_checkpoint.pth')
# model.load_state_dict(state['model_state_dict'])

complied = Compile(model,
                   nn.SmoothL1Loss,
                   torch.optim.AdamW,
                   1e-4,
                   1e-2,
                   50,
                   32,
                   train_dataloader,
                   val_dataloader,
                   metrics={'r2': R2Score(6).cuda(),
                            'mae': MeanAbsoluteError().cuda()}
                   )

if __name__ == '__main__':
    complied.fit()
