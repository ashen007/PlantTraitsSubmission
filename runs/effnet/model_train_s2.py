import torch
from torch import nn
from torchmetrics.regression import R2Score, MeanAbsoluteError
from train import Compile
from models.effnet_s2 import CustomEffnetS2
from models.effnet_mm import CustomEffnetMM
from make_dataloaders import get_23_train_dataloader, get_23_val_dataloader

train_dataloader = get_23_train_dataloader(32)
val_dataloader = get_23_val_dataloader(32)

# load model
model = CustomEffnetS2()
state = torch.load('./best_checkpoint.pth')
model.load_state_dict(state['model_state_dict'])

complied = Compile(model,
                   nn.MSELoss,
                   torch.optim.AdamW,
                   1.5e-5,
                   1e-4,
                   10,
                   32,
                   train_dataloader,
                   val_dataloader,
                   metrics={'r2': R2Score(30).cuda(),
                            'mae': MeanAbsoluteError().cuda()}
                   )

if __name__ == '__main__':
    complied.fit()
