import torch
from torch import nn
from torchmetrics.regression import R2Score, MeanAbsoluteError
from train import Compile
from models.effnet import CustomEffnet
from models.effnet_mm import CustomEffnetMM
from make_dataloaders import get_train_dataloader, get_val_dataloader
from make_dataloaders import get_23_train_dataloader, get_23_val_dataloader

train_dataloader = get_23_train_dataloader(32)
val_dataloader = get_23_val_dataloader(32)

# train_dataloader = get_train_dataloader(32)
# val_dataloader = get_val_dataloader(32)

# load model
model = CustomEffnet()
state = torch.load('./step_5/best_checkpoint.pth')
model.load_state_dict(state['model_state_dict'])

complied = Compile(model,
                   nn.MSELoss,
                   torch.optim.AdamW,
                   3e-5,
                   1e-4,
                   20,
                   32,
                   train_dataloader,
                   val_dataloader,
                   metrics={'r2': R2Score(6).cuda(),
                            'mae': MeanAbsoluteError().cuda()}
                   )

if __name__ == '__main__':
    complied.fit()
