import torch
from torch import nn
from torchmetrics.regression import R2Score, MeanAbsoluteError
from train import Compile
from models.swin import SwinTransformer
from models.effnet import CustomEffnet
from models.effnet23 import CustomEffnetFullY
from make_dataloaders import get_train_dataloader, get_val_dataloader
from make_dataloaders import get_23_train_dataloader, get_23_val_dataloader

train_dataloader = get_23_train_dataloader(16, full_y=True)
val_dataloader = get_23_val_dataloader(16, full_y=True)

# train_dataloader = get_train_dataloader(16)
# val_dataloader = get_val_dataloader(16)

# load model
model = CustomEffnetFullY()
# model = SwinTransformer()
# state = torch.load('./step_1/best_checkpoint.pth')
# model.load_state_dict(state['model_state_dict'])

complied = Compile(model,
                   nn.MSELoss,
                   torch.optim.AdamW,
                   1.5e-5,
                   1e-4,
                   10,
                   16,
                   train_dataloader,
                   val_dataloader,
                   metrics={'r2': R2Score(30).cuda(),
                            'mae': MeanAbsoluteError().cuda()}
                   )

if __name__ == '__main__':
    complied.fit()
