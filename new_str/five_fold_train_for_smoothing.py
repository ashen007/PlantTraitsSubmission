import torch
import cv2
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torchmetrics.regression import R2Score, MeanAbsoluteError
from train import Compile
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from dataloader.transformers import TRANSFORMER, TEST_TRANSFORMER
from models.effnet import CustomEffnet

SEED = 2024
folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
data = pd.read_csv('../data/2024/processed/train.csv')


class PlantDataset(Dataset):

    def __init__(self, df, transformer=None):
        self.df = df
        self.columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
        self.dir = '../data/2024/train_images/'
        self.df['box'] = self.df['box'].apply(lambda x: np.fromstring(x.replace('\n', '')
                                                                      .replace('[', '')
                                                                      .replace(']', '')
                                                                      .replace('  ', ' '), sep=' '))
        self.boxes = self.df.pop('box')
        self.transform = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError

        img_id = self.df.loc[idx, 'id']
        y = torch.tensor(self.df.loc[idx, self.columns].values, dtype=torch.float32)

        img = cv2.imread(f'{self.dir}/{img_id}.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        box = self.boxes.loc[idx]
        img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, y


for f_id, (t_idx, v_idx) in enumerate(folds.split(data)):
    train_data = data.iloc[t_idx, :].reset_index(drop=True)
    val_data = data.iloc[v_idx, :].reset_index(drop=True)
    pred_val_fold = []

    train_dataset = PlantDataset(train_data, TRANSFORMER)
    valid_dataset = PlantDataset(val_data, TEST_TRANSFORMER)

    train_dataloader = DataLoader(train_dataset, 12, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(valid_dataset, 12, shuffle=False, drop_last=False)

    model = CustomEffnet()
    # state = torch.load(f'./best_checkpoint_f_{f_id}.pth')
    # model.load_state_dict(state['model_state_dict'])
    # model.cuda()

    complied = Compile(model,
                       nn.MSELoss,
                       torch.optim.AdamW,
                       1e-5,
                       1e-4,
                       5,
                       12,
                       train_loader=train_dataloader,
                       save_to=f'best_checkpoint_f_{f_id}.pth',
                       val_loader=val_dataloader,
                       metrics={'r2': R2Score(6).cuda(),
                                'mae': MeanAbsoluteError().cuda()})

    complied.fit()
    model.eval()

    for x, _ in tqdm(valid_dataset):
        with torch.no_grad():
            y = model(x.cuda().unsqueeze(0)).detach().cpu().numpy()

        pred_val_fold.append(y.squeeze(0))

    pred_val_fold = pd.DataFrame(np.asarray(pred_val_fold), columns=valid_dataset.columns).set_index(val_data['id'])
    pred_val_fold.to_csv(f'./folds/fold_{f_id}.csv')
