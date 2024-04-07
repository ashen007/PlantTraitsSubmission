import torch
import cv2
import numpy as np
import pandas as pd
import joblib
import imageio.v3 as imageio

from torch import nn
from tqdm import tqdm
from move import move_to
from torchmetrics.regression import R2Score, MeanAbsoluteError
from train import Compile
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from dataloader.transformers import TRANSFORMER, TEST_TRANSFORMER
from models.effnet import CustomEffnet

SEED = 2024
folds = KFold(n_splits=3, shuffle=True, random_state=SEED)
data = pd.read_csv('../data/2024/processed/train.csv')
test_preds = []


class TestDataset(Dataset):
    def __init__(self, X_jpeg_bytes, x_features, y):
        self.X_jpeg_bytes = X_jpeg_bytes
        self.y = y
        self.transforms = TEST_TRANSFORMER
        self.xs_cols = x_features.columns
        self.boxes = pd.read_csv('../data/2024/boxes_test.csv', index_col='id')

        self.boxes['box'] = self.boxes['box'].apply(
            lambda x: np.fromstring(x.replace('\n', '')
                                    .replace('[', '')
                                    .replace(']', '')
                                    .replace('  ', ' '), sep=' ')
        )

    def __len__(self):
        return len(self.X_jpeg_bytes)

    def __getitem__(self, index):
        try:
            box = self.boxes.loc[self.y[index], 'box']
            X_sample = self.transforms(
                image=imageio.imread(self.X_jpeg_bytes[index])[int(box[1]):int(box[3]), int(box[0]):int(box[2])])[
                'image']
        except:
            X_sample = self.transforms(
                image=imageio.imread(self.X_jpeg_bytes[index]))['image']

        y_sample = self.y[index]

        return move_to(X_sample, 'cuda').unsqueeze(0), y_sample


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


def predict_test(checkpoint):
    tar_features = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    log_features = ['X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']

    # load model
    model = CustomEffnet()
    state = torch.load(checkpoint)
    model.load_state_dict(state['model_state_dict'])

    df = pd.read_pickle('../data/2024/processed/test.pkl')
    pipe = joblib.load('../data/2024/processed/scaler.joblib')
    test_dataset = TestDataset(df['jpeg_bytes'].values, df, df['id'].values)
    preds = []

    model.eval()
    model.cuda()

    for x, idx in tqdm(test_dataset):
        with torch.no_grad():
            y = model(x).detach().cpu().numpy()

        logits = pipe.inverse_transform(y).squeeze()
        logits = logits[:6]

        row = {'id': idx}

        for k, v in zip(tar_features, logits):

            if k in log_features:
                row[k.replace('_mean', '')] = 10 ** v

            else:
                row[k.replace('_mean', '')] = v

        preds.append(row)

    preds = pd.DataFrame(preds)

    return preds


if __name__ == '__main__':
    for f_id, (t_idx, v_idx) in enumerate(folds.split(data)):
        train_data = data.iloc[t_idx, :].reset_index(drop=True)
        val_data = data.iloc[v_idx, :].reset_index(drop=True)
        pred_val_fold = []

        train_dataset = PlantDataset(train_data, TRANSFORMER)
        valid_dataset = PlantDataset(val_data, TEST_TRANSFORMER)

        train_dataloader = DataLoader(train_dataset, 16, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(valid_dataset, 16, shuffle=False, drop_last=False)

        model = CustomEffnet()

        complied = Compile(model,
                           nn.MSELoss,
                           torch.optim.AdamW,
                           1e-5,
                           1e-4,
                           13,
                           16,
                           train_loader=train_dataloader,
                           save_to=f'best_checkpoint_f_{f_id}.pth',
                           val_loader=val_dataloader,
                           metrics={'r2': R2Score(6).cuda(),
                                    'mae': MeanAbsoluteError().cuda()})

        complied.fit()
        # model.eval()
        #
        # for x, _ in tqdm(valid_dataset):
        #     with torch.no_grad():
        #         y = model(x.cuda().unsqueeze(0)).detach().cpu().numpy()
        #
        #     pred_val_fold.append(y.squeeze(0))
        #
        # val_data.iloc[:, 1:] = (0.7 * val_data.iloc[:, 1:] + 0.3 * np.asarray(pred_val_fold))

        # predicted_test = predict_test(f'./best_checkpoint_i{_}_f{f_id}.pth')
        # val_data.to_csv(f'./folds/test_preds_f_{f_id}.csv')
