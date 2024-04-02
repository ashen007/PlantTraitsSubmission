import os

import torch
import pandas as pd
import tqdm
import albumentations as A
import cv2
import joblib

from models.swin import SwinTransformer
from models.effnet import CustomEffnet
from models.effnet23 import CustomEffnetFullY
from dataloader.testdata import TestDataset

if __name__ == '__main__':
    tar_features = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    log_features = ['X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']

    # load model
    model = CustomEffnetFullY()
    state = torch.load('step_5/best_checkpoint.pth')
    model.load_state_dict(state['model_state_dict'])

    df = pd.read_pickle('../../data/test.pkl')
    # pipe = joblib.load('../../data/processed/scaler.joblib')
    pipe = joblib.load('../../data/2023/processed/scaler_23.joblib')
    test_dataset = TestDataset(df['jpeg_bytes'].values, df, df['id'].values)
    preds = []

    model.eval()
    model.cuda()

    for x, idx in tqdm.tqdm(test_dataset):
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

    # restore to original scale
    preds.to_csv('./subs/submission_5.csv', index=False)
