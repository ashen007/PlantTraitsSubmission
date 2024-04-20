import os

import numpy as np
import torch
import pandas as pd
import tqdm
import joblib

from move import move_to
# from models.convnext_small import ConvNext
# from models.seresnet import SeResNet
from models.swin import SwinTrans
from dataloader.testdata import PlantDataset
# from dataloader.PlantTriatData import PlantDataset
from dataloader.transformers import TEST_TRANSFORMER
from torchmetrics.regression import R2Score

if __name__ == '__main__':
    tar_features = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
    log_features = ['X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']

    # load model
    model = SwinTrans()
    state = torch.load('ckpts/best_ckpt_384_swin.pth')
    model.load_state_dict(state['model_state_dict'])

    df_test = pd.read_csv('../data/test.csv')
    pipe = joblib.load('../data/scaler.joblib')

    # df_valid = pd.read_csv('../data/valid.csv')
    test_dataset = PlantDataset(df_test, TEST_TRANSFORMER)
    preds = []

    model.eval()
    model.cuda()

    for x, _ in tqdm.tqdm(test_dataset):
        x = move_to(x, 'cuda')

        with torch.no_grad():
            y_ = model(x).detach().cpu().numpy()

        # logits = pipe.inverse_transform(y_).squeeze()
        row = dict()  # {'id': idx}

        for k, v in zip(tar_features, y_[0]):
            # if k in log_features:
            #     row[k.replace('_mean', '')] = np.exp(v)
            #
            # else:
            #     row[k.replace('_mean', '')] = v

            row[k.replace('_mean', '')] = v

        preds.append(row)

    preds = pd.DataFrame(preds)

    # restore to original scale
    preds.to_csv('./submission_swin.csv', index=False)
