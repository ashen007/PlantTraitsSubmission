import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.regression import R2Score
from train import Compile
from dataloader.transformers import TRANSFORMER, TEST_TRANSFORMER
from dataloader.PlantTriatData import PlantDataset
from models.convnext_small import ConvNext
# from models.effnet import EffNet
from sklearn.model_selection import KFold
from move import move_to

folds = KFold(shuffle=True, random_state=48)

df_train = pd.read_csv('../data/train.csv')
df_valid = pd.read_csv('../data/valid.csv')

train_dataloader = DataLoader(PlantDataset(df_train, TRANSFORMER), shuffle=True, batch_size=8, drop_last=True)
oob_valid_dataloader = DataLoader(PlantDataset(df_valid, TEST_TRANSFORMER), shuffle=True, batch_size=8, drop_last=True)


def k_fold_train(folds, model):
    for f_id, (t_ids, v_ids) in enumerate(folds.split(df_train)):
        train_set, valid_set = df_train.loc[t_ids, :], df_train.loc[v_ids, :]
        train_dataloader = DataLoader(PlantDataset(train_set.reset_index(drop=True), TRANSFORMER),
                                      shuffle=True, batch_size=16, drop_last=True)
        valid_dataloader = DataLoader(PlantDataset(valid_set.reset_index(drop=True), TEST_TRANSFORMER),
                                      shuffle=True, batch_size=16, drop_last=True)

        print(f'fold:: {f_id}')

        model = model()
        # state = torch.load(f'./folds/best_ckpt_128_{f_id}.pth')
        # model.load_state_dict(state['model_state_dict'])

        learner = Compile(model,
                          torch.nn.MSELoss,
                          torch.optim.AdamW,
                          1.5e-5,
                          1e-3,
                          5,
                          16,
                          train_dataloader,
                          f'./folds/best_ckpt_256_{f_id}.pth',
                          valid_dataloader,
                          {'r2': R2Score(6).cuda()})

        learner.fit()

        # reload best ckpt
        state = torch.load(f'./folds/best_ckpt_256_{f_id}.pth')
        model.load_state_dict(state['model_state_dict'])

        # metrics
        metric_r2 = R2Score(6).cuda()
        metric_r2.reset()

        # oob evaluation
        for X, Y in tqdm.tqdm(oob_valid_dataloader, desc='oob validation:: '):
            X = move_to(X, 'cuda')
            Y = move_to(Y, 'cuda')

            with torch.no_grad():
                y_pred = model(X)

            metric_r2.update(y_pred, Y)

        print(f'oob evaluation R2:: {metric_r2.compute().item()}')


def full_batch_train(model):
    model = model()
    state = torch.load(f'./folds/best_ckpt_380.pth')
    model.load_state_dict(state['model_state_dict'])

    learner = Compile(model,
                      torch.nn.SmoothL1Loss,
                      torch.optim.AdamW,
                      3.75e-5,
                      1e-4,
                      3,
                      8,
                      train_dataloader,
                      f'./folds/best_ckpt_512.pth',
                      oob_valid_dataloader,
                      {'r2': R2Score(6).cuda()})

    learner.fit()


if __name__ == '__main__':
    full_batch_train(ConvNext)
