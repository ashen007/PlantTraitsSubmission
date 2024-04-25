import numpy as np
import torch
import pandas as pd
import tqdm

from move import move_to
from models.convnext_small import ConvNext
from models.seresnet import SeResNet
from models.swin import SwinTrans
from dataloader.testdata import PlantDataset
from dataloader.transformers import TEST_TIME_TRANSFORMER

if __name__ == '__main__':
    tar_features = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
    log_features = ['X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']

    # load model
    model = SwinTrans()
    state = torch.load('ckpts/ignore/best_ckpt_256_swin_with_box.pth')
    model.load_state_dict(state['model_state_dict'])

    df_test = pd.read_csv('../data/test.csv')
    # pipe = joblib.load('../data/scaler.joblib')
    # df_valid = pd.read_csv('../data/valid.csv')
    test_dataset = PlantDataset(df_test, TEST_TIME_TRANSFORMER)
    preds_folds = []

    model.eval()
    model.cuda()

    for i in range(5):
        preds = []

        for x, _ in tqdm.tqdm(test_dataset, desc='Prediction:: '):
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

        preds_folds.append(pd.DataFrame(preds).values)

    col_names = [c.replace('_mean', '') for c in tar_features]
    preds = pd.DataFrame(np.asarray(preds_folds).mean(axis=0), columns=col_names)

    # restore to original scale
    preds.to_csv('./submission.csv', index=False)
