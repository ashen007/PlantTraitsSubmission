import os.path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = 'data/'
DUMP = 'data/processed'
SEED = 48


def train_val_data(path):
    df = pd.read_csv(os.path.join(path, 'cleaned_train.csv'))
    df = df.fillna(0)
    # tar_mean = df.columns[-12:-6]
    # tar_sd = df.columns[-6:]

    # for m, s in zip(tar_mean, tar_sd):
    #     df[m] = df[m] + df[s]

    # processed
    # df[tar_mean] = (df[tar_mean] - df[tar_mean].min()) / (df[tar_mean].max() - df[tar_mean].min())

    # cleanup
    train, val = train_test_split(df, test_size=0.2, random_state=SEED)
    # train = train.drop(tar_sd, axis=1)
    # val = val.drop(tar_sd, axis=1)
    # train = train.rename(columns={col: col.replace('_mean', '') for col in tar_mean})
    # val = val.rename(columns={col: col.replace('_mean', '') for col in tar_mean})

    # dump
    train.to_csv(os.path.join(DUMP, 'train.csv'), index=False)
    val.to_csv(os.path.join(DUMP, 'val.csv'), index=False)


if __name__ == '__main__':
    train_val_data(ROOT)
