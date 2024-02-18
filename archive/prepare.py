import os.path

import joblib
import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

ROOT = 'data/'
DUMP = 'data/processed'
SEED = 48


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def train_val_data(path):
    df = pd.read_csv(os.path.join(path, 'train_clean.csv'))
    df = df.fillna(0)
    drop_files = set([f'{i}.jpeg' for i in (df['id']).values.tolist()]).difference(
        set(os.listdir('../data/train_images')))
    drop_ids = [int(i.split('.')[0]) for i in drop_files]
    df = df.set_index('id')
    df = df.drop(drop_ids, axis=0)

    # processed
    scaler = MinMaxScaler()
    df[df.columns[:-6]] = scaler.fit_transform(np.asarray(df[df.columns[:-6]]))
    # df_sq = df[df.columns[:-12]].apply(lambda x: x**2, axis=1)
    # df_sqrt = df[df.columns[:-12]].apply(lambda x: np.sqrt(x), axis=1)
    # df = pd.concat((df_sq, df_sqrt, df), axis=1)
    # df = df.dropna()

    # cleanup
    # train, val = train_test_split(df, test_size=0.2, random_state=SEED)

    # dump
    df.to_csv(os.path.join(DUMP, 'train.csv'))
    # val.to_csv(os.path.join(DUMP, 'val.csv'))
    joblib.dump(scaler, '../data/scaler.joblib')


if __name__ == '__main__':
    train_val_data(ROOT)
