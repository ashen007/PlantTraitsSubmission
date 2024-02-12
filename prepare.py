import os.path
import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer

ROOT = 'data/'
DUMP = 'data/processed'
SEED = 48


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def train_val_data(path):
    df = pd.read_csv(os.path.join(path, 'cleaned_train.csv'))
    df = df.fillna(0)
    drop_files = set([f'{i}.jpeg' for i in (df['id']).values.tolist()]).difference(
        set(os.listdir('data/train_images')))
    drop_ids = [int(i.split('.')[0]) for i in drop_files]
    df = df.set_index('id')
    df = df.drop(drop_ids, axis=0)

    # processed
    df[df.columns[1:-12]] = (df[df.columns[1:-12]] - df[df.columns[1:-12]].min()) / (
            df[df.columns[1:-12]].max() - df[df.columns[1:-12]].min())

    df_sin = sin_transformer(np.mean(df[df.columns[1:-12]])).fit_transform(df[df.columns[1:-12]])
    df_cos = cos_transformer(np.mean(df[df.columns[1:-12]])).fit_transform(df[df.columns[1:-12]])

    df = pd.concat((df[df.columns[0]], df_sin, df_cos, df[df.columns[1:]]), axis=1)

    # cleanup
    train, val = train_test_split(df, test_size=0.2, random_state=SEED)

    # dump
    train.to_csv(os.path.join(DUMP, 'train.csv'))
    val.to_csv(os.path.join(DUMP, 'val.csv'))


if __name__ == '__main__':
    train_val_data(ROOT)
