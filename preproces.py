import os.path
import joblib
import numpy as np
import pandas as pd
import tqdm

from feature_engine.transformation import LogCpTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

ROOT = 'data/'
DUMP = 'data/processed'
SEED = 48


def pre_process(path):
    df = pd.read_csv(os.path.join(path, 'train_clean.csv'))
    df = df.fillna(0)

    drop_files = set([f'{i}.jpeg' for i in (df['id']).values.tolist()]).difference(
        set(os.listdir('./data/train_images')))

    drop_ids = [int(i.split('.')[0]) for i in drop_files]
    df = df.set_index('id')
    features = df.columns[:-12]
    targets = df.columns[-12:-6]
    df = df.drop(drop_ids, axis=0).drop(df.columns[-6:], axis=1)

    # pre process
    pipe_target = Pipeline([('log_trans', LogCpTransformer(base='10', C=1e-5)),
                            ('min_max_scale', MinMaxScaler())])
    df.loc[:, targets] = pipe_target.fit_transform(np.asarray(df.loc[:, targets]))
    df = df.loc[:, targets]

    # pipe_feat = Pipeline([('log_trans', LogCpTransformer(base='10', C=1e-5)),
    #                       ('min_max_scale', MinMaxScaler())])
    # df.loc[:, features] = pipe_feat.fit_transform(np.asarray(df.loc[:, features]))

    # dump
    df.to_csv(os.path.join(DUMP, 'train.csv'))
    joblib.dump(pipe_target, './data/target_pipe.joblib')


if __name__ == '__main__':
    pre_process(ROOT)
