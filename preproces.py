import os.path
import joblib
import numpy as np
import pandas as pd
import tqdm

from feature_engine.transformation import LogTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = 'data/'
DUMP = 'data/processed'
SEED = 48


def pre_process(path):
    df = pd.read_csv(os.path.join(path, 'train.csv'), usecols=['id', 'X4_mean', 'X11_mean', 'X18_mean',
                                                               'X26_mean', 'X50_mean', 'X3112_mean'])
    df = np.abs(df)

    drop_files = set([f'{i}.jpeg' for i in (df['id']).values.tolist()]).difference(
        set(os.listdir('./data/train_images')))

    drop_ids = [int(i.split('.')[0]) for i in drop_files]
    df = df.set_index('id')
    df = df.drop(drop_ids, axis=0)  # .drop(df.columns[-6:], axis=1)

    # pre process
    pipe_target = Pipeline([('log_trans', LogTransformer(base='10')),
                            ('min_max_scale', StandardScaler())
                            ])
    df[df.columns] = pipe_target.fit_transform(np.asarray(df))

    preds = pd.read_csv('archive/runs/v3/increase_input/submission_1.csv', index_col='id')
    preds = pd.DataFrame(pipe_target.inverse_transform(np.asarray(preds))).set_index(preds.index)

    # dump
    # df.to_csv(os.path.join(DUMP, 'train.csv'))
    preds.to_csv('runs/v3/increase_input/submission.csv')
    # joblib.dump(pipe_target, './data/target_pipe.joblib')


if __name__ == '__main__':
    print()
    # pre_process(ROOT)
