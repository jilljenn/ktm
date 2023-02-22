"""
If options.folds is 'weak', the k-fold is performed on samples.
Otherwise, it is performed on users.
"""
from pathlib import Path
import re
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


def get_paths(options, model_name):
    X_file = Path(options.X_file)
    folder = X_file.parent
    m = re.match(r'X-(.*).npz', X_file.name)
    suffix = m.group(1)
    y_file = folder / f'y-{suffix}.npy'
    y_pred_file = folder / f'y-{suffix}-{model_name}-pred.csv'
    df = pd.read_csv(folder / 'data.csv')
    return df, X_file, folder, y_file, y_pred_file


def save_folds(full, nb_folds=5):
    """
    Note: can be improved with validation sets besides the test sets
    """
    if 'timestamp' not in full.columns:
        full['timestamp'] = np.zeros(len(full))
    all_users = full['user'].unique()
    folds = []
    kfold = KFold(nb_folds, shuffle=True, random_state=42)
    for train_users, test_users in kfold.split(all_users):
        folds.append((
            full.query('user in @train_users').sort_values(
                'timestamp').index.to_numpy(),
            full.query('user in @test_users').sort_values(
                'timestamp').index.to_numpy()
        ))
    return folds


def save_weak_folds(full, nb_folds=5):
    kfold = KFold(nb_folds, shuffle=True, random_state=42)
    return kfold.split(full)


def load_folds(options=None, df=None):
    """
    Either folds are specified in the CSV file or they are generated
    according to whether options.folds is 'weak' or not.
    Returns pairs of arrays of train/test indices.
    """
    if df is not None and 'fold' in df.columns:
        i_train = df.query("fold != 'test'").index.to_numpy()
        i_test = df.query("fold == 'test'").index.to_numpy()
        return [(i_train, i_test)]
    print('No folds specified in CSV file')

    if options.folds == 'weak':
        return save_weak_folds(df)
    return save_folds(df)
