from pathlib import Path
import random
import re
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

"""
K-fold on users.
First K - 1 folds are fully for train.
On the K-th fold of test users, this represents for example 40% for test:

0 %      36 %     60 %       100 %
tr tr tr va va va te te te te
-------- -------- -----------
 TRAIN    VALID      TEST

"""
VALID = 0.36  # Valid is 40% of train (= 24%), so starts from 36%
TEST = 0.6    # Test is 40%, so from 60%


def get_paths(options):
    X_file = Path(options.X_file)
    folder = X_file.parent
    m = re.match(r'X-(.*).npz', X_file.name)
    suffix = m.group(1)
    y_file = folder / f'y-{suffix}.npy'
    y_pred_file = folder / f'y-{suffix}-pred.csv'
    df = pd.read_csv(folder / 'data.csv')
    return df, X_file, folder, y_file, y_pred_file


def save_folds(full, nb_folds=5):
    if 'timestamp' not in full.columns:
        full['timestamp'] = np.zeros(len(full))
    nb_samples = len(full)
    all_users = full['user_id'].unique()
    random.shuffle(all_users)
    fold_size = len(all_users) // nb_folds
    everything = []
    valid_folds = []
    test_folds = []
    for i in range(nb_folds):
        upper_bound = (i + 1) * fold_size if i < nb_folds - 1 else len(all_users)
        ids_of_fold = set(all_users[i * fold_size:upper_bound])
        test_fold = []
        valid_fold = []
        for user_id in ids_of_fold:
            fold = full.query('user_id == @user_id').sort_values('timestamp').index
            everything += list(fold)
            n_samples_user = len(fold)
            valid_fold.extend(fold[round(VALID * n_samples_user):round(TEST * n_samples_user)])
            test_fold.extend(fold[round(TEST * n_samples_user):])
        valid_filename = 'folds/{}weak{}valid{}.npy'.format(round(100 * VALID), nb_samples, i)
        test_filename = 'folds/{}weak{}fold{}.npy'.format(round(100 * TEST), nb_samples, i)
        if not os.path.exists('folds'):
            os.makedirs('folds')
        np.save(valid_filename, valid_fold)
        np.save(test_filename, test_fold)
        valid_folds.append(valid_filename)
        test_folds.append(test_filename)
    assert sorted(everything) == list(range(nb_samples))
    return test_folds, valid_folds


def save_weak_folds(full, nb_folds=5):
    nb_samples = len(full)
    all_samples = range(nb_samples)
    kfold = KFold(nb_folds, shuffle=True, random_state=42)
    return kfold.split(full)


def load_folds(folder, options=None, df=None):
    """
    Actually returns arrays of filenames, but it would be better to return arrays of indices
    """
    print(folder)
    if df is not None and 'fold' in df.columns:
        print(df.head())
        nb_samples = len(df)
        test_folds = df.query("fold == 'test'").index.to_numpy()
        return [test_folds], [test_folds]
    print('No folds')

    if options.folds == 'weak':
        return save_weak_folds(df)
    else:
        return save_folds(df)
