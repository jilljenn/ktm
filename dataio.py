from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
import os.path


VALID = 0.36  # Valid is 40% of train (= 24%), so starts from 36%
TEST = 0.6    # Test is 40%, so from 60%


def save_folds(full, nb_folds=5):
    if 'timestamp' not in full.columns:
        full['timestamp'] = np.zeros(len(full))
    nb_samples = len(full)
    all_users = full['user_id'].unique()
    random.shuffle(all_users)
    fold_size = len(all_users) // nb_folds
    everything = []
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
        np.save('folds/{}weak{}valid{}.npy'.format(round(100 * VALID), nb_samples, i), valid_fold)
        np.save('folds/{}weak{}fold{}.npy'.format(round(100 * TEST), nb_samples, i), test_fold)
    assert sorted(everything) == list(range(nb_samples))


def save_weak_folds(full, nb_folds=5):
    nb_samples = len(full)
    all_samples = range(nb_samples)
    kfold = KFold(nb_folds, shuffle=True)
    for i, (train, test) in enumerate(kfold.split(full)):
        np.save('folds/weakest{}fold{}.npy'.format(nb_samples, i), test)
