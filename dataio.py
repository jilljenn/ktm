from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
import os.path


def save_folds(full, nb_folds=5, weakness=0.2):
    nb_samples = len(full)
    all_users = full['user_id'].unique()
    random.shuffle(all_users)
    fold_size = len(all_users) // nb_folds
    everything = []
    for i in range(nb_folds):
        upper_bound = (i + 1) * fold_size if i < nb_folds - 1 else len(all_users)
        ids_of_fold = set(all_users[i * fold_size:upper_bound])
        fold = full.query('user_id in @ids_of_fold').index
        everything += list(fold)
        n_samples = len(fold)
        fold = fold[round(weakness * n_samples):]
        np.save('folds/{}{}fold{}.npy'.format(
            'weak' if weakness > 0 else '', nb_samples, i), fold)
    assert sorted(everything) == list(range(nb_samples))


def save_weak_folds(full, nb_folds=5):
    nb_samples = len(full)
    all_samples = range(nb_samples)
    kfold = KFold(nb_folds, shuffle=True)
    for i, (train, test) in enumerate(kfold.split(full)):
        np.save('folds/weakest{}fold{}.npy'.format(nb_samples, i), test)
