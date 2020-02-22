from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
import os.path


def save_folds(full, nb_folds=5):
    nb_samples = len(full)
    all_users = full['user_id'].unique()
    random.shuffle(all_users)
    fold_size = len(all_users) // nb_folds
    everything = []
    for i in range(nb_folds):
        if i < nb_folds - 1:
            ids_of_fold = set(all_users[i * fold_size:(i + 1) * fold_size])
        else:
            ids_of_fold = set(all_users[i * fold_size:])
        fold = full.query('user_id in @ids_of_fold').index
        np.save('folds/{}fold{}.npy'.format(nb_samples, i), fold)
        everything += list(fold)
    assert sorted(everything) == list(range(nb_samples))


def save_weak_folds(full, nb_folds=5):
    nb_samples = len(full)
    all_samples = range(nb_samples)
    kfold = KFold(nb_folds, shuffle=True)
    for i, (train, test) in enumerate(kfold.split(full)):
        np.save('folds/weak{}fold{}.npy'.format(nb_samples, i), test)
