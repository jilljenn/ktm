from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
import glob
import os.path


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
    kfold = KFold(nb_folds, shuffle=True)
    for i, (train, test) in enumerate(kfold.split(full)):
        np.save('folds/weakest{}fold{}.npy'.format(nb_samples, i), test)


def load_folds(folder, options=None, df=None):
    nb_samples = len(df)
    valid_folds = None
    if options.test:
        test_folds = [options.test]
    else:
        test_folds = sorted(glob.glob(os.path.join(folder, 'folds/60weak{}fold*.npy'.format(nb_samples))))
        valid_folds = sorted(glob.glob(os.path.join(folder, 'folds/36weak{}valid*.npy'.format(nb_samples))))
    if not test_folds:
        print('No folds')
        test_folds, valid_folds = save_folds(df)
        # Or, for example, weak generalization:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        #                                                     shuffle=False)
    if not valid_folds:
        print('No valid_folds')
        if test_folds:
            valid_folds = test_folds
    return test_folds, valid_folds
