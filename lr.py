from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from collections import defaultdict
from scipy.sparse import load_npz
from datetime import datetime
from eval_metrics import avgstd
import numpy as np
import pandas as pd
import os.path
import math
import glob
import time
import json
import sys
import yaml


SENSITIVE_ATTR = "school_id"

"""
df = pd.read_csv("data/assist09/preprocessed_data.csv",sep="\t")
df["weight"] = df.groupby(SENSITIVE_ATTR).user_id.transform('nunique')
#df["weight"] = df["weight"] / len(df["user_id"].unique())
df["weight"] = 1 / df["weight"]
"""

FULL = False
X_file = sys.argv[1] #options.X_file
folder = os.path.dirname(X_file)
y_file = X_file.replace('X', 'y').replace('npz', 'npy')

X = load_npz(X_file)
nb_samples, _ = X.shape
y = np.load(y_file).astype(np.int32)
print(X.shape, y.shape)

# Know number of users
"""
with open(os.path.join(folder, 'config.yml')) as f:
    config = yaml.load(f)
    X_users = X[:, :config['nb_users']]
    print(X_users.shape)
    assert all(X_users.sum(axis=1) == 1)
    # sys.exit(0)
"""

# Are folds fixed already?
X_trains = {}
weights_train = {}
y_trains = {}
X_tests = {}
y_tests = {}
FOLD = 'strong'
folds = glob.glob(os.path.join(folder, 'folds/50weak{}fold*.npy'.format(nb_samples)))
if folds:
    print(folds)
    for i, filename in enumerate(folds):
        i_test = np.load(filename)
        print('Fold', i, i_test.shape)
        i_train = list(set(range(nb_samples)) - set(i_test))
        X_trains[i] = X[i_train]
        y_trains[i] = y[i_train]
        X_tests[i] = X[i_test]
        y_tests[i] = y[i_test]
        #weights_train[i] = np.array(df["weight"])[i_train]
        #weights_test[i] = np.array(df["weight"])[i_test]
elif FULL:
    X_trains[0] = X
    X_tests[0] = X
    y_trains[0] = y
    y_tests[0] = y
else:
    print('No folds so train test split')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=False)
    X_trains[0] = X_train
    X_tests[0] = X_test
    y_trains[0] = y_train
    y_tests[0] = y_test
    

results = defaultdict(list)
predictions = []
for i in X_trains:
    X_train, X_test, y_train, y_test = (X_trains[i], X_tests[i],
                                        y_trains[i], y_tests[i])
    model = LogisticRegression(solver='liblinear')  # Has L2 regularization by default
    dt = time.time()

    # weights_train[i] should contain the same value as sample_weights

    """
    nb_samples = len(y_train)
    nb_users = config['nb_users']
    
    X_train_users = X_train[:, :config['nb_users']]
    nb_samples_per_user = X_train_users.sum(axis=0).A1
    nb_samples_per_user[nb_samples_per_user == 0] = 1
    print(X_train_users.shape)
    print(nb_samples_per_user.shape)
    print((X_train_users @ (1 / nb_samples_per_user)).shape)
    # sample_weights = np.ones(nb_samples)
    print(nb_samples / nb_users / nb_samples_per_user)
    sample_weights = X_train_users @ (nb_samples / nb_users / nb_samples_per_user)
    """
    
    model.fit(X_train, y_train)#, sample_weight=sample_weights)

    print('[time] Training', time.time() - dt, 's')

    for dataset, X, y in [('Train', X_train, y_train),
                          ('Test', X_test, y_test)]:
        dt = time.time()
        y_pred = model.predict_proba(X)[:, 1]

        # Store predictions of the fold
        if dataset == 'Test':
            predictions.append({
                'fold': i,
                'pred': y_pred.tolist(),
                'y': y.tolist()
            })

        if len(y_pred) < 10:
            print(dataset, 'predict:', y_pred)
            print(dataset, 'was:', y)
        try:  # This may fail if there are too few classes
            nll = log_loss(y, y_pred)
            auc = roc_auc_score(y, y_pred)
        except ValueError:
            nll = auc = -1

        metrics = {'ACC': np.mean(y == np.round(y_pred)),
                   'NLL': nll,
                   'AUC': auc}
        for metric, value in metrics.items():
            results['{} {}'.format(dataset, metric)].append(value)
            print(dataset, metric, 'on fold {}:'.format(i), value)
        print('[time]', time.time() - dt, 's')

    np.save(os.path.join(folder, 'coef{}.npy'.format(i)), model.coef_)

print('# Final results')
for metric in results:
    print('{}: {}'.format(metric, avgstd(results[metric])))

iso_date = datetime.now().isoformat()
with open(os.path.join(folder, 'results-{}.json'.format(iso_date)), 'w') as f:
    json.dump({
        'predictions': predictions,
        'model': 'LR',
        'folds': FOLD
    }, f)
