from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from collections import defaultdict
from scipy.sparse import load_npz
from scipy.stats import sem, t
import numpy as np
import os.path
import math
import glob
import time
import sys


def avgstd(l):
    '''
    Given a list of values, returns a 95% confidence interval
    if the standard deviation is unknown.
    '''
    n = len(l)
    mean = sum(l) / n
    if n == 1:
        return '%.3f' % round(mean, 3)
    std_err = sem(l)
    confidence = 0.95
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return '%.3f ± %.3f' % (round(mean, 3), round(h, 3))


FULL = False
X_file = sys.argv[1]
folder = os.path.dirname(X_file)
y_file = X_file.replace('X', 'y').replace('npz', 'npy')

X = load_npz(X_file)
nb_samples, _ = X.shape
y = np.load(y_file).astype(np.int32)
print(X.shape, y.shape)

# Are folds fixed already?
X_trains = {}
y_trains = {}
X_tests = {}
y_tests = {}
folds = glob.glob(os.path.join(folder, 'folds/{}fold*.npy'.format(nb_samples)))
if folds:
    for i, filename in enumerate(folds):
        i_test = np.load(filename)
        print('Fold', i, i_test.shape)
        i_train = list(set(range(nb_samples)) - set(i_test))
        X_trains[i] = X[i_train]
        y_trains[i] = y[i_train]
        X_tests[i] = X[i_test]
        y_tests[i] = y[i_test]
elif FULL:
    X_trains[0] = X
    X_tests[0] = X
    y_trains[0] = y
    y_tests[0] = y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=False)
    X_trains[0] = X_train
    X_tests[0] = X_test
    y_trains[0] = y_train
    y_tests[0] = y_test
    

results = defaultdict(list)
for i in X_trains:
    X_train, X_test, y_train, y_test = (X_trains[i], X_tests[i],
                                        y_trains[i], y_tests[i])
    model = LogisticRegression(solver='liblinear')  # Has L2 regularization by default
    dt = time.time()
    model.fit(X_train, y_train)
    print('[time] Training', time.time() - dt, 's')

    for dataset, X, y in [('Train', X_train, y_train),
                          ('Test', X_test, y_test)]:
        dt = time.time()
        y_pred = model.predict_proba(X)[:, 1]
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
