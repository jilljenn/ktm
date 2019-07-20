from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import load_npz
import numpy as np
import glob
import os.path
import sys


FULL = False
X_file = sys.argv[1]
folder = os.path.dirname(X_file)
y_file = os.path.join(folder, 'y.npy')

X = load_npz(X_file)
nb_samples, _ = X.shape
y = np.load(y_file).astype(np.int32)

# Are folds fixed already?
X_trains = {}
y_trains = {}
X_tests = {}
y_tests = {}
if os.path.isfile(os.path.join(folder, 'fold0.npy')):
    for i, filename in enumerate(glob.glob(os.path.join(folder, 'fold*.npy'))):
        i_test = np.load(filename)
        print('Fold', i, i_test.shape)
        i_train = list(set(range(nb_samples)) - set(i_test))
        X_trains[i] = X[i_train]
        y_trains[i] = y[i_train]
        X_tests[i] = X[i_test]
        y_tests[i] = y[i_test]


if X_trains:
    print('Yepee', len(X_trains))
    X_train, X_test, y_train, y_test = (X_trains[0], X_tests[0],
                                        y_trains[0], y_tests[0])
    print(X_train.shape, X_test.shape)
elif FULL:
    X_train, X_test, y_train, y_test = X, X, y, y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=False)


model = LogisticRegression()  # Has L2 regularization by default
model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)[:, 1]
if len(y_pred_train) < 10:
    print('Train predict:', y_pred_train)
    print('Train was:', y_train)
print('Train ACC:', np.mean(y_train == np.round(y_pred_train)))
print('Train AUC:', roc_auc_score(y_train, y_pred_train))

y_pred_test = model.predict_proba(X_test)[:, 1]
if len(y_pred_test) < 10:
    print('Test predict:', y_pred_test)
    print('Test was:', y_test)
print('Test ACC:', np.mean(y_test == np.round(y_pred_test)))
try:
    print('Test AUC:', roc_auc_score(y_test, y_pred_test))
except ValueError:
    pass

np.save(os.path.join(folder, 'coef.npy'), model.coef_)
