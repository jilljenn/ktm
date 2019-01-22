from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from scipy.sparse import load_npz, vstack
import pywFM
import argparse
import numpy as np
import os
import sys


# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = os.path.join(os.path.dirname(__file__),
                                        'libfm/bin/')

parser = argparse.ArgumentParser(description='Run FM')
parser.add_argument('X_file', type=str, nargs='?')
parser.add_argument('--iter', type=int, nargs='?', default=200)
parser.add_argument('--d', type=int, nargs='?', default=20)
parser.add_argument('--subset', type=int, nargs='?', default=0)
options = parser.parse_args()

X_file = options.X_file
y_file = X_file.replace('X', 'y').replace('npz', 'npy')
folder = os.path.dirname(X_file)
original_folder = '/Users/jilljenn/code/ktm/data/duck'

X = load_npz(X_file)
y = np.load(y_file)

try:
    i_train = np.load(os.path.join(original_folder, 'i_train{}.npy'.format(options.subset)))
    i_test = np.load(os.path.join(original_folder, 'i_test{}.npy'.format(options.subset)))
    y_test = np.load(os.path.join(folder, 'y-ui-test.npy'.format(options.subset)))
    print('Yepee')
    print('Yepee', i_train, X.shape)
    X_train = X[i_train]
    y_train = y[i_train]
    X_test = X[i_test]
    # y_test = y[i_test]
    print(folder)
    if folder.endswith('0'):
        print('ok', X_train.shape, X_test.shape)
        X_train = vstack((X_train, X_test))
        print('ok', y_train.shape, y_test.shape)
        y_train = np.concatenate((y_train, y_test))
        print('yepee')
except Exception as e:
    print('NOES', e)
    sys.exit(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=False)

params = {
    'task': 'classification',
    'num_iter': options.iter,
    'rlog': True,
    'learning_method': 'mcmc',
    'k2': options.d
}
fm = pywFM.FM(**params)
model = fm.run(X_train, y_train, X_test, y_test)
y_pred_test = np.array(model.predictions)
np.save(os.path.join(folder, 'y_pred{}.npy'.format(options.subset)), y_pred_test)

print('Test predict:', y_pred_test)
print('Test was:', y_test)
print('Test ACC:', np.mean(y_test == np.round(y_pred_test)))
try:
    print('Test AUC', roc_auc_score(y_test, y_pred_test))
    print('Test NLL', log_loss(y_test, y_pred_test))
except ValueError:
    pass
