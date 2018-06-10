from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.sparse import load_npz
import pywFM
import argparse
import numpy as np
import os


# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = os.path.join(os.path.dirname(__file__),
                                        'libfm/bin/')

parser = argparse.ArgumentParser(description='Run FM')
parser.add_argument('X_file', type=str, nargs='?')
parser.add_argument('--iter', type=int, nargs='?', default=200)
parser.add_argument('--d', type=int, nargs='?', default=20)
options = parser.parse_args()

X_file = options.X_file
y_file = X_file.replace('X', 'y').replace('npz', 'npy')

X = load_npz(X_file)
y = np.load(y_file)
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

print('Test predict:', y_pred_test)
print('Test was:', y_test)
print('Test ACC:', np.mean(y_test == np.round(y_pred_test)))
try:
    print('Test AUC', roc_auc_score(y_test, y_pred_test))
except ValueError:
    pass
