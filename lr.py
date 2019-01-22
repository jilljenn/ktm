from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import load_npz
import numpy as np
import sys


FULL = True
X_file = sys.argv[1]
y_file = X_file.replace('X', 'y').replace('npz', 'npy')

X = load_npz(X_file)
y = np.load(y_file)
if FULL:
    X_train, X_test, y_train, y_test = X, X, y, y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=False)

model = LogisticRegression()  # Has L2 regularization by default
model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)[:, 1]
print('Train predict:', y_pred_train)
print('Train was:', y_train)
print('Train ACC:', np.mean(y_train == np.round(y_pred_train)))
print('Train AUC:', roc_auc_score(y_train, y_pred_train))

y_pred_test = model.predict_proba(X_test)[:, 1]
print('Test predict:', y_pred_test)
print('Test was:', y_test)
print('Test ACC:', np.mean(y_test == np.round(y_pred_test)))
try:
    print('Test AUC:', roc_auc_score(y_test, y_pred_test))
except ValueError:
    pass

np.save(X_file.replace('.npz', '-coef.npy'), model.coef_)
