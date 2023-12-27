"""
Factorization machines on sparse features
"""
import argparse
from datetime import datetime
from pathlib import Path
import json
import os
from sklearn.metrics import roc_auc_score, log_loss
from scipy.sparse import load_npz
import pywFM
import numpy as np
from dataio import get_paths, load_folds
import sklearn


# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = str(Path('libfm/bin').absolute()) + '/'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FMClassifier(sklearn.base.BaseEstimator):
    def __init__(self, embedding_size=20, nb_iterations=40):
        super().__init__()
        self.embedding_size = embedding_size
        self.nb_iterations = nb_iterations

    def fit(self, X, y):
        """
        X is usually sparse, nb_samples x nb_features
        y is binary
        """
        fm = pywFM.FM(task='classification', num_iter=self.nb_iterations,
                      k2=self.embedding_size, rlog=True)  # MCMC method
        # rlog contains the RMSE at each epoch, we do not need it here
        model = fm.run(X, y, X, y)

        # Store parameters
        self.mu = model.global_bias
        self.W = np.array(model.weights)
        self.V = model.pairwise_interactions
        self.V2 = np.power(self.V, 2)
        self.rlog = model.rlog
        return self

    def predict_proba(self, X):
        X2 = X.copy()
        if scipy.sparse.issparse(X):
            X2.data **= 2
        else:
            X2 **= 2
                     
        y_pred = (self.mu + X @ self.W +
                  0.5 * (np.power(X @ self.V, 2).sum(axis=1)
                         - (X2 @ self.V2).sum(axis=1)).A1)
        return sigmoid(y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FM')
    parser.add_argument('X_file', type=str, nargs='?')
    parser.add_argument('--iter', type=int, nargs='?', default=20)
    parser.add_argument('--d', type=int, nargs='?', default=20)
    parser.add_argument('--subset', type=int, nargs='?', default=0)
    parser.add_argument('--metrics', type=bool, nargs='?', const=True,
        default=False)
    parser.add_argument('--folds', type=str, nargs='?', default='weak')
    options = parser.parse_args()


    df, X_file, folder, y_file, y_pred_file = get_paths(options, 'FM')
    X_sp = load_npz(X_file).tocsr()
    nb_samples, _ = X_sp.shape
    y = np.load(y_file).astype(np.int32)


    predictions = []
    params = {
        'task': 'classification',
        'num_iter': options.iter,
        'rlog': True,
        'learning_method': 'mcmc',
        'k2': options.d
    }
    fm = pywFM.FM(**params)
    for i, (i_train, i_test) in enumerate(load_folds(options, df)):
        X_train, X_test, y_train, y_test = (X_sp[i_train], X_sp[i_test],
                                            y[i_train], y[i_test])

        model = fm.run(X_train, y_train, X_test, y_test)
        y_pred_test = np.array(model.predictions)

        predictions.append({
            'fold': 0,
            'pred': y_pred_test.tolist(),
            'y': y_test.tolist()
        })

        if options.metrics:
            df_test = df.iloc[i_test]
            assert len(df_test) == len(y_pred_test)
            df_test['pred'] = y_pred_test
            df_test.to_csv(y_pred_file, index=False)

        print('Test predict:', y_pred_test)
        print('Test was:', y_test)
        print('Test ACC:', np.mean(y_test == np.round(y_pred_test)))
        try:
            print('Test AUC', roc_auc_score(y_test, y_pred_test))
            print('Test NLL', log_loss(y_test, y_pred_test))
        except ValueError:
            pass

        iso_date = datetime.now().isoformat()
        np.save(folder / 'w.npy', np.array(model.weights))
        np.save(folder / 'V.npy', model.pairwise_interactions)
        saved_results = {
            'predictions': predictions,
            'model': vars(options),
            'mu': model.global_bias,
        }
        with open(folder / f'results-{iso_date}.json', 'w') as f:
            json.dump(saved_results, f)
        break
