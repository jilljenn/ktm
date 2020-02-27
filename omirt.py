from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from autograd import grad
import autograd.numpy as np
import argparse
import sys
import os.path
from scipy.sparse import load_npz
import glob
import pandas as pd
import yaml
from datetime import datetime
import json


def log_loss(y, pred):
    return -(y * np.log(pred) + (1 - y) * np.log(1 - pred)).sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class OMIRT:
    def __init__(self, n_users=10, n_items=5, d=3, gamma=1., gamma_v=0.):
        self.n_users = n_users
        self.n_items = n_items
        self.d = d
        self.GAMMA = gamma
        self.GAMMA_V = gamma_v
        self.LAMBDA = 0.1
        self.mu = 0.
        # self.w = np.random.random(n_users + n_items)
        # self.V = np.random.random((n_users + n_items, d))
        self.y_pred = []
        self.predictions = []
        self.w = np.random.random(n_users)
        self.item_bias = np.random.random(n_items)
        self.V = np.random.random((n_users, d))
        self.item_embed = np.random.random((n_items, d))
        # self.V2 = np.power(self.V, 2)

    def load(self, folder):
        # Load mu
        if self.d == 0:
            w = np.load(os.path.join(folder, 'coef0.npy')).reshape(-1)
        else:
            w = np.load(os.path.join(folder, 'w.npy'))
            V = np.load(os.path.join(folder, 'V.npy'))
            self.V = V[:self.n_users]
            self.item_embed = V[self.n_users:]
        self.w = w[:self.n_users]
        self.item_bias = w[self.n_users:]
        print('w user', self.w.shape)
        print('w item', self.item_bias.shape)

    def full_fit(self, X, y):
        # pywFM and libFM
        print('full fit', X.shape, y.shape)
        
        for _ in range(100):
            if _ % 10 == 0:
                print(self.loss(X, y, self.mu, self.w, self.V, self.item_bias, self.item_embed))
            # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
            gradient = grad(lambda w: self.loss(X, y, self.mu, w, self.V, self.item_bias, self.item_embed))(self.w)
            # print('grad', gradient.shape)
            self.w -= self.GAMMA * gradient
            self.item_bias -= self.GAMMA * grad(lambda item_bias: self.loss(X, y, self.mu, self.w, self.V, item_bias, self.item_embed))(self.item_bias)
            if self.GAMMA_V:
                self.V -= self.GAMMA_V * grad(lambda V: self.loss(X, y, self.mu, self.w, V, self.item_bias, self.item_embed))(self.V)
                self.item_embed -= self.GAMMA_V * grad(lambda item_embed: self.loss(X, y, self.mu, self.w, self.V, self.item_bias, item_embed))(self.item_embed)
                
            # print(self.predict(X))

    def fit(self, X, y):
        # pywFM and libFM
        
        for _ in range(1):
            # print(self.loss(X, y, self.mu, self.w, self.V))
            # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
            gradient = grad(lambda w: self.loss(X, y, self.mu, w, self.V))(self.w)
            # print('grad', gradient.shape)
            self.w -= self.GAMMA * gradient
            if self.GAMMA_V:
                self.V -= self.GAMMA_V * grad(lambda V: self.loss(X, y, self.mu, self.w, V))(self.V)
            # print(self.predict(X))
            
    def predict(self, X, mu=None, w=None, V=None, item_bias=None, item_embed=None):
        if mu is None:
            mu = self.mu
            w = self.w
            V = self.V
            item_bias = self.item_bias
            item_embed = self.item_embed

        users = X[:, 0]
        items = X[:, 1]
            
        # for a in [X, w, V]:
        #     print(a.shape, a.dtype, type(a))
        V2 = np.power(V, 2)
        # print('before', X.shape)
        X_fm = X#.toarray()
        # print('puis', X.shape)
        # X2_fm = X_fm.copy()
        # X2_fm.data **= 2
        X2_fm = X#np.power(X, 2)

        #print(X.shape, w.shape)
        # return sigmoid(mu + X @ w)
        #y_pred = mu + X @ w

        # print('shape s x d', (self.V[users] * self.V[self.n_users + items]).shape)

        y_pred = mu + w[users] + self.item_bias[items]
        if self.d > 0:
            y_pred += np.sum(V[users] * self.item_embed[items], axis=1)
        return sigmoid(y_pred)

    def update(self, X, y):
        s = len(X)
        for x, outcome in zip(X, y):
            pred = self.predict(x.reshape(-1, 2))
            # print('update', x, pred, outcome)
            self.y_pred.append(pred.item())
            self.fit(x.reshape(-1, 2), outcome)
            # print(self.w.sum(), self.item_embed.sum())
        print(roc_auc_score(y, self.y_pred))

    def loss(self, X, y, mu, w, V, bias, embed):
        pred = self.predict(X, mu, w, V, bias, embed)
        return log_loss(y, pred) + self.LAMBDA * (
            mu ** 2 + np.sum(w ** 2) +
            np.sum(V ** 2))

    def save_results(self, model, y_test):
        iso_date = datetime.now().isoformat()
        self.predictions.append({
            'fold': 0,
            'pred': self.y_pred,
            'y': y_test.tolist()
        })
        with open(os.path.join(folder, 'results-{}.json'.format(iso_date)), 'w') as f:
            json.dump({
                'description': 'OMIRT',
                'predictions': self.predictions,
                'model': model  # Possibly add a checksum of the fold in the future
            }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OMIRT')
    parser.add_argument('X_file', type=str, nargs='?', default='dummy')
    parser.add_argument('--iter', type=int, nargs='?', default=200)
    parser.add_argument('--d', type=int, nargs='?', default=20)
    parser.add_argument('--lr', type=float, nargs='?', default=1.)
    parser.add_argument('--lr2', type=float, nargs='?', default=0.)
    parser.add_argument('--small', type=bool, nargs='?', const=True, default=False)
    options = parser.parse_args()
    print(vars(options))

    if options.X_file == 'dummy':
        ofm = OMIRT(n_users=10, n_items=5, d=3)
        df = pd.DataFrame.from_dict([
            {'user_id': 0, 'item_id': 0, 'correct': 0},
            {'user_id': 0, 'item_id': 1, 'correct': 1}
        ])
        print(df)
        X = np.array(df[['user_id', 'item_id']])
        y = np.array(df['correct'])
        print(X, y)
        # print(ofm.predict(X))
        ofm.fit(X, y)
        print(ofm.predict(X))
        sys.exit(0)
    
    X_file = options.X_file
    #y_file = X_file.replace('X', 'y').replace('npz', 'npy')
    folder = os.path.dirname(X_file)

    with open(os.path.join(folder, 'config.yml')) as f:
        config = yaml.load(f)
        print(config)

    df = pd.read_csv(X_file)
    X = np.array(df[['user_id', 'item_id']])
    y = np.array(df['correct'])
    nb_samples = len(y)

    # Are folds fixed already?
    X_trains = {}
    y_trains = {}
    X_tests = {}
    y_tests = {}
    folds = glob.glob(os.path.join(folder, 'folds/weak{}fold*.npy'.format(nb_samples)))
    if folds:
        for i, filename in enumerate(folds):
            i_test = np.load(filename)
            print('Fold', i, i_test.shape)
            i_train = list(set(range(nb_samples)) - set(i_test))
            X_trains[i] = X[i_train]
            y_trains[i] = y[i_train]
            if options.small:
                i_test = i_test[:5000]  # Try on 50 first test samples
            X_tests[i] = X[i_test]
            y_tests[i] = y[i_test]


    if X_trains:
        X_train, X_test, y_train, y_test = (X_trains[0], X_tests[0],
                                            y_trains[0], y_tests[0])
        print(X_train.shape, X_test.shape)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            shuffle=False)

    n = X_train.shape[1]
    ofm = OMIRT(config['nb_users'], config['nb_items'], options.d,
                gamma=options.lr, gamma_v=options.lr2)
    ofm.full_fit(X_train, y_train)
    # ofm.load(folder)
    y_pred = ofm.predict(X_train)
    print('train auc', roc_auc_score(y_train, y_pred))

    y_pred = ofm.predict(X_test)
    print(X_test[:5])
    print(y_test[:5])
    print(y_pred[:5])
    print('test auc', roc_auc_score(y_test, y_pred))
    
    # ofm.update(X_test, y_test)
    if len(X_test) > 10000:
        ofm.save_results(vars(options), y_test)
