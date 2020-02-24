from scipy.sparse import csr_matrix
#from sklearn.metrics import log_loss
from autograd import grad
import autograd.numpy as np
import argparse
import sys
import os.path
from scipy.sparse import load_npz
import glob
import pandas as pd
import yaml


def log_loss(y, pred):
    return -(y * np.log(pred) + (1 - y) * np.log(1 - pred)).sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class OMIRT:
    def __init__(self, n_users=10, n_items=5, d=3):
        self.n_users = n_users
        self.n_items = n_items
        self.GAMMA = 0.005
        self.LAMBDA = 0.1
        self.mu = 0.
        self.w = np.random.random(n_users + n_items)
        self.V = np.random.random((n_users + n_items, d))
        # self.user_bias = np.random.random(n_users)
        # self.item_bias = np.random.random(n_items)
        # self.user_embed = np.random.random((n_users, d))
        # self.item_embed = np.random.random((n_items, d))
        # self.V2 = np.power(self.V, 2)

    def fit(self, X, y):
        # pywFM and libFM
        
        for _ in range(10):
            print(self.loss(X, y, self.mu, self.w, self.V))
            # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
            gradient = grad(lambda w: self.loss(X, y, self.mu, w, self.V))(self.w)
            # print('grad', gradient.shape)
            self.w -= self.GAMMA * gradient
            self.V -= self.GAMMA * grad(lambda V: self.loss(X, y, self.mu, self.w, V))(self.V)
            print(self.predict(X))

    def predict(self, X, mu=None, w=None, V=None):
        if mu is None:
            mu = self.mu
            w = self.w
            V = self.V

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

        y_pred = (mu + w[users] + w[self.n_users + items] +
                  np.sum(V[users] * V[self.n_users + items], axis=1))
        return sigmoid(y_pred)

    def update(self, X, y):
        pass

    def loss(self, X, y, mu, w, V):
        pred = self.predict(X, mu, w, V)
        return log_loss(y, pred) + self.LAMBDA * (
            mu ** 2 + np.sum(w ** 2) +
            np.sum(V ** 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OMIRT')
    parser.add_argument('X_file', type=str, nargs='?', default='dummy')
    parser.add_argument('--iter', type=int, nargs='?', default=200)
    parser.add_argument('--d', type=int, nargs='?', default=20)
    options = parser.parse_args()

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
    ofm = OMIRT(config['nb_users'], config['nb_items'], options.d)
    ofm.fit(X_train, y_train)
    
