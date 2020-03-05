from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, log_loss as ll
from sklearn.model_selection import train_test_split
from eval_metrics import all_metrics
from itertools import combinations
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
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


SENSITIVE_ATTR = 'school_id'  # Should be defined according to the dataset
THIS_GROUP = 4
all_pairs = np.array(list(combinations(range(100), 2)))
EPS = 1e-15


def log_loss(y, pred):
    this_pred = np.clip(pred, EPS, 1 - EPS)
    return -(y * np.log(this_pred) + (1 - y) * np.log(1 - this_pred)).sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class OMIRT:
    def __init__(self, X, y, i_, n_users=10, n_items=5, d=3, lambda_=0., gamma=1., gamma_v=0., n_iter=1000, df=None, fair=False):
        self.X = X
        self.y = y
        self.i_ = i_
        self.n_users = n_users
        self.n_items = n_items
        self.d = d
        self.GAMMA = gamma
        self.GAMMA_V = gamma_v
        self.LAMBDA = lambda_
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
        self.n_iter = n_iter
        self.fair = fair
        self.metrics = defaultdict(list)
        self.prepare_sets()

    def prepare_sets(self):
        for key in self.i_:
            setattr(self, 'X_' + key, self.X[self.i_[key]])
            setattr(self, 'y_' + key, self.y[self.i_[key]])        
        
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
        
        for _ in range(500):
            if _ % 100 == 0:
                pred = self.predict(X)
                print('loss', ll(y, pred))
                print(self.loss(X, y, self.mu, self.w, self.V, self.item_bias, self.item_embed) / len(y), self.w.sum(), self.item_bias.sum())
            # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
            gradient = grad(lambda w: self.loss(X, y, self.mu, w, self.V, self.item_bias, self.item_embed))(self.w)
            # print('grad', gradient.shape)
            self.w -= self.GAMMA * gradient
            self.item_bias -= self.GAMMA * grad(lambda item_bias: self.loss(X, y, self.mu, self.w, self.V, item_bias, self.item_embed))(self.item_bias)
            
            if self.GAMMA_V:
                self.V -= self.GAMMA_V * grad(lambda V: self.loss(X, y, self.mu, self.w, V, self.item_bias, self.item_embed))(self.V)
                self.item_embed -= self.GAMMA_V * grad(lambda item_embed: self.loss(X, y, self.mu, self.w, self.V, self.item_bias, item_embed))(self.item_embed)
                
            # print(self.predict(X))

    def full_relaxed_fit(self):
        # pywFM and libFM
        #print('full relaxed fit', X.shape, y.shape)

        c = 0
        for step in tqdm(range(self.n_iter)):
            if step % 1000 == 0:
                self.metrics['step'].append(step)

                for group in {'', '_1', '_0'}:
                    for dataset in {'train', 'valid', 'test'}:
                        pred = self.predict(getattr(self, 'X_' + dataset + group))
                        auc = roc_auc_score(getattr(self, 'y_' + dataset + group), pred)
                        key = 'auc' + group + ' ' + dataset
                        self.metrics[key].append(auc)
                        print(key, auc)
                
                # print('auc_0', roc_auc_score(y[self.unprotected], pred[self.unprotected]))
                print(c)

            if step > 0 and step % 50 == 0:
                auc_1 = self.relaxed_auc(self.X_valid_1, self.y_valid_1, self.mu, self.w, self.V, self.item_bias, self.item_embed)
                auc_0 = self.relaxed_auc(self.X_valid_0, self.y_valid_0, self.mu, self.w, self.V, self.item_bias, self.item_embed)
                c += np.sign(auc_1 - auc_0) * 0.01
                c = np.clip(c, -1, 1)
            # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
            gradient = grad(lambda w: self.auc_loss(c, self.mu, w, self.V, self.item_bias, self.item_embed))(self.w)
            # print('grad', gradient.shape, gradient)
            self.w -= self.GAMMA * gradient
            self.item_bias -= self.GAMMA * grad(lambda item_bias: self.auc_loss(c, self.mu, self.w, self.V, item_bias, self.item_embed))(self.item_bias)
            if self.GAMMA_V:
                self.V -= self.GAMMA_V * grad(lambda V: self.auc_loss(c, self.mu, self.w, V, self.item_bias, self.item_embed))(self.V)
                self.item_embed -= self.GAMMA_V * grad(lambda item_embed: self.auc_loss(c, self.mu, self.w, self.V, self.item_bias, item_embed))(self.item_embed)
                
            # print(self.predict(X))
            
    def fit(self, X, y):
        # pywFM and libFM
        
        for _ in range(1):
            # print(self.loss(X, y, self.mu, self.w, self.V))
            # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
            gradient = grad(lambda w: self.loss(X, y, self.mu, w, self.V, self.item_bias, self.item_embed))(self.w)
            # print('grad', gradient.shape)
            self.w -= 1 * gradient
            self.GAMMA_V = 0.1 
            if self.GAMMA_V:
                self.V -= self.GAMMA_V * grad(lambda V: self.loss(X, y, self.mu, self.w, V, self.item_bias, self.item_embed))(self.V)
            # print(self.predict(X))
            
    def predict_logits(self, X, mu=None, w=None, V=None, item_bias=None, item_embed=None):
        if mu is None:
            mu = self.mu
            w = self.w
            V = self.V
            item_bias = self.item_bias
            item_embed = self.item_embed

        users = X[:, 0]
        items = X[:, 1]

        y_pred = mu + w[users] + item_bias[items]
        if self.d > 0:
            y_pred += np.sum(V[users] * item_embed[items], axis=1)
        return y_pred

    def predict(self, X, mu=None, w=None, V=None, item_bias=None, item_embed=None):
        if mu is None:
            mu = self.mu
            w = self.w
            V = self.V
            item_bias = self.item_bias
            item_embed = self.item_embed

        y_pred = self.predict_logits(X, mu, w, V, item_bias, item_embed)
        return sigmoid(y_pred)
    
    def update(self, X, y):
        s = len(X)
        self.y_pred = []
        for x, outcome in zip(X, y):
            pred = self.predict(x.reshape(-1, 2))
            # print('update', x, pred, outcome)
            self.y_pred.append(pred.item())
            self.fit(x.reshape(-1, 2), outcome)
            # print(self.w.sum(), self.item_embed.sum())
        print(roc_auc_score(y, self.y_pred))

    def loss(self, mu, w, V, bias, embed):
        pred = self.predict(self.X_train, mu, w, V, bias, embed)
        return log_loss(self.y_train, pred) + self.LAMBDA * (
            mu ** 2 + np.sum(w ** 2) +
            np.sum(bias ** 2) + np.sum(embed ** 2) +
            np.sum(V ** 2))

    def auc_loss(self, c, mu, w, V, bias, embed):
        auc = self.relaxed_auc(self.X_train, self.y_train, mu, w, V, bias, embed)
        auc_1 = self.relaxed_auc(self.X_train_1, self.y_train_1, mu, w, V, bias, embed)
        auc_0 = self.relaxed_auc(self.X_train_0, self.y_train_0, mu, w, V, bias, embed)
        # return -auc_1  # Only optimize AUC of subgroup
        return c * (auc_1 - auc_0)
        return 100 - auc - self.fair * c * (auc_1 - auc_0) + self.LAMBDA * (mu ** 2 + np.sum(w ** 2) + np.sum(V ** 2) + np.sum(bias ** 2) + np.sum(embed ** 2))

    def relaxed_auc(self, X, y, mu, w, V, bias, embed):
        assert len(y) > 100
        batch = np.random.choice(len(y), 100)
        y_batch = y[batch]
        pred = self.predict_logits(X[batch], mu, w, V, bias, embed)
        auc = 0
        n = len(y)
        metabatch = np.random.choice(len(all_pairs), 100)
        ii = all_pairs[metabatch][:, 0]
        jj = all_pairs[metabatch][:, 1]
        auc = sigmoid((pred[ii] - pred[jj]) * (y_batch[ii] - y_batch[jj])).sum()
        return auc

    def save_results(self, model, test):
        iso_date = datetime.now().isoformat()
        self.predictions.append({
            'fold': 0,
            'pred': self.y_pred,
            'y': self.y_test.tolist()
        })
        saved_results = {
            'description': 'OMIRT',
            'predictions': self.predictions,
            'model': model  # Possibly add a checksum of the fold in the future
        }
        with open(os.path.join(folder, 'results-{}.json'.format(iso_date)), 'w') as f:
            json.dump(saved_results, f)
        all_metrics(saved_results, test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OMIRT')
    parser.add_argument('X_file', type=str, nargs='?', default='dummy')
    parser.add_argument('--d', type=int, nargs='?', default=20)
    parser.add_argument('--iter', type=int, nargs='?', default=1000)
    parser.add_argument('--lr', type=float, nargs='?', default=1.)
    parser.add_argument('--lr2', type=float, nargs='?', default=0.)
    parser.add_argument('--reg', type=float, nargs='?', default=0.)
    parser.add_argument('--small', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--auc', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--fair', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--online', type=bool, nargs='?', const=True, default=False)
    options = parser.parse_args()
    print(vars(options))

    if options.X_file == 'dummy':  # I'm pretty sure this does not work anymore
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
    # Definition of protected subgroup
    df['attribute'] = (df[SENSITIVE_ATTR] == THIS_GROUP).astype(int)
    # Or attribute % 2 == 0
    print(df.head())
    print(df['attribute'].value_counts())
    
    X = np.array(df[['user_id', 'item_id', 'attribute']])
    y = np.array(df['correct'])
    nb_samples = len(y)
    
    # Are folds fixed already?    
    folds = glob.glob(os.path.join(folder, 'folds/50weak{}fold*.npy'.format(nb_samples)))
    if not folds:
        print('No folds')
        # For example:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        #                                                     shuffle=False)

    i_ = {}
    for i, filename in enumerate(folds):
        i_['test'] = set(np.load(filename))  # Everything should be in JSON folds
        i_['train'] = list(set(range(nb_samples)) - i_['test'])
        random.shuffle(i_['train'])
        PIVOT = round(0.4 * len(i_['train']))
        i_['valid'] = set(i_['train'][:PIVOT])
        i_['train'] = set(i_['train'][PIVOT:])

        if options.small:
            i_['test'] = i_['test'][:5000]  # Try on 5000 first test samples

        print('Fold', i, len(i_['test']))
        i_['1'] = set(np.array(range(len(X)))[X[:, 2] == 1])
        i_['0'] = set(np.array(range(len(X)))[X[:, 2] == 0])
        assert i_['0'] | i_['1'] == set(range(len(X)))

        for dataset in {'train', 'valid', 'test'}:
            for attribute in '01':
                i_[dataset + '_' + attribute] = i_[dataset] & i_[attribute]

        for key in i_:
            i_[key] = list(i_[key])

        ofm = OMIRT(X, y, i_, config['nb_users'], config['nb_items'], options.d,
                    lambda_=options.reg, gamma=options.lr, gamma_v=options.lr2,
                    n_iter=options.iter, fair=options.fair)

        if options.auc:
            ofm.full_relaxed_fit()
        else:
            ofm.full_fit()

        # ofm.load(folder)
        y_pred = ofm.predict(ofm.X_train)
        print('train auc', roc_auc_score(ofm.y_train, y_pred))

        for metric in ofm.metrics:
            if metric == 'step':
                continue
            plt.plot(ofm.metrics['step'], ofm.metrics[metric], label=metric)
        plt.legend()
        plt.show()

        y_pred = ofm.predict(ofm.X_test)
        ofm.y_pred = y_pred.tolist()  # Save for future use
        print(ofm.X_test[:5])
        print(ofm.y_test[:5])
        print(ofm.y_pred[:5])
        print('test auc', roc_auc_score(ofm.y_test, y_pred))

        test = df.iloc[i_['test']]

        if options.online:
            ofm.update()
        if len(ofm.X_test) > 10000:
            ofm.save_results(vars(options), test)
