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
from bijection import sample_pairs


SENSITIVE_ATTR = 'school_id'  # Should be defined according to the dataset
THIS_GROUP = 25
BATCH_SIZE = 1000
EPS = 1e-15


def log_loss(y, pred):
    this_pred = np.clip(pred, EPS, 1 - EPS)
    return -(y * np.log(this_pred) + (1 - y) * np.log(1 - this_pred))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


class OMIRT:
    def __init__(self, X, y, i_, n_users=10, n_items=5, d=3, lambda_=0., gamma=1., gamma_v=0., n_epoch=10, df=None, fair=False, training='ll'):
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
        self.item_slopes = np.random.random(n_items)
        self.V = np.random.random((n_users, d))
        self.item_embed = np.random.random((n_items, d))
        # self.V2 = np.power(self.V, 2)
        self.fair = fair
        self.metrics = defaultdict(list)
        self.prepare_sets()
        attr_ids = self.X_train[:, 2]
        n_attr = len(np.unique(attr_ids))
        self.n_samples = len(i_['train'])
        print(self.n_samples, 'samples')
        self.W_attr = np.zeros((n_attr, self.n_samples))
        self.W_attr[attr_ids, range(self.n_samples)] = 1
        self.W_attr /= self.W_attr.sum(axis=1)[:, None]  # Normalize
        self.n_epoch = n_epoch
        self.batch_size = BATCH_SIZE
        self.n_batches = self.n_samples // self.batch_size
        print('n_iter will be', self.n_epoch, self.n_batches, self.n_epoch * self.n_batches)
        self.c = 0.
        self.training = training

    def prepare_sets(self):
        gasp = 0
        for key in self.i_:
            setattr(self, 'X_' + key, self.X[self.i_[key]])
            setattr(self, 'y_' + key, self.y[self.i_[key]])
            setattr(self, 'n_' + key, len(self.y[self.i_[key]]))
            print(key, type(self.i_[key]), type(self.i_[key][0]), type(self.y), getattr(self, 'X_' + key).size)
            gasp += getattr(self, 'X_' + key).size
        print(gasp / 1e6)

    def prepare_batch(self, i_batch):
        for y in range(2):
            for z in range(2):
                i_['batch_{}_{}'.format(y, z)] = []  # Clear current batch
            i_['batch_{}'.format(y)] = []

        for i in self.i_['train'][i_batch * self.batch_size:(i_batch + 1) * self.batch_size]:
            i_['batch_{}_{}'.format(self.y[i], self.X[i, 2])].append(i)
            i_['batch_{}'.format(self.y[i])].append(i)

        for y in range(2):
            for z in range(2):
                setattr(self, 'X_batch_{}_{}'.format(y, z), self.X[i_['batch_{}_{}'.format(y, z)]])
                setattr(self, 'n_batch_{}_{}'.format(y, z), len(i_['batch_{}_{}'.format(y, z)]))
            setattr(self, 'X_batch_{}'.format(y), self.X[i_['batch_{}'.format(y)]])
            setattr(self, 'n_batch_{}'.format(y), len(i_['batch_{}'.format(y)]))


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

    def compute_metrics(self, step):
        self.metrics['step'].append(step)
        print(step)
        auc = {}
        loss = {}
        for dataset in {'train', 'valid', 'test'}:
            for group in {'', '_1', '_0'}:
                pred = self.predict(getattr(self, 'X_' + dataset + group))
                auc[group] = roc_auc_score(getattr(self, 'y_' + dataset + group), pred)
                if self.training == 'auc':
                    loss[group] = self.auc_loss(self.c, self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes)
                else:
                    loss[group] = self.loss(self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes)
                    # print('full loss', self.auc_loss(self.c, self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes))
            key = 'delta auc ' + dataset
            self.metrics[key].append(auc['_1'] - auc['_0'])
            self.metrics['auc ' + dataset].append(auc[''])
            self.metrics['loss ' + dataset].append(loss[''])
            # print(key, auc)

    def full_fit(self):
        # pywFM and libFM
        # print('full fit', X.shape, y.shape)
        
        for step in tqdm(range(self.n_epoch)):
            if step % 10 == 0:
                self.compute_metrics(step)
                pred = self.predict(self.X_train)
                print('loss', ll(self.y_train, pred))

            # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
            gradient = grad(lambda w: self.loss(self.mu, w, self.V, self.item_bias, self.item_embed, self.item_slopes))(self.w)
            # print('grad', gradient.shape)
            self.w -= self.GAMMA * gradient
            self.item_bias -= self.GAMMA * grad(lambda item_bias: self.loss(self.mu, self.w, self.V, item_bias, self.item_embed, self.item_slopes))(self.item_bias)
            self.item_slopes -= self.GAMMA * grad(lambda item_slopes: self.loss(self.mu, self.w, self.V, self.item_bias, self.item_embed, item_slopes))(self.item_slopes)
            
            if self.GAMMA_V:
                self.V -= self.GAMMA_V * grad(lambda V: self.loss(self.mu, self.w, V, self.item_bias, self.item_embed, self.item_slopes))(self.V)
                self.item_embed -= self.GAMMA_V * grad(lambda item_embed: self.loss(self.mu, self.w, self.V, self.item_bias, item_embed, self.item_slopes))(self.item_embed)
                
            # print(self.predict(X))

    def full_relaxed_fit(self):
        #print('full relaxed fit', X.shape, y.shape)

        self.c = 0
        step = 0
        for epoch in tqdm(range(self.n_epoch)):
            random.shuffle(self.i_['train'])
            
            for i_batch in range(self.n_batches):
                step += 1

                if step % 100 == 0:
                    self.compute_metrics(step)
                    self.loss(self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes, display=True)
                
                if self.fair and step > 0 and step % 50 == 0:
                    auc_1 = self.relaxed_auc('valid', '_1', self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes, 10000) 
                    auc_0 = self.relaxed_auc('valid', '_0', self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes, 10000)
                    self.c += np.sign(auc_1 - auc_0) * 0.01
                    self.c = np.clip(self.c, -1, 1)

                self.prepare_batch(i_batch)
                # for z in range(2):
                #     for y in range(2):
                #         print(z, y, getattr(self, 'X_batch_{}_{}'.format(y, z))[:5])  # Display batch

                # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
                gradient = grad(lambda w: self.auc_loss(self.c, self.mu, w, self.V, self.item_bias, self.item_embed, self.item_slopes))(self.w)
                # print('grad', gradient.shape, gradient)
                self.w -= self.GAMMA * gradient
                self.item_bias -= self.GAMMA * grad(lambda item_bias: self.auc_loss(self.c, self.mu, self.w, self.V, item_bias, self.item_embed, self.item_slopes))(self.item_bias)
                self.item_slopes -= self.GAMMA * grad(lambda item_slopes: self.auc_loss(self.c, self.mu, self.w, self.V, self.item_bias, self.item_embed, item_slopes))(self.item_slopes)
                if self.GAMMA_V:
                    self.V -= self.GAMMA_V * grad(lambda V: self.auc_loss(self.c, self.mu, self.w, V, self.item_bias, self.item_embed, self.item_slopes))(self.V)
                    self.item_embed -= self.GAMMA_V * grad(lambda item_embed: self.auc_loss(self.c, self.mu, self.w, self.V, self.item_bias, item_embed, self.item_slopes))(self.item_embed)
            
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
            
    def predict_logits(self, X, mu=None, w=None, V=None, item_bias=None, item_embed=None, slopes=None):
        if mu is None:
            mu = self.mu
            w = self.w
            V = self.V
            item_bias = self.item_bias
            item_embed = self.item_embed
            slopes = self.item_slopes

        users = X[:, 0]
        items = X[:, 1]
        attempts = X[:, 3]

        y_pred = mu + w[users] + item_bias[items] + attempts * slopes[items]
        if self.d > 0:
            y_pred += np.sum(V[users] * item_embed[items], axis=1)
        return y_pred

    def predict(self, X, mu=None, w=None, V=None, item_bias=None, item_embed=None, item_slopes=None):
        if mu is None:
            mu = self.mu
            w = self.w
            V = self.V
            item_bias = self.item_bias
            item_embed = self.item_embed
            item_slopes = self.item_slopes

        y_pred = self.predict_logits(X, mu, w, V, item_bias, item_embed, item_slopes)
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

    def loss(self, mu, w, V, bias, embed, slopes, display=False):
        if self.training == 'auc':
            pred_1 = self.predict(self.X_batch_1, mu, w, V, bias, embed, slopes)
            pred_0 = self.predict(self.X_batch_0, mu, w, V, bias, embed, slopes)
            ll = np.concatenate((log_loss(np.ones(self.n_batch_1), pred_1),
                                 log_loss(np.zeros(self.n_batch_0), pred_0)))
            if display:
                print('ll', ll.mean())
            return ll.mean()
        
        pred = self.predict(self.X_train, mu, w, V, bias, embed, slopes)
        ll = log_loss(self.y_train, pred)
        reg = self.LAMBDA * (
            mu ** 2 + np.sum(w ** 2) +
            np.sum(bias ** 2) + np.sum(embed ** 2) +
            np.sum(V ** 2))
        if self.training == 'll':
            return ll.sum() + reg
        ll_per_group = self.W_attr @ ll
        if self.training == 'mean':
            return self.n_samples * ll_per_group.mean() + reg
        if self.training == 'min':
            return self.n_samples * np.dot(softmax(-ll_per_group), ll_per_group) + reg

    def auc_loss(self, c, mu, w, V, bias, embed, slopes):
        auc = self.relaxed_auc('batch', '', mu, w, V, bias, embed, slopes)
        
        loss = self.loss(mu, w, V, bias, embed, slopes)
        
        auc_1 = self.relaxed_auc('batch', '_1', mu, w, V, bias, embed, slopes)
        auc_0 = self.relaxed_auc('batch', '_0', mu, w, V, bias, embed, slopes)
        # return -auc  # Optimize overall AUC
        # return -auc_1  # Only optimize AUC of subgroup
        # return (auc_1 - auc_0) ** 2  # Only minimize delta AUC
        # return -auc - auc_1
        # return -auc_1 - auc_0
        # return -auc + self.fair * np.abs(auc_1 - auc_0) + self.LAMBDA * (mu ** 2 + np.sum(w ** 2) + np.sum(V ** 2) + np.sum(bias ** 2) + np.sum(embed ** 2))
        # return 100 * loss + self.fair * np.abs(auc_1 - auc_0)
        # print(100 * loss)
        # print(auc_1 - auc_0) ** 2
        # return loss + self.fair * (auc_1 - auc_0) ** 2
        return loss + self.fair * c * (auc_1 - auc_0)
        # return -auc + self.fair * (auc_1 - auc_0) ** 2 + self.LAMBDA * (mu ** 2 + np.sum(w ** 2) + np.sum(V ** 2) + np.sum(bias ** 2) + np.sum(embed ** 2))
        # return -auc + self.fair * self.c * (auc_1 - auc_0) + self.LAMBDA * (mu ** 2 + np.sum(w ** 2) + np.sum(V ** 2) + np.sum(bias ** 2) + np.sum(embed ** 2))

    def relaxed_auc(self, dataset, suffix, mu, w, V, bias, embed, slopes, B=100):
        auc = 0
        n = len(y)
        # print('n samples', getattr(self, 'n_batch_1' + suffix))
        # print('meow', B, type(B))
        # print(getattr(self, 'n_{}_0{}'.format(dataset, suffix)))
        n_pos = getattr(self, 'n_{}_1{}'.format(dataset, suffix))
        n_neg = getattr(self, 'n_{}_0{}'.format(dataset, suffix))
        if n_pos == 0 or n_neg == 0:
            return 0.
        ii = np.random.choice(n_pos, B)
        jj = np.random.choice(n_neg, B)
        # sys.exit(0)
        pred_1 = self.predict_logits(getattr(self, 'X_{}_1{}'.format(dataset, suffix))[ii], mu, w, V, bias, embed, slopes)
        pred_0 = self.predict_logits(getattr(self, 'X_{}_0{}'.format(dataset, suffix))[jj], mu, w, V, bias, embed, slopes)
        return sigmoid(pred_1 - pred_0).mean()
    
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
    parser.add_argument('--epoch', type=int, nargs='?', default=10)
    parser.add_argument('--lr', type=float, nargs='?', default=1.)
    parser.add_argument('--lr2', type=float, nargs='?', default=0.)
    parser.add_argument('--reg', type=float, nargs='?', default=0.)
    parser.add_argument('--small', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--training', type=str, nargs='?', default='ll')
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
    # df['attribute'] = (df[SENSITIVE_ATTR] == THIS_GROUP).astype(int)
    df['attribute'] = (df[SENSITIVE_ATTR] % 2 == 0).astype(int)
    # Or attribute % 2 == 0
    print(df.head())
    print(df['attribute'].value_counts())
    
    X = np.array(df[['user_id', 'item_id', 'attribute', 'attempts']])
    y = np.array(df['correct'])
    nb_samples = len(y)
    
    # Are folds fixed already?    
    folds = sorted(glob.glob(os.path.join(folder, 'folds/60weak{}fold*.npy'.format(nb_samples))))
    valids = sorted(glob.glob(os.path.join(folder, 'folds/36weak{}valid*.npy'.format(nb_samples))))
    if not folds:
        print('No folds')
        # For example:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        #                                                     shuffle=False)

    i_ = {}
    for i, (filename, valid) in enumerate(zip(folds, valids)):
        i_['test'] = set(np.load(filename))
        i_['valid'] = set(np.load(valid))
        i_['train'] = set(range(nb_samples)) - i_['test'] - i_['valid']
        print(len(i_['train']), len(i_['valid']), len(i_['test']))
        # random.shuffle(i_['train'])
        # PIVOT = round(0.4 * len(i_['train']))  # When it was weak generalization
        # i_['valid'] = set(i_['train'][:PIVOT])
        # i_['train'] = set(i_['train'][PIVOT:])

        if options.small:
            i_['test'] = i_['test'][:5000]  # Try on 5000 first test samples

        print('Fold', i, len(i_['test']))
        indices = np.array(range(len(X))).astype(int)
        i_['1'] = set(indices[X[:, 2] == 1])
        i_['0'] = set(indices[X[:, 2] == 0])
        assert i_['0'] | i_['1'] == set(range(len(X)))

        i_['y1'] = set(indices[y == 1])
        i_['y0'] = set(indices[y == 0])
        print('+', list(i_['y1'])[:5], type(list(i_['y1'])[0]))
        assert i_['y0'] | i_['y1'] == set(range(len(X)))
        
        for dataset in {'train', 'valid', 'test'}:
            for attr in '01':
                for label in '01':
                    i_[dataset + '_' + label + '_' + attr] = i_[dataset] & i_[attr] & i_['y' + label]
                i_[dataset + '_' + attr] = i_[dataset] & i_[attr]

        for key in i_:
            print(key, type(list(i_[key])[0]))
            i_[key] = list(i_[key])

        ofm = OMIRT(X, y, i_, config['nb_users'], config['nb_items'], options.d,
                    lambda_=options.reg, gamma=options.lr, gamma_v=options.lr2,
                    n_epoch=options.epoch, fair=options.fair, training=options.training)

        if options.training == 'auc':
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

        plt.show()
