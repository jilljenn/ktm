from collections import defaultdict, Counter
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
"""
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Add, Activation
"""
import tensorflow as tf


SENSITIVE_ATTR = 'school_id'  # Should be defined according to the dataset
THIS_GROUP = 25
BATCH_SIZE = 1000
EPS = 1e-15


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def log_loss(y, pred):
    # this_pred = np.clip(tf.squeeze(pred), EPS, 1 - EPS)
    this_pred = tf.clip_by_value(tf.squeeze(pred), EPS, 1 - EPS)
    print(y.shape, this_pred.shape)
    return -(y * np.log(this_pred) + (1 - y) * np.log(1 - this_pred))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def relu(x):
    return x * (x > 0)


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
        # self.item_bias = np.random.random(n_items)
        self.item_slopes = np.random.random(n_items)
        # self.w = np.random.random(n_users)
        # self.V = np.random.random((n_users, d))
        self.V = np.random.random((10, 3))
        self.w = np.random.random(3)
        self.item_bias = np.random.random(3)
        self.item_embed = np.random.random((n_items, d))
        self.users = np.random.random((n_users, 5))
        self.items = np.random.random((n_items, 5))
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
        self.prepare_model()

    def prepare_model(self):
        n_dim = 4
        # n_dim = 1
        '''
        self.model = Sequential([
            Embedding(self.n_users + self.n_items, n_dim, input_length=4),
            Flatten(),
            Dense(units=4, activation='relu'),
            # Dense(units=2, activation='relu'),
            Dense(units=1, activation='sigmoid')#, kernel_regularizer=regularizers.l2(0.001))
        ])
        '''
        self.tf_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.n_users + self.n_items, n_dim, input_length=4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=4, activation='relu'),
            # Dense(units=2, activation='relu'),
            tf.keras.layers.Dense(units=1)#, activation='sigmoid')#, kernel_regularizer=regularizers.l2(0.001))
        ])
        self.tf_model.build()
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        """
        self.model.compile(
            loss=keras.losses.binary_crossentropy,
            # optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
            optimizer=keras.optimizers.Adam(lr=0.01),
            metrics=['accuracy', auroc])
        """

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
                '''if self.training == 'auc':
                    loss[group] = self.auc_loss(self.c, self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes)
                else:
                    loss[group] = self.loss(self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes)'''
                    # print('full loss', self.auc_loss(self.c, self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes))
            key = 'delta auc ' + dataset
            self.metrics[key].append(auc['_1'] - auc['_0'])
            self.metrics['auc ' + dataset].append(auc[''])
            # self.metrics['loss ' + dataset].append(loss[''])
            # print(key, auc)

    def encode(self, X):
        users = X[:, 0]
        items = X[:, 1]
        return np.concatenate((self.users[users], self.items[items]), axis=1)

    def deep_fit(self):
        print('Training')

        es = keras.callbacks.EarlyStopping(patience=1)
        
        # X_train = self.encode(self.X_train)
        print(Counter(self.y_train))
        X_train = self.X_train[:, :4]
        X_valid = self.X_valid[:, :4]
        self.model.fit(X_train, self.y_train,
                       validation_data=(X_valid, self.y_valid),
                       epochs=50, batch_size=10000, callbacks=[es])

        # X_test = self.encode(self.X_test)
        X_test = self.X_test[:, :4]
        # print('train', self.model.evaluate(X_train, self.y_train))
        pred = self.model.predict(X_train)
        # print(Counter(pred.reshape(-1)))
        print('auc', roc_auc_score(self.y_train, pred))
        
        # print('test', self.model.evaluate(X_test, self.y_test))
        pred = self.model.predict(X_test)
        print('auc', roc_auc_score(self.y_test, pred))

    # @tf.function
    def tf_fit(self):

        #train_ds = tf.data.Dataset.from_tensor_slices(
        #    (self.X_train, self.y_train))

        loss_ = tf.keras.losses.BinaryCrossentropy()

        step = 0
        for epoch in tqdm(range(self.n_epoch)):
            if epoch and epoch % 1 == 0:
                print(epoch, loss)
                self.compute_metrics(step)
                pred = tf.sigmoid(tf.squeeze(self.tf_model(self.X_train)))
                print(pred[:5])
                print('auc', roc_auc_score(self.y_train, pred.numpy()))
                # print('loss', ll(self.y_train, pred), self.V.sum(), self.w.sum())

            for i_batch in range(self.n_batches):
                step += 1
                X_batch = self.X_train[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]
                y_batch = self.y_train[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]
            #for X_train, y_train in [(self.X_train, self.y_train)]:
                # print(X_train)
                '''for variable in self.tf_model.trainable_variables:
                    print(variable.name)'''
                
                if step > 0 and step % 50 == 0:
                    auc_1 = self.tf_auc(self.X_valid_1, self.y_valid_1)
                    auc_0 = self.tf_auc(self.X_valid_0, self.y_valid_0)
                    self.c += np.sign(auc_1 - auc_0) * 0.01
                    self.c = np.clip(self.c, -1, 1)
                # self.prepare_batch(i_batch)
                
                with tf.GradientTape() as tape:
                    # pred = tf.squeeze(self.tf_model(X_train, training=True))
                    # print('auc', roc_auc_score(y_train, pred.numpy()))
                    # loss = loss_(y_train, pred)
                    loss = self.tf_loss(X_batch, y_batch)
                    # loss = self.auc_loss(self.c, self.mu, self.w, self.V, self.item_bias, self.item_embed, self.item_slopes)
                    # print(loss)
            
                # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
                grads = tape.gradient(loss, self.tf_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.tf_model.trainable_variables))

    def full_fit(self):
        # pywFM and libFM
        # print('full fit', X.shape, y.shape)
        
        for step in tqdm(range(self.n_epoch)):
            if step % 10 == 0:
                self.compute_metrics(step)
                pred = self.predict(self.X_train)
                print('loss', ll(self.y_train, pred), self.V.sum(), self.w.sum())

            # self.mu -= self.GAMMA * grad(lambda mu: self.loss(X, y, mu, self.w, self.V))(self.mu)
            gradient = grad(lambda w: self.loss(self.mu, w, self.V, self.item_bias, self.item_embed, self.item_slopes))(self.w)
            # print('grad', gradient.shape, gradient)
            self.w -= self.GAMMA * gradient
            self.item_bias -= self.GAMMA * grad(lambda item_bias: self.loss(self.mu, self.w, self.V, item_bias, self.item_embed, self.item_slopes))(self.item_bias)
            self.item_slopes -= self.GAMMA * grad(lambda item_slopes: self.loss(self.mu, self.w, self.V, self.item_bias, self.item_embed, item_slopes))(self.item_slopes)
            
            if self.GAMMA_V:
                grad_v = grad(lambda V: self.loss(self.mu, self.w, V, self.item_bias, self.item_embed, self.item_slopes))(self.V)
                # print(grad_v)
                self.V -= self.GAMMA_V * grad_v
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

    '''
    def predict_logits(self, X, mu=None, w=None, V=None, item_bias=None, item_embed=None, slopes=None):
        return self.tf_model.predict(X)
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

        # y_pred = mu + w[users] + item_bias[items] + attempts * slopes[items]
        # y_pred = np.concatenate((self.users[users], self.items[items]), axis=1)
        # print(y_pred.shape)
        # print(V.shape)
        # print(w.shape)
        # print(item_bias.shape)
        # return relu(y_pred @ V + w) @ item_bias
        """if self.d > 0:
            y_pred += np.sum(V[users] * item_embed[items], axis=1)"""

        
        
        return y_pred'''

    def predict(self, X, mu=None, w=None, V=None, item_bias=None, item_embed=None, item_slopes=None):
        if mu is None:
            mu = self.mu
            w = self.w
            V = self.V
            item_bias = self.item_bias
            item_embed = self.item_embed
            item_slopes = self.item_slopes
        # return self.tf_model.predict(X)
        return tf.sigmoid(tf.squeeze(self.tf_model.predict(X)))

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
        
        # loss = self.loss(mu, w, V, bias, embed, slopes)
        
        auc_1 = self.relaxed_auc('batch', '_1', mu, w, V, bias, embed, slopes)
        auc_0 = self.relaxed_auc('batch', '_0', mu, w, V, bias, embed, slopes)
        return -auc  # Optimize overall AUC
        # return -auc_1  # Only optimize AUC of subgroup
        # return (auc_1 - auc_0) ** 2  # Only minimize delta AUC
        # return -auc - auc_1
        # return -auc_1 - auc_0
        # return -auc + self.fair * np.abs(auc_1 - auc_0) + self.LAMBDA * (mu ** 2 + np.sum(w ** 2) + np.sum(V ** 2) + np.sum(bias ** 2) + np.sum(embed ** 2))
        # return 100 * loss + self.fair * np.abs(auc_1 - auc_0)
        # print(100 * loss)
        # print(auc_1 - auc_0) ** 2
        # return loss + self.fair * (auc_1 - auc_0) ** 2
        # return loss + self.fair * c * (auc_1 - auc_0)
        # return -auc + self.fair * (auc_1 - auc_0) ** 2 + self.LAMBDA * (mu ** 2 + np.sum(w ** 2) + np.sum(V ** 2) + np.sum(bias ** 2) + np.sum(embed ** 2))
        # return -auc + self.fair * self.c * (auc_1 - auc_0) + self.LAMBDA * (mu ** 2 + np.sum(w ** 2) + np.sum(V ** 2) + np.sum(bias ** 2) + np.sum(embed ** 2))

    def tf_loss(self, X, y):
        auc = self.tf_auc(X, y)
        z_1 = X[:, 2] == 1
        z_0 = X[:, 2] == 0
        auc_1 = self.tf_auc(X[z_1], y[z_1])
        auc_0 = self.tf_auc(X[z_0], y[z_0])
        # print(auc, auc_1, auc_0)
        return -auc + (auc_1 - auc_0) ** 2
        # return -auc + self.c * (auc_1 - auc_0)

    # @tf.function
    def tf_auc(self, X, y, B=100):
        pos = y == 1
        neg = y == 0
        if len(y[pos]) == 0 or len(y[neg]) == 0:
            return 0.
        ii = tf.random.uniform([B], minval=0, maxval=len(y[pos]), dtype=tf.int32)
        jj = tf.random.uniform([B], minval=0, maxval=len(y[neg]), dtype=tf.int32)
        kk = np.random.choice(len(y[pos]), B).astype(np.int32)
        pred = tf.squeeze(self.tf_model(X, training=True))
        pred_1 = tf.gather(pred[pos], ii)
        pred_0 = tf.gather(pred[neg], jj)
        return tf.reduce_mean(tf.sigmoid(pred_1 - pred_0))
        test = X[:5]
        print(test)
        print(test[:, 2] == 0)
        print(test[test[:, 2] == 0])
        print(y[y == 1])
        sys.exit(0)
        return tf.reduce_sum(pred)

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
    print(df['user_id'].min(), df['user_id'].max(), len(np.unique(df['user_id'])))
    print(df['item_id'].min(), df['item_id'].max())
    X[:, 1] += df['user_id'].max() + 1
    # sys.exit(0)
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
        elif options.training == 'deep':
            # ofm.deep_fit()
            ofm.tf_fit()
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
        ofm.y_pred = y_pred.numpy().tolist()  # Save for future use
        print(ofm.X_test[:5])
        print(ofm.y_test[:5])
        print(ofm.y_pred[:5])
        print('test auc', roc_auc_score(ofm.y_test, y_pred))

        test = df.iloc[i_['test']]

        plt.show()
        
        if options.online:
            ofm.update()
        if len(ofm.X_test) > 10000:
            ofm.save_results(vars(options), test)
