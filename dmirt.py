import keras
from keras import regularizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Flatten, Add, Activation, Dot, Input, dot, add, concatenate, Lambda, multiply
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import glob
import os
import sys


'''
df = pd.read_csv('ratings-full.csv')
n_users = df.user.nunique()
n_items = df.item.nunique()
X = np.array(df[['user', 'item']])
y = np.array(df['label'])
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, shuffle=True,
    test_size=0.2)
# print([c.shape for c in [X_trainval, y_trainval, X_test, y_test]])
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval,
    shuffle=True, test_size=0.2)
'''


parser = argparse.ArgumentParser(description='Run DMIRT')
parser.add_argument('X_file', type=str, nargs='?', default='dummy')
parser.add_argument('--d', type=int, nargs='?', default=1)
options = parser.parse_args()


X_file = options.X_file
folder = os.path.dirname(X_file)
df = pd.read_csv(X_file)
print('user', df.user_id.min(), df.user_id.max())
print('item', df.item_id.min(), df.item_id.max())
# df['item_id'] += df.user_id.max() + 1
n_users = df.user_id.nunique()
n_items = df.item_id.nunique()
# sys.exit(0)

X = np.array(df[['user_id', 'item_id']])
y = np.array(df['correct'])
nb_samples = len(y)
folds = sorted(glob.glob(os.path.join(folder, 'folds/60weak{}fold*.npy'.format(nb_samples))))
valids = sorted(glob.glob(os.path.join(folder, 'folds/36weak{}valid*.npy'.format(nb_samples))))


def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def make_ones(x):
    return K.reshape(K.ones_like(x), (-1, 1))


for i, (filename, valid) in enumerate(zip(folds, valids)):
    i_test = set(np.load(filename))
    i_valid = set(np.load(valid))
    i_train = set(range(nb_samples)) - i_test - i_valid

    for dataset in {'train', 'valid', 'test'}:
        indices = list(globals()['i_' + dataset])
        print(X[indices, 0].shape)
        # sys.exit(0)
        globals()['X_' + dataset] = [X[indices, 0], X[indices, 1]]
        globals()['y_' + dataset] = y[indices]

    n_dim = options.d

    users = Input(shape=(1,))
    items = Input(shape=(1,))
    # ones = Input(tensor=tf.reshape(tf.ones_like(users), (-1, 1)))
    ones = Lambda(make_ones)(users)
    print(ones)

    # user_bias = Embedding(n_users, 1)(users)
    # item_bias = Embedding(n_items, 1)(items)
    user_embed = Flatten()(Embedding(n_users, 1 + n_dim)(users))
    user_embed = concatenate([ones, user_embed])
    item_embed = Flatten()(Embedding(n_items, 1 + n_dim)(items))
    item_embed = concatenate([item_embed, ones])
    product = multiply([user_embed, item_embed])

    features = concatenate([user_embed, item_embed, product])
    hidden = Dense(2 * n_dim, activation='relu')(features)
    logit = Dense(1)(hidden)
    
    # logit = dot([user_embed, item_embed], axes=-1)
    # logit = add([user_bias, item_bias, pairwise])
    pred = Activation('sigmoid')(logit)

    # out = Dense(1, activation='sigmoid')(logit)
    # model = Model([multi.input, irt.input], outputs=out)
    model = Model([users, items], outputs=pred)

    print(model.summary())

    plot_model(model, to_file='model.png')

    # print(model.predict(X_train).shape)
    # sys.exit(0)

    model.compile(
        loss=keras.losses.binary_crossentropy,
        # optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True),
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy', auroc])

    es = keras.callbacks.EarlyStopping(patience=3)

    print(X[list(i_train)][:5])
    print(X_train[0].shape, X_train[0][:5])
    print(X_train[1].shape, X_train[1][:5])
    # sys.exit(0)

    model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid),
              epochs=150, batch_size=10000, callbacks=[es])

    print(model.evaluate(X_test, y_test))
