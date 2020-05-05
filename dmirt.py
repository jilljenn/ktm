import keras
from keras import regularizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Flatten, Add, Activation, Dot, Input, dot, add, concatenate, Lambda, multiply, AveragePooling1D
from keras.utils import plot_model
from keras.constraints import NonNeg
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from dataio import load_folds
import os.path
import tensorflow as tf
import argparse
import glob
import os
import sys
from tensorboard.plugins import projector

def register_embedding(embedding_tensor_name, meta_data_fname, log_dir):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_tensor_name
    embedding.metadata_path = meta_data_fname
    projector.visualize_embeddings(log_dir, config)

def save_labels_tsv(labels, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        for label in labels:
            f.write('{}\n'.format(label))

LOG_DIR = 'tmp'  # Tensorboard log dir
META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
EMBEDDINGS_TENSOR_NAME = 'embeddings2'
EMBEDDINGS_FPATH = os.path.join(LOG_DIR, EMBEDDINGS_TENSOR_NAME + '.ckpt')
STEP = 0


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
parser.add_argument('--test', type=str, nargs='?', default='',
                    help='Path to numpy array containing indices of folds.')
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
test_folds, valid_folds = load_folds(folder, options, df)
print('test folds', test_folds)


def auroc(y_true, y_pred):
    '''
    Needed this workaround for Keras to compute AUC,
    But it computes it on batches so some batches do not have
    enough diversity (not all 1 nor not all 0).
    '''
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def make_ones(x):
    return K.reshape(K.ones_like(x), (-1, 1, 1))


for i, (filename, valid) in enumerate(zip(test_folds, valid_folds)):
    i_test = set(np.load(filename))
    i_valid = set(np.load(valid))
    i_train = set(range(nb_samples)) - i_test - i_valid

    for dataset in {'train', 'valid', 'test'}:
        indices = list(globals()['i_' + dataset])
        print(X[indices, 0].shape)
        # sys.exit(0)
        # For example, X_train is a list of the user_ids, and the item_ids
        globals()['X_' + dataset] = [X[indices, 0], X[indices, 1]]
        globals()['y_' + dataset] = y[indices]

    n_dim = options.d

    users = Input(shape=(1,))
    items = Input(shape=(1,))
    # ones = Input(tensor=tf.reshape(tf.ones_like(users), (-1, 1)))
    ones = Lambda(make_ones)(users)
    print('ones', ones)

    user_bias = Embedding(n_users, 1,
                          embeddings_regularizer=regularizers.l2(0.1 / nb_samples))(users)
    item_bias = Embedding(n_items, 1,
                          embeddings_regularizer=regularizers.l2(0.1 / nb_samples),
                          name='acquix')(items)
    
    user_embed = Embedding(n_users, n_dim,
                           embeddings_regularizer=regularizers.l2(0.1 / nb_samples))(users)
    # user_embed = concatenate([ones, user_embed])
    # print(user_embed)
    item_embed = Embedding(n_items, n_dim,
                           embeddings_constraint=NonNeg(),
                           embeddings_regularizer=regularizers.l2(0.1 / nb_samples))(items)
    # item_embed = concatenate([item_embed, ones])
    # product = multiply([user_embed, item_embed])
    # pairwise = Flatten()(AveragePooling1D(n_dim, data_format='channels_first')(product))
    # sys.exit(0)

    # features = concatenate([user_embed, item_embed, product])
    # hidden = product
    # hidden = Dense(2 * n_dim, activation='relu')(features)
    # logit = Dense(1, use_bias=False)(hidden)
    # logit = 
    # logit = Flatten()(add([item_bias, pairwise]))
    
    # logit = dot([user_embed, item_embed], axes=-1)
    logit = Flatten()(add([user_bias, item_bias]))
    pred = Activation('sigmoid')(logit)

    # out = Dense(1, activation='sigmoid')(logit)
    # model = Model([multi.input, irt.input], outputs=out)
    model = Model([users, items], outputs=pred)

    print(model.summary())
    # sys.exit(0)

    plot_model(model, to_file='model.png')

    # print(model.predict(X_train).shape)
    # sys.exit(0)

    model.compile(
        loss=keras.losses.binary_crossentropy,
        # optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True),
        optimizer=keras.optimizers.Adam(lr=0.01),
        metrics=['accuracy'])

    print(model)
    for layer in model.layers:
        print(layer.name, [c.shape for c in layer.get_weights()])
    # sys.exit(0)

    log_dir = 'tmp'
    '''with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
        np.savetxt(f, y_train[:100])'''
    
    es = keras.callbacks.EarlyStopping(patience=1)
    '''tb = keras.callbacks.TensorBoard(log_dir=log_dir,
                                     batch_size=1000,
                                     embeddings_freq=1,
                                     embeddings_layer_names=['acquix'],
                                     embeddings_metadata='metadata.tsv',
                                     embeddings_data=X_train[:100])'''

    print(X[list(i_train)][:5])
    print(X_train[0].shape, X_train[0][:5])
    print(X_train[1].shape, X_train[1][:5])
    # sys.exit(0)

    model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid),
              epochs=50, batch_size=1000, callbacks=[es])

    print('But why', X_test[0].shape)
    from collections import Counter
    print(Counter(y_test.tolist()))
    print(model.evaluate(X_test, y_test))

    y_pred = model.predict(X_test)
    print('Test AUC', roc_auc_score(y_test, y_pred))

    register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, LOG_DIR)
    weights = model.layers[2].get_weights()[0]
    print(weights.shape)
    save_labels_tsv(np.ones(len(weights)), META_DATA_FNAME, LOG_DIR)
    embeddings = tf.Variable(weights, name=EMBEDDINGS_TENSOR_NAME)
    
    saver = tf.compat.v1.train.Saver([embeddings])  # Must pass list or dict
    saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)
