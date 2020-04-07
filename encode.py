"""
Encoding features for Knowledge Tracing Machines.
Select a <dataset> and the features you want to include.

Case 1: There is only one skill per item.
=======
data/<dataset>/data.csv should contain the following columns:
user, item, skill, correct, wins, fails
where wins and fails are the number of successful and unsuccessful
attempts at that skill.

Case 2: There may be several skills associated to an item.
=======
data/<dataset>/data.csv just needs to contain:
user, item, correct
and data/<dataset>/q_mat.npz should be a q-matrix under scipy.sparse format.
If you want to compute wins and fails like in PFA,
you should run encode_tw.py instead of this file, with the --pfa option.

If you want to add extra side information, you can be inspired by
--tutor or --answer below.

Note: IDs for users and items need not be disjoint,
as we use hstack below to concatenate sparse matrices.

See paper: https://arxiv.org/abs/1811.03388

Authors: Jill-Jênn Vie, 2020
"""
from scipy.sparse import coo_matrix, save_npz, load_npz, hstack
from collections import Counter
import pandas as pd
import numpy as np
import argparse
import yaml
import os
import sys


parser = argparse.ArgumentParser(description='Encode datasets')
parser.add_argument('--dataset', type=str, nargs='?', default='dummy')
parser.add_argument('--users', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--items', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--skills', type=bool, nargs='?', const=True,
                    default=False)
parser.add_argument('--wins', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--fails', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--tutor', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--answer', type=bool, nargs='?', const=True,
                    default=False)
options = parser.parse_args()

os.chdir(os.path.join('data', options.dataset))  # Move to dataset folder
all_features = ['users', 'items', 'skills', 'wins', 'fails', 'tutor', 'answer']
active_features = [features for features in all_features
                   if vars(options)[features]]
features_suffix = ''.join([features[0] for features in active_features])


def onehotize(col, depth):
    nb_events = len(col)
    rows = list(range(nb_events))
    return coo_matrix(([1] * nb_events, (rows, col)), shape=(nb_events, depth))


def df_to_sparse(df, config, active_features):
    '''
    Prepare sparse features
    '''
    X = {}
    X['users'] = onehotize(df['user'], config['nb_users'])
    X['items'] = onehotize(df['item'], config['nb_items'])
    X['tutor'] = onehotize(df['tutor'], len(df['tutor'].unique()))
    X['answer'] = onehotize(df['answer'], len(df['answer'].unique()))
    if 'skill' in df:
        X['skills'] = onehotize(df['skill'], config['nb_skills'])
        X['wins'] = X['skills'].copy()
        X['wins'].data = df['wins']
        X['fails'] = X['skills'].copy()
        X['fails'].data = df['fails']
    elif os.path.isfile('q_mat.npz'):
        q_matrix = load_npz('q_mat.npz')
        X['skills'] = q_matrix[df['item']]
        print('nb skills', Counter(X['skills'].sum(axis=1).A1))
    X_train = hstack([X[agent] for agent in active_features]).tocsr()
    y_train = df['correct'].values
    return X_train, y_train


df = pd.read_csv('data.csv')
with open('config.yml') as f:
    config = yaml.safe_load(f)
print('Configuration', config)
X, y = df_to_sparse(df, config, active_features)
print(df.head())
if options.dataset == 'dummy':
    print(X.todense())

save_npz('X-{:s}.npz'.format(features_suffix), X)
np.save('y-{:s}.npy'.format(features_suffix), y)
print('Successfully created X-{:s}.npz and y-{:s}.npy '
      'in data/{} folder'.format(
          features_suffix, features_suffix, options.dataset))
