from scipy.sparse import coo_matrix, save_npz, load_npz, hstack
import pandas as pd
import numpy as np
import argparse
import yaml
import os


parser = argparse.ArgumentParser(description='Encode datasets')
parser.add_argument('--dataset', type=str, nargs='?', default='dummy')
parser.add_argument('--users', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--items', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--skills', type=bool, nargs='?', const=True,
                    default=False)
parser.add_argument('--wins', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--fails', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()

os.chdir(os.path.join('data', options.dataset))  # Move to dataset folder
all_features = ['users', 'items', 'skills', 'wins', 'fails']
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
    X['skills'] = onehotize(df['skill'], config['nb_skills'])
    X['wins'] = X['skills'].copy()
    X['wins'].data = df['wins']
    X['fails'] = X['skills'].copy()
    X['fails'].data = df['fails']
    X_train = hstack([X[agent] for agent in active_features]).tocsr()
    y_train = df['correct'].values
    return X_train, y_train


df = pd.read_csv('data.csv')
with open('config.yml') as f:
    config = yaml.load(f)
print('Configuration', config)
X, y = df_to_sparse(df, config, active_features)
print(df.head())
if options.dataset == 'dummy':
    print(X.todense())

save_npz('X-{:s}.npz'.format(features_suffix), X)
np.save('y-{:s}.npy'.format(features_suffix), y)
