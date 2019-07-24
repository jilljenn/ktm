from this_queue import OurQueue
from collections import defaultdict, Counter
from scipy.sparse import load_npz, save_npz, csr_matrix
from itertools import product
from math import log
import pandas as pd
import numpy as np
import argparse
import random
import time
import sys
import os

NB_TIME_WINDOWS = 5
NB_FOLDS = 5

parser = argparse.ArgumentParser(description='Prepare ASSISTments 2012 data')
parser.add_argument('--no_na', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--tw', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()

dt = time.time()
full = pd.read_csv('data/assistments12/needed.csv')  # "Only" 176.7 MB
# full = pd.read_csv('data/assistments12/original-das3h.csv')  # Only 81 MB

if options.no_na:  # Drop entries where there is no skill
    full = full.dropna(subset=['skill_id']).reset_index()
else:
    full['skill_id'] = full['skill_id'].astype(pd.Int64Dtype())
print(full.head())

nb_samples, _ = full.shape
full['problem_id'] += full['user_id'].max()  # To ease encoding
full['skill_id'] += full['problem_id'].max()  # We ensure uniqueness of IDs
full['i'] = range(nb_samples)
print('Loading data', time.time() - dt)

skill_values = full['skill_id'].dropna().unique()
nb_users = len(full['user_id'].unique())
print(nb_users, 'users',
      len(full['problem_id'].unique()), 'problems',
      len(skill_values), 'skills',
      nb_samples, 'samples')

# Create folds of indices and save them
if not os.path.isfile(
        'data/assistments12/folds/{}fold0.npy'.format(nb_samples)):
    dt = time.time()
    all_users = full['user_id'].unique()
    random.shuffle(all_users)
    fold_size = nb_users // NB_FOLDS
    everything = []
    for i in range(NB_FOLDS):
        if i < NB_FOLDS - 1:
            ids_of_fold = set(all_users[i * fold_size:(i + 1) * fold_size])
        else:
            ids_of_fold = set(all_users[i * fold_size:])
        fold = full.query('user_id in @ids_of_fold').index
        np.save('data/assistments12/folds/{}fold{}.npy'.format(nb_samples, i),
                fold)
        everything += list(fold)
    assert sorted(everything) == list(range(nb_samples))
    print('Save folds', time.time() - dt)

conversion = {
    'user_id': 'user',
    'problem_id': 'item',
    'skill_id': 'kc'
}

# Preprocess codes
dt = time.time()
codes = dict(zip([value for field, key in conversion.items()
                  for value in full[field].dropna().unique()], range(1000000)))
print('Preprocess codes', time.time() - dt)

dt = time.time()
# Extra codes for counters within time windows (wins, attempts)
extra_codes = dict(zip([(field, value, pos)
                        for value in skill_values
                        for pos in range(NB_TIME_WINDOWS)
                        for field in {'wins', 'attempts'}],
                       range(len(codes), len(codes) + 1000000)))
print(len(codes) + len(extra_codes), 'features', time.time() - dt)
print(max(codes.values()), len(codes))

convert = np.vectorize(codes.get)
for field, key in conversion.items():
    dt = time.time()
    if field != 'skill_id':  # Will not work because of NaN values
        full[key] = convert(full[field])
    print('Get key', key, time.time() - dt)
dt = time.time()
all_values = np.array(full[['user_id', 'problem_id']])
print('To np array', time.time() - dt, all_values.dtype)

dt = time.time()
rows = list(range(nb_samples))
rows += rows  # Initialize user, item
cols = list(full['user'])
cols.extend(full['item'])
data = [1] * (2 * nb_samples)
assert len(rows) == len(cols) == len(data)
print('Initialized', len(rows), 'entries', time.time() - dt)


def add(r, c, d):
    rows.append(r)
    cols.append(c)
    data.append(d)


if options.tw:
    df = full.dropna(subset=['skill_id'])

    dt = time.time()
    # Prepare counters for time windows
    q = defaultdict(lambda: OurQueue())
    # Thanks to this zip tip, it's faster https://stackoverflow.com/a/34311080
    for i_sample, user, item, skill_id, t, correct in zip(
            df['i'], df['user'], df['item'],
            df['skill_id'], df['timestamp'], df['correct']):
        add(i_sample, codes[skill_id], 1)
        for pos, value in enumerate(q[user, skill_id].get_counters()):
            if value:
                add(i_sample, extra_codes['attempts', skill_id, pos],
                    1 + log(value))
        for pos, value in enumerate(q[user, skill_id, 'correct']
                                    .get_counters()):
            if value:
                add(i_sample, extra_codes['wins', skill_id, pos],
                    1 + log(value))
        q[user, skill_id].push(t)
        if correct:
            q[user, skill_id, 'correct'].push(t)

    print(len(q), 'queues', time.time() - dt)
    print('Total', len(rows), 'entries')

dt = time.time()
X = csr_matrix((data, (rows, cols)))
print('Into sparse matrix', X.shape, y.shape, time.time() - dt)
y = np.array(full['correct'])
dt = time.time()
save_npz('/Users/jilljenn/code/ktm/data/assistments12/X-original-das3h.npz', X)
np.save('/Users/jilljenn/code/ktm/data/assistments12/y-original-das3h.npy', y)
print('Saving', time.time() - dt)
