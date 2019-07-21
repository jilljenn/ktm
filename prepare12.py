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
import os

NB_TIME_WINDOWS = 5
NB_FOLDS = 5

parser = argparse.ArgumentParser(description='Prepare ASSISTments 2012 data')
parser.add_argument('--tw', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()

# dt = time.time()
# small = pd.read_csv('notebooks/one_user.csv').sort_values('timestamp')
# full = pd.read_csv(  # 3.01 Gigowatts!!!
#     'data/assistments12/2012-2013-data-with-predictions-4-final.csv')
# full['timestamp'] = pd.to_datetime(full['start_time']).map(
#     lambda t: t.timestamp()).round().astype(np.int32)
# full['skill_id'] = full['skill_id'].astype(pd.Int64Dtype())
# full['correct'] = full['correct'].astype(np.int32)
# full = full.sort_values('timestamp')
# print('Loading data', time.time() - dt)
# dt = time.time()
# full[['user_id', 'problem_id', 'skill_id', 'timestamp', 'correct']].to_csv(
#     'data/assistments12/needed.csv', index=None)
# print('Save data', time.time() - dt)

dt = time.time()
full = pd.read_csv('data/assistments12/needed.csv')  # "Only" 176.7 MB
full['skill_id'] = full['skill_id'].astype(pd.Int64Dtype())
print(full.head())

full['problem_id'] += full['user_id'].max()
full['skill_id'] += full['problem_id'].max()
full['i'] = range(nb_samples)
print('Loading data', time.time() - dt)
skill_values = full['skill_id'].dropna().unique()
print(len(full['user_id'].unique()), 'users',
      len(full['problem_id'].unique()), 'problems',
      len(skill_values), 'skills')
nb_samples, _ = full.shape

if not os.path.isfile('data/assistments12/fold0.npy'):
    dt = time.time()
    all_users = full['user_id'].unique()
    random.shuffle(all_users)
    fold_size = len(all_users) // NB_FOLDS
    everything = []
    for i in range(NB_FOLDS):
        if i < NB_FOLDS - 1:
            ids_of_fold = set(all_users[i * fold_size:(i + 1) * fold_size])
        else:
            ids_of_fold = set(all_users[i * fold_size:])
        fold = full.query('user_id in @ids_of_fold').index
        np.save('data/assistments12/fold{}.npy'.format(i), fold)
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
print(Counter(type(v) for v in codes2.keys()))

dt = time.time()
# Extra codes for counters within time windows (wins, attempts)
extra_codes = dict(zip([(field, value, pos)
                        for value in skill_values
                        for pos in range(NB_TIME_WINDOWS)
                        for field in {'wins', 'attempts'}],
                       range(len(codes2), len(codes2) + 1000000)))
print(len(codes) + len(extra_codes), 'features', time.time() - dt)
print(max(codes.values()), len(codes2))

convert = np.vectorize(codes2.get)
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
data = [1] * (nb_samples)
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
        add(i_sample, codes2[skill_id], 1)
        counters = q[user, skill_id].push(t)
        for pos, value in enumerate(counters):
            add(i_sample, codes['attempts', skill_id, pos], 1 + log(value))
        if correct:
            counters = q[user, skill_id, 'correct'].push(t)
            for pos, value in enumerate(counters):
                add(i_sample, codes['wins', skill_id, pos], 1 + log(value))

    print(len(q), 'queues', time.time() - dt)
    print('Total', len(rows), 'entries')

dt = time.time()
X = csr_matrix((data, (rows, cols)))
print('Into sparse matrix', time.time() - dt)
y = np.array(full['correct'])
print(X.shape, y.shape)
dt = time.time()
# save_npz('/Users/jilljenn/code/ktm/data/assistments12/X-das3h.npz', X)
# np.save('/Users/jilljenn/code/ktm/data/assistments12/y-das3h.npy', y)
print('Saving', time.time() - dt)
