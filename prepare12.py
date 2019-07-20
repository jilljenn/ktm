from notebooks.this_queue import OurQueue
from collections import defaultdict
from scipy.sparse import load_npz, save_npz, csr_matrix
from itertools import product
from math import log
import pandas as pd
import numpy as np
import random
import time
import sys
import os


NB_TIME_WINDOWS = 5
NB_FOLDS = 5


dt = time.time()
# small = pd.read_csv('notebooks/one_user.csv').sort_values('timestamp')

# full = pd.read_csv(  # 3.01 Gigowatts!!!
#     'data/assistments12/2012-2013-data-with-predictions-4-final.csv')
# full[['user_id', 'problem_id', 'skill', 'start_time', 'correct']].to_csv(
#     'data/assistments12/needed.csv', index=None)
full = pd.read_csv('data/assistments12/needed.csv')  # "Only" 356.3 MB
print(full.head())
print(full.shape)
print(len(full['user_id'].unique()), 'users',
      len(full['problem_id'].unique()), 'problems',
      len(full['skill'].dropna().unique()), 'skills')

# Check that 1 item is only associated to 1 skill
# qm = defaultdict(set)
# for item, skill in zip(full['problem_id'], full['skill']):
#     qm[item].add(skill)
# print(len(qm), 'items')
# for item in qm:
#     if len(qm[item]) > 1:
#         print('oh', item, len(qm[item]))

full['timestamp'] = pd.to_datetime(full['start_time']).map(
    lambda t: t.timestamp())
full = full.sort_values('timestamp')
print('Loading data', time.time() - dt)
nb_samples, _ = full.shape
full['i'] = range(nb_samples)

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
    'skill': 'kc'
}

# Preprocess codes
dt = time.time()
codes = dict(zip([(key, value) for field, key in conversion.items()
                  for value in full[field].dropna().unique()], range(1000000)))
print('Preprocess codes', time.time() - dt)

# Extra codes for counters
dt = time.time()
for field, key in conversion.items():
    full[key] = full[field].map(lambda x: codes[key, x], na_action='ignore')
# Actually, I wonder if a direct map with a dictionary is faster
print('Get key (this part is longer)', time.time() - dt)

print(len(codes))
dt = time.time()
skill_values = full['skill'].dropna().unique()
keys = [(field, codes[conversion['skill'], value], pos)
        for value in skill_values
        for pos in range(NB_TIME_WINDOWS) for field in {'wins', 'attempts'}]
bonus = zip(keys, range(len(codes), len(codes) + 1000000))
codes.update(bonus)
print(len(codes), 'features/columns all codes', time.time() - dt)
print(max(codes.values()), len(codes))

dt = time.time()
rows = list(range(nb_samples))
rows += rows  # Initialize user, item
cols = list(full['user'])
cols.extend(full['item'])
data = [1] * (2 * nb_samples)
assert len(rows) == len(cols) == len(data)
print('Initialized', len(rows), 'entries', time.time() - dt)


# At this stage we can save a IRT baseline
# dt = time.time()
# X = csr_matrix((data, (rows, cols)))
# print('Into sparse matrix', time.time() - dt)
# print(X.shape)
# y = np.array(full['correct'])
# print(y.shape)
# save_npz('/Users/jilljenn/code/ktm/data/assistments12/X-ui.npz', X)
# np.save('/Users/jilljenn/code/ktm/data/assistments12/y-ui.npy', y)
# sys.exit(0)


def add(r, c, d):
    rows.append(r)
    cols.append(c)
    data.append(d)


df = full.dropna(subset=['kc'])

# Prepare counters for time windows
q = defaultdict(lambda: OurQueue())
# Thanks to this zip tip, it's faster https://stackoverflow.com/a/34311080
for i_sample, user, item, skill, t, correct in zip(
        df['i'], df['user'], df['item'], df['kc'].astype(np.int32),
        df['timestamp'], df['correct']):
    add(i_sample, skill, 1)
    counters = q[user, skill].push(t)
    for pos, value in enumerate(counters):
        add(i_sample, codes['attempts', skill, pos], 1 + log(value))
    if correct:
        counters = q[user, skill, 'correct'].push(t)
        for pos, value in enumerate(counters):
            add(i_sample, codes['wins', skill, pos], 1 + log(value))

print(len(q), 'queues')
print('Total', len(rows), 'entries')


X = csr_matrix((data, (rows, cols)))
y = np.array(full['correct'])
print(X.shape, y.shape)
save_npz('/Users/jilljenn/code/ktm/data/assistments12/X.npz', X)
np.save('/Users/jilljenn/code/ktm/data/assistments12/y.npy', y)
