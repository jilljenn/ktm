from this_queue import OurQueue
from collections import defaultdict, Counter
from scipy.sparse import load_npz, save_npz, csr_matrix, find
from dataio import save_folds
from itertools import product
from math import log
import pandas as pd
import numpy as np
import argparse
import time
import sys
import os

NB_TIME_WINDOWS = 5
NB_FOLDS = 5

parser = argparse.ArgumentParser(description='Prepare data for DAS3H')
parser.add_argument('--dataset', type=str, nargs='?', default='das3h')
parser.add_argument('--no_na', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--tw', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()

dt = time.time()
os.chdir('data/{}'.format(options.dataset))
# full = pd.read_csv('preprocessed_data.csv', sep='\t')  # "Only" 176.7 MB
# full['timestamp'] *= 3600 * 24
# full = pd.read_csv('needed.csv')  # "Only" 176.7 MB
full = pd.read_csv('original-das3h.csv')  # Only 81 MB

# if options.no_na:  # Drop entries where there is no skill
#     full = full.dropna(subset=['skill_id']).reset_index()
# else:
#     full['skill_id'] = full['skill_id'].astype(pd.Int64Dtype())

nb_samples, _ = full.shape
shift_items = 1 + full['user_id'].max()  # We shift IDs to ensure uniqueness
full['item_id'] += shift_items
shift_skills = int(1 + full['item_id'].max())
q_mat = defaultdict(list)
if 'skill_id' in full.columns:
    # Actually, we should build a simple q_mat
    full['skill_id'] += shift_skills
    for item_id, skill_id in dict(zip(full['item_id'],
                                      full['skill_id'])).items():
        q_mat[item_id].append(skill_id)
else:
    q_matrix = load_npz('q_mat.npz')
    print(q_matrix.toarray())
    _, nb_skills = q_matrix.shape
    rows, cols, _ = find(q_matrix)
    for i, j in zip(rows, cols):
        q_mat[shift_items + i].append(shift_skills + j)

full['i'] = range(nb_samples)
print('Loading data:', nb_samples, 'samples', time.time() - dt)
print(full.head())

all_values = {}
for col in {'user_id', 'item_id', 'skill_id'}:
    if col in full.columns:
        all_values[col] = full[col].dropna().unique()
    else:
        all_values['skill_id'] = list(range(shift_skills,
                                            shift_skills + nb_skills))

# Create folds of indices and save them
if not os.path.isfile('folds/{}fold0.npy'.format(nb_samples)):
    save_folds(full, NB_FOLDS)

conversion = {
    'user_id': 'user',
    'item_id': 'item',
    'skill_id': 'kc'
}

# Preprocess codes
dt = time.time()
codes = dict(zip([value for field, key in conversion.items()
                  for value in all_values[field]], range(1000000)))
print('Preprocess codes', time.time() - dt)

dt = time.time()
# Extra codes for counters within time windows (wins, attempts)
extra_codes = dict(zip([(field, value, pos)
                        for value in all_values['skill_id']
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
    # df = full.dropna(subset=['skill_id'])
    df = full

    dt = time.time()
    # Prepare counters for time windows
    q = defaultdict(lambda: OurQueue())
    # Thanks to this zip tip, it's faster https://stackoverflow.com/a/34311080
    for i_sample, user, item_id, t, correct in zip(
           df['i'], df['user'], df['item_id'], df['timestamp'], df['correct']):
        for skill_id in q_mat[item_id]:
            add(i_sample, codes[skill_id], 1)
            for pos, value in enumerate(q[user, skill_id].get_counters(t)):
                if value:
                    add(i_sample, extra_codes['attempts', skill_id, pos],
                        log(1 + value))
            for pos, value in enumerate(q[user, skill_id, 'correct']
                                        .get_counters(t)):
                if value:
                    add(i_sample, extra_codes['wins', skill_id, pos],
                        log(1 + value))
            q[user, skill_id].push(t)
            if correct:
                q[user, skill_id, 'correct'].push(t)

    print(len(q), 'queues', time.time() - dt)
    print('Total', len(rows), 'entries')

dt = time.time()
X = csr_matrix((data, (rows, cols)))
y = np.array(full['correct'])
print('Into sparse matrix', X.shape, y.shape, time.time() - dt)
dt = time.time()
save_npz('X.npz', X)
np.save('y.npy', y)
print('Saving', time.time() - dt)
