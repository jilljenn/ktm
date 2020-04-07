from utils.this_queue import OurQueue
from collections import defaultdict, Counter
from scipy.sparse import load_npz, save_npz, csr_matrix, find
from dataio import save_folds, save_weak_folds
from itertools import product
from math import log
import pandas as pd
import numpy as np
import argparse
import time
import os

NB_TIME_WINDOWS = 5

parser = argparse.ArgumentParser(description='Prepare data for DAS3H')
parser.add_argument('--dataset', type=str, nargs='?', default='dummy_tw')
parser.add_argument('--tw', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--pfa', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()

dt = time.time()
os.chdir('data/{}'.format(options.dataset))
full = pd.read_csv('needed.csv')  # Only 176.7 MB for ASSISTments 2012 (3 GB)
# full = pd.read_csv('preprocessed_data.csv',sep="\t")
if 'skill_id' in full.columns:
    full['skill_id'] = full['skill_id'].astype(pd.Int64Dtype())  # Can be NaN

nb_samples, _ = full.shape
shift_skills = 0
if full['user_id'].dtype == np.int64:  # We shift IDs to ensure that
    shift_items = 1 + full['user_id'].max()  # user/item/skill IDs are distinct
    full['item_id'] += shift_items
    shift_skills = int(1 + full['item_id'].max())

# Handle skills (either q-matrix, or skill_id, or skill_ids from 0)
q_mat = defaultdict(list)
nb_skills = None
if 'skill_id' in full.columns:
    print('Found a column skill_id')
    full['skill_id'] += shift_skills
elif os.path.isfile('q_mat.npz'):
    print('Found a q-matrix')
    q_matrix = load_npz('q_mat.npz')
    _, nb_skills = q_matrix.shape
    rows, cols, _ = find(q_matrix)
    for i, j in zip(rows, cols):
        q_mat[shift_items + i].append(shift_skills + j)

full['i'] = range(nb_samples)
print('Loading data:', nb_samples, 'samples', time.time() - dt)
print(full.head())

all_values = {}
if nb_skills is None:
    nb_skills = 112  # Only way to know for Algebra 2005
for col in {'user_id', 'item_id', 'skill_id'}:
    if col in full.columns:
        all_values[col] = full[col].dropna().unique()
    else:
        all_values['skill_id'] = list(range(shift_skills,
                                            shift_skills + nb_skills))

# Create folds of indices and save them
if not os.path.isfile('folds/{}fold0.npy'.format(nb_samples)):
    save_folds(full)
save_folds(full)

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

# Extra codes for counters within time windows (wins, attempts)
extra_codes = dict(zip([(field, value, pos)
                        for value in all_values['skill_id']
                        for pos in range(NB_TIME_WINDOWS)
                        for field in {'wins', 'attempts'}],
                       range(len(codes), len(codes) + 1000000)))
print('Gather all', len(codes) + len(extra_codes), 'features')

convert = np.vectorize(codes.get)
for field, key in conversion.items():
    dt = time.time()
    if field != 'skill_id':  # Will not work because of potential NaN values
        full[key] = convert(full[field])
        print('Encode', key, time.time() - dt)

dt = time.time()
rows = list(range(nb_samples)) + list(range(nb_samples))  # User & Item
cols = list(full['user']) + list(full['item'])
data = [1] * (2 * nb_samples)
assert len(rows) == len(cols) == len(data)
print('Initialized', len(rows), 'entries', time.time() - dt)


def add(r, c, d):
    rows.append(r)
    cols.append(c)
    data.append(d)


def identity(x):
    return x


suffix = 'ui'
if options.tw:
    suffix = 'das3h'
    link_function = identity
elif options.pfa:
    suffix = 'swf'
    link_function = log

if options.tw or options.pfa:  # Build time windows features
    df = full
    if 'skill_id' in full.columns:
        df = df.dropna(subset=['skill_id'])
        df['skill_ids'] = df['skill_id'].astype(str)
    else:
        df['skill_ids'] = [None] * len(df)

    dt = time.time()
    # Prepare counters for time windows
    q = defaultdict(lambda: OurQueue(only_forever=options.pfa))
    # Using zip is the fastest way to iterate DataFrames
    # Source: https://stackoverflow.com/a/34311080
    for i_sample, user, item_id, t, correct, skill_ids in zip(
           df['i'], df['user'], df['item_id'], df['timestamp'], df['correct'],
           df['skill_ids']):
        for skill_id in q_mat[item_id] or skill_ids.split('~~'):  # Fallback
            skill_id = int(skill_id)
            add(i_sample, codes[skill_id], 1)
            for pos, value in enumerate(q[user, skill_id].get_counters(t)):
                if value > 0:
                    add(i_sample, extra_codes['attempts', skill_id, pos],
                        link_function(1 + value))
            for pos, value in enumerate(q[user, skill_id, 'correct']
                                        .get_counters(t)):
                if value > 0:
                    add(i_sample, extra_codes['wins', skill_id, pos],
                        link_function(1 + value))
            q[user, skill_id].push(t)
            if correct:
                q[user, skill_id, 'correct'].push(t)

    print('Run all', len(q), 'queues', time.time() - dt)
    print('Total', len(rows), 'entries')

dt = time.time()
X = csr_matrix((data, (rows, cols)))
y = np.array(full['correct'])
print('Into sparse matrix', X.shape, y.shape, time.time() - dt)
dt = time.time()
save_npz('X-{}.npz'.format(suffix), X)
np.save('y-{}.npy'.format(suffix), y)
print('Saving', time.time() - dt)
