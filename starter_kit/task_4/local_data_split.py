'''
this script splits the training set into two subsets for local evaluation

input: training split of the competition data
output: a 80-20 split of the input data. the ratio can be manually changed in the script
'''

import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(123)
from pdb import set_trace
from pathlib import Path


PATH = Path('../../data/neurips')

pd.options.mode.chained_assignment = None

data = pd.read_csv(PATH / 'train_data/train_task_3_4.csv')

user_ids = data['UserId'].unique()
random.shuffle(user_ids)
n_rows = len(user_ids)
train_users = user_ids[0: (n_rows * 8)//10]
valid_users = user_ids[(n_rows * 8)//10:]

train_df = data.loc[data['UserId'].isin(train_users)]
valid_df = data.loc[data['UserId'].isin(valid_users)]

# Add target indicator to validation dataframe to indicate held-out questions when evaluation question selection
valid_df['IsTarget'] = np.random.randint(1, 11, valid_df.shape[0])
valid_df['IsTarget'] = (valid_df['IsTarget']<=2).astype(int)

if not os.path.isdir(PATH / 'test_input'):
    os.mkdir(PATH / 'test_input')

train_df.to_csv(PATH / 'test_input/train_task_4.csv', index=False)
valid_df.to_csv(PATH / 'test_input/valid_task_4.csv', index=False)
