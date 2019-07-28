# Feel free to modify this file
# It was intended to reproduce Choffin et al.'s experiments

import pandas as pd
import numpy as np

# small = pd.read_csv('notebooks/one_user.csv').sort_values('timestamp')
df = pd.read_csv(  # 3.01 Gigowatts!!!
    'assistments12/2012-2013-data-with-predictions-4-final.csv')
df['timestamp'] = pd.to_datetime(df['start_time']).map(
    lambda t: t.timestamp()).round().astype(np.int32)
df['skill_id'] = df['skill_id'].astype(pd.Int64Dtype())

# Keep only users with at least 10 interactions
# df = df.groupby("user_id").filter(lambda x: len(x) >= 10)  # The Choffin Way
nb_interactions_per_user = Counter(df['user_id'])
df['nb_interactions'] = df['user_id'].map(nb_interactions_per_user)
df = df.query('nb_interactions >= 10')

# Remove entries where skill_id is NaN
df = df[~df["skill_id"].isnull()]

# Remove entries where outcome is not binary but continuous
df = df[df.correct.isin([0, 1])]
df['correct'] = df['correct'].astype(np.int32)

df = df.sort_values('timestamp').rename(columns={'problem_id': 'item_id'})
print('Loading data', time.time() - dt)
dt = time.time()
df[['user_id', 'item_id', 'skill_id', 'timestamp', 'correct']].to_csv(
    'assistments12/needed.csv', index=None)
print('Save data', time.time() - dt)
