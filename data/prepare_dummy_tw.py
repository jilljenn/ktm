import pandas as pd
import numpy as np


df = pd.read_csv('dummy_tw/preprocessed_data.csv', sep='\t')
df['timestamp'] *= 3600 * 24  # I prefer seconds
(df[['user_id', 'item_id', 'timestamp', 'correct']]
    .astype(np.int32)
    .to_csv('dummy_tw/needed.csv'))
