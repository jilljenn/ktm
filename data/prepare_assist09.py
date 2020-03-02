import numpy as np
import pandas as pd
from scipy import sparse
import argparse

parser = argparse.ArgumentParser(description='Prepare datasets.')
parser.add_argument('--min_interactions', type=int, nargs='?', default=10)
parser.add_argument('--remove_nan_skills', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--slicing_friendly', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()


assist09 = pd.read_csv("assist09/skill_builder_data_corrected_collapsed.csv",
						encoding = "ISO-8859-1", index_col=False)
assist09.drop(['Unnamed: 0'], axis=1, inplace=True)
timestamps = pd.read_csv("assist09/timestamp_data.csv")

assist09_w_time = assist09.merge(timestamps, left_on="order_id", right_on="problem_log_id",
								 how="inner")

assist09_w_time["timestamp"] = assist09_w_time["start_time"]
assist09_w_time["timestamp"] = pd.to_datetime(assist09_w_time["timestamp"])
assist09_w_time["timestamp"] = assist09_w_time["timestamp"] - assist09_w_time["timestamp"].min()
assist09_w_time["timestamp"] = assist09_w_time["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
assist09_w_time.sort_values(by="timestamp", inplace=True)
assist09_w_time.reset_index(inplace=True, drop=True)

# Filter out users that have less than min_interactions interactions
assist09_w_time = assist09_w_time.groupby("user_id").filter(lambda x: len(x) >= options.min_interactions)

# Remove NaN skills
if options.remove_nan_skills:
	assist09_w_time = assist09_w_time[~assist09_w_time["skill_id"].isnull()]
else:
    assist09_w_time.loc[assist09_w_time["skill_id"].isnull(), "skill_id"] = -1

if options.slicing_friendly:
	assist09_w_time = assist09_w_time[['user_id', 'problem_id', 'skill_id', 'correct', 'timestamp', 'school_id', 'teacher_id']]
	assist09_w_time["school_id"] = np.unique(assist09_w_time["school_id"], return_inverse=True)[1]
	assist09_w_time["teacher_id"] = np.unique(assist09_w_time["teacher_id"], return_inverse=True)[1]
else:
	assist09_w_time = assist09_w_time[['user_id', 'problem_id', 'skill_id', 'correct', 'timestamp']]

assist09_w_time["item_id"] = np.unique(assist09_w_time["problem_id"], return_inverse=True)[1]
assist09_w_time["user_id"] = np.unique(assist09_w_time["user_id"], return_inverse=True)[1]

assist09_w_time.reset_index(inplace=True, drop=True)

# Build q-matrix
listOfKC = []
for kc_raw in assist09_w_time["skill_id"].unique():
    for elt in str(kc_raw).split('_'):
        listOfKC.append(str(int(float(elt))))
listOfKC = np.unique(listOfKC)

dict1_kc = {} ; dict2_kc = {}
for k, v in enumerate(listOfKC):
    dict1_kc[v] = k
    dict2_kc[k] = v

# Build Q-matrix
Q_mat = np.zeros((len(assist09_w_time["item_id"].unique()), len(listOfKC)))
item_skill = np.array(assist09_w_time[["item_id","skill_id"]])
for i in range(len(item_skill)):
    splitted_kc = str(item_skill[i,1]).split('_')
    for kc in splitted_kc:
        Q_mat[item_skill[i,0],dict1_kc[str(int(float(kc)))]] = 1
assist09_w_time.drop(['skill_id','problem_id'], axis=1, inplace=True)
assist09_w_time = assist09_w_time[assist09_w_time.correct.isin([0,1])] # Remove potential continuous outcomes
assist09_w_time['correct'] = assist09_w_time['correct'].astype(np.int32) # Cast outcome as int32

# Save data
sparse.save_npz("assist09/q_mat.npz", sparse.csr_matrix(Q_mat))
assist09_w_time.to_csv("assist09/preprocessed_data.csv", sep="\t", index=False)

