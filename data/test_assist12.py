from collections import defaultdict
import pandas as pd
import numpy as np


# At the time of running, there was a skill field in needed.csv; not anymore
full = pd.read_csv('assistments12/needed.csv')  # "Only" 326.3 MB

is_diff = (np.array(np.isnan(full['skill_id'])) !=
           np.array(full['skill'] != full['skill']))
# is_diff.sum() == 81733: not great

skills = defaultdict(set)
for skill_id, skill in zip(full['skill_id'].fillna(-1),
                           full['skill'].fillna(-1)):
    skills[skill_id].add(skill)
for skill_id in skills:
    if len(skills[skill_id]) > 1:
        print('oh', skill_id, len(skills[skill_id]))
# Conclusion: there are some skill_ids where skill is NA (no name)

# Check that 1 item is only associated to 1 skill: OK
# qm = defaultdict(set)
# for item, skill in zip(full['problem_id'], full['skill']):
#     qm[item].add(skill)
# print(len(qm), 'items')
# for item in qm:
#     if len(qm[item]) > 1:
#         print('oh', item, len(qm[item]))
