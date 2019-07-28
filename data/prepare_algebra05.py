from collections import Counter
import pandas as pd
import numpy as np


df = (pd.read_csv('data/algebra05/algebra_2005_2006_train.txt',
                  sep='\t').dropna(subset=['Step Start Time', 'KC(Default)'])
                           .rename(columns={
                                    'Anon Student Id': 'user_id',
                                    'Correct First Attempt': 'correct'
                                   }))
df['item_id'] = df['Problem Name'] + '~' + df['Step Name']
print(df.head(5))

values = sorted(df['Step Start Time'].unique())
print(values[:5], values[-5:])

df['timestamp'] = pd.to_datetime(df['Step Start Time']).map(
    lambda t: t.timestamp()).round().astype(np.int32)

# Check if skills are a function of the item_id (answer: NO)
steps_with_kc = df.groupby(['item_id', 'KC(Default)']).size().reset_index()
print(Counter(steps_with_kc['item_id']).most_common(5))
print(steps_with_kc.query('`item_id` == "LIT59~b+r*(x+y) = v-s"'))

all_skills = set()
for skill in df['KC(Default)']:
    for token in skill.split('~~'):
        all_skills.add(token)
encode = dict(zip(all_skills, range(1000000)))

skill_ids = []
for skills in df['KC(Default)']:
    skill_ids.append('~~'.join(str(encode[skill])
                               for skill in skills.split('~~')))
df['skill_ids'] = skill_ids

df[['user_id', 'item_id', 'skill_ids', 'timestamp', 'correct']].to_csv(
    'data/algebra05/needed.csv')
