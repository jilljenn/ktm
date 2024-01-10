"""
Efficient implementation of knowledge tracing machines using scikit-learn.
"""
import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GroupShuffleSplit
import pandas as pd


parser = argparse.ArgumentParser(description='Run simple KTM')
parser.add_argument(
    'csv_file', type=str, nargs='?', default='data/dummy/data.csv')
parser.add_argument('--model', type=str, nargs='?', default='iswf')
options = parser.parse_args()


df = pd.read_csv(options.csv_file)
pipe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('lr', LogisticRegression(solver='liblinear'))
])


cv = GroupShuffleSplit(n_splits=5, random_state=42)
METRICS = ['accuracy', 'roc_auc', 'neg_log_loss']
if options.model == 'irt':
    FEATURES = ['user', 'item']
elif options.model == 'pfa':
    FEATURES = ['skill', 'wins', 'fails']
else:
    FEATURES = ['item', 'skill', 'wins', 'fails']

cv_results = cross_validate(
    pipe, df[FEATURES], df['correct'],
    scoring=METRICS,  # Use all scores
    return_train_score=True, n_jobs=-1,  # Use all cores
    cv=cv, groups=df['user'], verbose=10
)
for metric in METRICS:
    print(metric, cv_results[f"test_{metric}"].mean())


for i_train, i_test in cv.split(df, groups=df['user']):
    df_train = df.iloc[i_train]
    df_test = df.iloc[i_test]

    # IRT
    pipe.fit(df_train[['user', 'item']], df_train['correct'])
    print(pipe.predict_proba(df_test[['user', 'item']])[:, 1])

    # PFA
    pipe.fit(df_train[['skill', 'wins', 'fails']], df_train['correct'])
    print(pipe.predict_proba(df_test[['skill', 'wins', 'fails']])[:, 1])
    break
