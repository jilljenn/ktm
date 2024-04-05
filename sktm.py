"""
Efficient implementation of knowledge tracing machines using scikit-learn.

Currently: will not work on wins and fails > 1.
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
    ('lr', LogisticRegression(solver='liblinear', C=1e-1, max_iter=300))
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

print('Full training PFA')
pipe.fit(df[['skill', 'wins', 'fails']], df['correct'])
print(pipe['lr'].coef_)

print('Full training UISWF')
pipe.fit(df[['user', 'item', 'skill', 'wins', 'fails']], df['correct'])

# Test for dummy dataset
coef = pipe['lr'].coef_[0]
print(coef.shape)
print(coef)
print(df.nunique())
nb = [5, 2, 2, 1, 2]
print(sum(nb))
print('Users', coef[:5])
print('Items', coef[5:7])
print('Skills', coef[7:9])
print('Wins', coef[9:10])
print('Fails', coef[10:12])
