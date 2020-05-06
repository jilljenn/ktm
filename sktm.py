from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd


df = pd.read_csv('data/dummy/data.csv')
estimators = [
    ('onehot', OneHotEncoder()),
    ('lr', LogisticRegression())
]
pipe = Pipeline(estimators)

# Just check the encoded variables
ohe = OneHotEncoder()
print(ohe.fit_transform(df[['user', 'item']]).toarray())

# IRT
pipe.fit(df[['user', 'item']], df['correct'])
print(pipe.predict_proba(df[['user', 'item']]))

# PFA
pipe.fit(df[['skill', 'wins', 'fails']], df['correct'])
print(pipe.predict_proba(df[['skill', 'wins', 'fails']]))
