import glob
from sklearn.metrics import roc_auc_score, ndcg_score
from collections import defaultdict
from scipy.stats import sem, t
import pandas as pd
import numpy as np
import json
import re
import os


def avgstd(l):
    '''
    Given a list of values, returns a 95% confidence interval
    if the standard deviation is unknown.
    '''
    n = len(l)
    mean = sum(l) / n
    if n == 1:
        return '%.3f' % round(mean, 3)
    std_err = sem(l)
    confidence = 0.95
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return '%.3f ± %.3f' % (round(mean, 3), round(h, 3))


if __name__ == '__main__':
    os.chdir('data/assist09')

    indices = np.load('folds/weak278607fold0.npy')
    print(len(indices))

    df = pd.read_csv('needed.csv')
    test = df.iloc[indices]

    predictions_per_user = defaultdict(lambda: defaultdict(list))
    metrics = defaultdict(list)

    r = re.compile(r'results-(.*).json')
    for filename in glob.glob('results*2020*'):
        m = r.search(filename)
        print(filename)
        dt = m.group(1)

        with open(filename) as f:
            results = json.load(f)
        fold = results['predictions'][0]['fold']
        y_pred = results['predictions'][0]['pred']
        y = results['predictions'][0]['y']

        for user, pred, true in zip(test['user_id'], y_pred, y):
            predictions_per_user[user]['pred'].append(pred)
            predictions_per_user[user]['y'].append(true)

        for user in predictions_per_user:
            this_pred = np.array(predictions_per_user[user]['pred'])
            this_true = np.array(predictions_per_user[user]['y'])
            if len(this_pred) > 1:
                metrics['ndcg'].append(ndcg_score([this_true], [this_pred]))
                metrics['ndcg@10'].append(ndcg_score([this_true], [this_pred], k=10))
                metrics['ndcg-'].append(ndcg_score([1 - this_true], [1 - this_pred]))
                metrics['ndcg@10-'].append(ndcg_score([1 - this_true], [1 -this_pred], k=10))
    
        print(len(y))
        print(y[:10], test[:10])

        print('auc', roc_auc_score(y, y_pred))
        print('ndcg', avgstd(metrics['ndcg']))
        print('ndcg@10', avgstd(metrics['ndcg@10']))
        print('ndcg-', avgstd(metrics['ndcg-']))
        print('ndcg@10-', avgstd(metrics['ndcg@10-']))
