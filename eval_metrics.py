import glob
from sklearn.metrics import roc_auc_score, ndcg_score
from collections import defaultdict
from scipy.stats import sem, t
import pandas as pd
import numpy as np
import json
import re
import os

SENSITIVE_ATTR = "school_id"

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
    # os.chdir('data/assist09')
    os.chdir('data/fr_en')

# indices = np.load('folds/278607fold0.npy')
indices = np.load('folds/weak926646fold0.npy')
print(len(indices))

df = pd.read_csv('preprocessed_data.csv',sep="\t")
test = df.iloc[indices]

#predictions_per_user = defaultdict(lambda: defaultdict(list))
#predictions_per_sensitive_attr = defaultdict(lambda: defaultdict(list))
#metrics_per_user = defaultdict(list)
#metrics_per_sensitive_attr = defaultdict(list)

r = re.compile(r'results-(.*).json')

#for filename in glob.glob('results*2020*'):
for filename in sorted(glob.glob('results*2020*'))[::-1][:2]:
    predictions_per_user = defaultdict(lambda: defaultdict(list))
    predictions_per_sensitive_attr = defaultdict(lambda: defaultdict(list))
    metrics_per_user = defaultdict(list)
    metrics_per_sensitive_attr = defaultdict(list)
    m = r.search(filename)
    print(filename)
    dt = m.group(1)

    with open(filename) as f:
        results = json.load(f)

    if 'model' in results:
        print(results['model'])
    else:
        print('LR')
        
    fold = results['predictions'][0]['fold']
    y_pred = results['predictions'][0]['pred']
    y = results['predictions'][0]['y']

    for user, pred, true in zip(test['user_id'], y_pred, y):
        predictions_per_user[user]['pred'].append(pred)
        predictions_per_user[user]['y'].append(true)

    for attr, pred, true in zip(test[SENSITIVE_ATTR], y_pred, y):
        predictions_per_sensitive_attr[attr]['pred'].append(pred)
        predictions_per_sensitive_attr[attr]['y'].append(true)            

    users_ids = []
    attr_ids = []
    ndcg_ = []
    for user in predictions_per_user:
        this_pred = np.array(predictions_per_user[user]['pred'])
        this_true = np.array(predictions_per_user[user]['y'])
        if len(this_pred) > 1:
            users_ids.append(user)
            metrics_per_user['ndcg'].append(ndcg_score([this_true], [this_pred]))
            metrics_per_user['ndcg@10'].append(ndcg_score([this_true], [this_pred], k=10))
            metrics_per_user['ndcg-'].append(ndcg_score([1 - this_true], [1 - this_pred]))
            ndcg_.append(ndcg_score([1 - this_true], [1 - this_pred]))
            metrics_per_user['ndcg@10-'].append(ndcg_score([1 - this_true], [1 -this_pred], k=10))
        if len(np.unique(this_true)) > 1:
            metrics_per_user['auc'].append(roc_auc_score(this_true, this_pred))

    for attr in predictions_per_sensitive_attr:
        this_pred = np.array(predictions_per_sensitive_attr[attr]['pred'])
        this_true = np.array(predictions_per_sensitive_attr[attr]['y'])
        if len(this_pred) > 1:
            metrics_per_sensitive_attr['ndcg'].append(ndcg_score([this_true], [this_pred]))
            metrics_per_sensitive_attr['ndcg@10'].append(ndcg_score([this_true], [this_pred], k=10))
            metrics_per_sensitive_attr['ndcg-'].append(ndcg_score([1 - this_true], [1 - this_pred]))
            ndcg_.append(ndcg_score([1 - this_true], [1 - this_pred]))
            metrics_per_sensitive_attr['ndcg@10-'].append(ndcg_score([1 - this_true], [1 -this_pred], k=10))
        if len(np.unique(this_true)) > 1:
            attr_ids.append(attr)
            metrics_per_sensitive_attr['auc'].append(roc_auc_score(this_true, this_pred))

    print('Test length', len(y))
    print(y[:10], test[:10])

    print('overall auc', roc_auc_score(y, y_pred))
    print('auc', avgstd(metrics_per_user['auc']))
    print('ndcg', avgstd(metrics_per_user['ndcg']))
    print('ndcg@10', avgstd(metrics_per_user['ndcg@10']))
    print('ndcg-', avgstd(metrics_per_user['ndcg-']))
    print('ndcg@10-', avgstd(metrics_per_user['ndcg@10-']))

    # Display ids of the students that have the lowest/highest ndcg-
    print("Lowest NDCG- = {} on user {}".format(np.around(np.min(ndcg_),5),users_ids[np.argmin(ndcg_)]))
    print(np.array(predictions_per_user[users_ids[np.argmin(ndcg_)]]['y']))
    print(np.array(predictions_per_user[users_ids[np.argmin(ndcg_)]]['pred']))
    print("Highest NDCG- = {} on user {}".format(np.around(np.max(ndcg_),5),users_ids[np.argmax(ndcg_)]))
    print(np.array(predictions_per_user[users_ids[np.argmax(ndcg_)]]['y']))
    print(np.array(predictions_per_user[users_ids[np.argmax(ndcg_)]]['pred']))

    # Display ids of the subgroups (sensitive attribute) that have the lowest/highest AUC
    #print("Lowest AUC = {} on subgroup {}".format(np.around(np.min(metrics_per_sensitive_attr['auc']),5),
    #                                              attr_ids[np.argmin(metrics_per_sensitive_attr['auc'])]))
    worst_indices = np.argsort(metrics_per_sensitive_attr['auc'])[:5]
    best_indices = np.argsort(metrics_per_sensitive_attr['auc'])[-5:][::-1]
    print(worst_indices)
    print(best_indices)
    print("Lowest AUCs = {} on subgroups {}".format(np.around(np.array(metrics_per_sensitive_attr['auc'])[worst_indices],5),
                                                    np.array(attr_ids)[worst_indices]))
    print(np.array(predictions_per_sensitive_attr[attr_ids[np.argmin(metrics_per_sensitive_attr['auc'])]]['y']))
    print(np.array(predictions_per_sensitive_attr[attr_ids[np.argmin(metrics_per_sensitive_attr['auc'])]]['pred']))
    print("Highest AUCs = {} on subgroups {}".format(np.around(np.array(metrics_per_sensitive_attr['auc'])[best_indices],5),
                                                     np.array(attr_ids)[best_indices]))
    print(np.array(predictions_per_sensitive_attr[attr_ids[np.argmax(metrics_per_sensitive_attr['auc'])]]['y']))
    print(np.array(predictions_per_sensitive_attr[attr_ids[np.argmax(metrics_per_sensitive_attr['auc'])]]['pred']))
