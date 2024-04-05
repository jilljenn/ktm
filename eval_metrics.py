import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, ndcg_score, log_loss, roc_curve
from collections import Counter, defaultdict
from scipy.stats import sem, t
import pandas as pd
import numpy as np
import json
import re
import os
import sys


SENSITIVE_ATTR = "school_id"
THIS_GROUP = 25


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


def all_metrics(results, test):
    ndcg_ = defaultdict(list)
    predictions_per_user = defaultdict(lambda: defaultdict(list))
    predictions_per_sensitive_attr = defaultdict(lambda: defaultdict(list))
    metrics_per_user = defaultdict(list)
    metrics_per_sensitive_attr = defaultdict(list)
    #roc_curves_per_sensitive_attr = defaultdict(lambda: defaultdict(list))

    model = 'LR' if results['model'] == 'LR' else 'FM' + str(len(str(results['model'])))
    print(model)

    fold = results['predictions'][0]['fold']
    y_pred = np.array(np.array(results['predictions'][0]['pred']))
    y = np.array(results['predictions'][0]['y'])

    try:
        assert len(y) == len(test)
    except AssertionError:
        print('This is not the right fold', len(y), len(test))
        sys.exit(0)

    for user, pred, true in zip(test['user'], y_pred, y):
        predictions_per_user[user]['pred'].append(pred)
        predictions_per_user[user]['y'].append(true)

    attribute = np.array(test[SENSITIVE_ATTR])
    protected = np.argwhere(attribute % 2 == 0).reshape(-1)
    unprotected = np.argwhere(attribute % 2 == 1).reshape(-1)
    # protected = np.argwhere(attribute == THIS_GROUP).reshape(-1)
    # unprotected = np.argwhere(attribute != THIS_GROUP).reshape(-1)
    print(type(y))
    print(len(y), len(y[protected]), len(y[unprotected]))

    for attr, pred, true in zip(test[SENSITIVE_ATTR], y_pred, y):
        predictions_per_sensitive_attr[attr]['pred'].append(pred)
        predictions_per_sensitive_attr[attr]['y'].append(true)

    users_ids = []
    attr_ids = []
    for user in predictions_per_user:
        this_pred = np.array(predictions_per_user[user]['pred'])
        this_true = np.array(predictions_per_user[user]['y'])
        if len(this_pred) > 1:
            users_ids.append(user)
            # print(this_true)
            metrics_per_user['nll'].append(log_loss(this_true, this_pred, labels=[0, 1]))
            metrics_per_user['ndcg'].append(ndcg_score([this_true], [this_pred]))
            metrics_per_user['ndcg@10'].append(ndcg_score([this_true], [this_pred], k=10))
            metrics_per_user['ndcg-'].append(ndcg_score([1 - this_true], [1 - this_pred]))
            ndcg_[model].append(ndcg_score([1 - this_true], [1 - this_pred]))
            metrics_per_user['ndcg@10-'].append(ndcg_score([1 - this_true], [1 -this_pred], k=10))
        if len(np.unique(this_true)) > 1:
            metrics_per_user['auc'].append(roc_auc_score(this_true, this_pred))

    nb_samples = []
    for attr in predictions_per_sensitive_attr:
        this_pred = np.array(predictions_per_sensitive_attr[attr]['pred'])
        this_true = np.array(predictions_per_sensitive_attr[attr]['y'])
        if len(this_pred) > 1:
            metrics_per_sensitive_attr['ndcg'].append(ndcg_score([this_true], [this_pred]))
            metrics_per_sensitive_attr['ndcg@10'].append(ndcg_score([this_true], [this_pred], k=10))
            metrics_per_sensitive_attr['ndcg-'].append(ndcg_score([1 - this_true], [1 - this_pred]))
            metrics_per_sensitive_attr['ndcg@10-'].append(ndcg_score([1 - this_true], [1 -this_pred], k=10))
        if len(np.unique(this_true)) > 1:
            attr_ids.append(attr)
            nb_samples.append(len(this_true))
            metrics_per_sensitive_attr['auc'].append(roc_auc_score(this_true, this_pred))
            #roc_curves_per_sensitive_attr[attr]["fpr"], roc_curves_per_sensitive_attr[attr]["tpr"], _ = roc_curve(this_true, this_pred)

    print('Test length', len(y))
    print(y[:10], test[:10])

    print('overall auc', np.round(roc_auc_score(y, y_pred), 3))
    print('overall nll', np.round(log_loss(y, y_pred), 3))
    print('sliced auc (per user)', avgstd(metrics_per_user['auc']))
    print('sliced auc (per group)', avgstd(metrics_per_sensitive_attr['auc']))
    print('sliced nll', avgstd(metrics_per_user['nll']))

    """
    print('ndcg', avgstd(metrics_per_user['ndcg']))
    print('ndcg@10', avgstd(metrics_per_user['ndcg@10']))
    print('ndcg-', avgstd(metrics_per_user['ndcg-']))
    print('ndcg@10-', avgstd(metrics_per_user['ndcg@10-']))
    """

    # Display ids of the students that have the lowest/highest ndcg-
    # print("Lowest NDCG- = {} on user {}".format(np.around(np.min(ndcg_[model]),5),users_ids[np.argmin(ndcg_[model])]))
    # print(np.array(predictions_per_user[users_ids[np.argmin(ndcg_)]]['y']))
    # print(np.array(predictions_per_user[users_ids[np.argmin(ndcg_)]]['pred']))
    # print("Highest NDCG- = {} on user {}".format(np.around(np.max(ndcg_[model]),5),users_ids[np.argmax(ndcg_[model])]))
    # print(np.array(predictions_per_user[users_ids[np.argmax(ndcg_)]]['y']))
    # print(np.array(predictions_per_user[users_ids[np.argmax(ndcg_)]]['pred']))

    """
    diff = abs(np.array(ndcg_[model]) - np.array(ndcg_['FM76']))
    this_pos = np.argmax(diff)
    this_user = users_ids[this_pos]
    print("Biggest difference NDCG- = {} {}{} LR{} on user {}".format(np.max(diff), model, ndcg_[model][this_pos], ndcg_['FM76'][this_pos], this_user))
    print(sorted(list(zip(predictions_per_user[this_user]['pred'], predictions_per_user[this_user]['y']))))
    """

    candidates = Counter()
    val = 0
    for subgroup, auc, nb in zip(attr_ids, metrics_per_sensitive_attr['auc'], nb_samples):
        candidates[subgroup] = (-auc, -nb)
    print(len(candidates), 'groups and ', test[SENSITIVE_ATTR].nunique(), 'schools in test')

    x = []
    nb = []
    for k, (xi, yi) in candidates.most_common():
        if val < 5:
            print(k, (-xi, -yi))
        x.append(-xi)
        nb.append(-yi)
        val += 1
    plt.stem(x, nb, use_line_collection=True)
    plt.xlabel('AUC value')
    plt.ylabel('Number of samples in group')
    plt.title('For each group, number of samples per AUC value')
    plt.show()

    # Display ids of the subgroups (sensitive attribute) that have the lowest/highest AUC
    print("Lowest AUC = {} on subgroup {}".format(np.around(np.min(metrics_per_sensitive_attr['auc']),5),
                                                  attr_ids[np.argmin(metrics_per_sensitive_attr['auc'])]))
    # print(np.array(predictions_per_sensitive_attr[attr_ids[np.argmin(metrics_per_sensitive_attr['auc'])]]['y'])[:10])
    # print(np.array(predictions_per_sensitive_attr[attr_ids[np.argmin(metrics_per_sensitive_attr['auc'])]]['pred'])[:10])
    print("Highest AUC = {} on subgroup {}".format(np.around(np.max(metrics_per_sensitive_attr['auc']),5),
                                                   attr_ids[np.argmax(metrics_per_sensitive_attr['auc'])]))
    # print(np.array(predictions_per_sensitive_attr[attr_ids[np.argmax(metrics_per_sensitive_attr['auc'])]]['y'])[:10])
    # print(np.array(predictions_per_sensitive_attr[attr_ids[np.argmax(metrics_per_sensitive_attr['auc'])]]['pred'])[:10])

    print('AUC of that group', roc_auc_score(y[protected], y_pred[protected]))
    fpr_protec, tpr_protec, _ = roc_curve(y[protected], y_pred[protected])
    # print('NDCG of that group', ndcg_score([y[protected]], [y_pred[protected]]))
    print('AUC of other group', roc_auc_score(y[unprotected], y_pred[unprotected]))
    # print('NDCG of other group', ndcg_score([y[unprotected]], [y_pred[unprotected]]))
    fpr_unprotec, tpr_unprotec, _ = roc_curve(y[unprotected], y_pred[unprotected])
    # Plot ROC curves of proteced vs. unprotected groups
    plt.plot(fpr_protec, tpr_protec, label="Protected group")
    plt.plot(fpr_unprotec, tpr_unprotec, label="Unprotected group")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves comparison between protected and unprotected groups")
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    os.chdir('data/assistments2009full')
    # os.chdir('data/fr_en')

    df = pd.read_csv('data.csv')

    # r = re.compile(r'results-(.*).json')

    # ndcg_ = defaultdict(list)
    for filename in sorted(glob.glob('results*2024*'))[::-1][:1]:

        print(filename)
        
        with open(filename) as f:
            results = json.load(f)

        i_test = results['predictions'][0]['i_test']
        all_metrics(results, df.iloc[i_test])
