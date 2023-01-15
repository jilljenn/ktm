from sklearn.metrics import (
    roc_auc_score, log_loss, ndcg_score, roc_curve, auc, accuracy_score,
    mean_squared_error)
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import logging
from eval_metrics import avgstd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Eval predictions')
parser.add_argument('pred_file', type=str, nargs='?')
options = parser.parse_args()

df = pd.read_csv(options.pred_file)
print('Overall AUC', roc_auc_score(df['correct'], df['pred']))
print('Overall ACC', accuracy_score(df['correct'], df['pred'].round()))
print('Overall NDCG', ndcg_score([df['correct']], [df['pred']]))

df['group'] = df['country'].map(lambda country: 1 if country in {'US', 'CA', 'GB', 'AU'} else 0)
print(df[['user_id', 'country']].drop_duplicates()['country'].value_counts())
print(df[['user_id', 'country']].drop_duplicates().shape)

def mean_round(l):
    return np.round(l).mean()

scores = df.groupby(['user_id', 'group'])[['correct', 'pred']].agg(mean_round)
for cutoff in [75, 80, 85]:
    scores[f'passes_{cutoff}'] = scores['correct'].map(
        lambda x: x >= cutoff / 100)
    counts = scores.groupby(['group', f'passes_{cutoff}']).agg('count')
    print('counts', counts)
    print('%', counts / len(scores))
    for group in [0, 1]:
        scores_group = scores.query("group == @group")
        fpr, tpr, threshold = roc_curve(
            scores_group[f'passes_{cutoff}'], scores_group['pred'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Group {group} AUC={roc_auc:.3f}')
    # plt.boxplot(scores)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'AUC for both groups if those who answer correctly {cutoff}% will pass')
    plt.legend()
    plt.show()

users = df['user_id'].unique()
metrics = defaultdict(list)
for user_id in users:
    df_user = df.query("user_id == @user_id")
    truth = [int(value) for value in df_user['correct'].tolist()]
    true_rate = np.mean(truth)
    pred = df_user['pred'].tolist()
    pred_rate = np.round(pred).mean()
    country = df_user["country"].tolist()[0]
    group = df_user["group"].tolist()[0]

    acc = accuracy_score(truth, np.round(pred))
    ndcg = ndcg_score([truth], [pred])
    mse = (true_rate - pred_rate) ** 2
    metrics[f'mse_{country}'].append(mse)
    metrics[f'mse_{group}'].append(mse)
    metrics[f'acc_{country}'].append(acc)
    metrics[f'acc_{group}'].append(acc)
    metrics[f'ndcg_{country}'].append(ndcg)
    metrics[f'ndcg_{group}'].append(ndcg)

    if len(set(truth)) == 1:
        '''logging.warning(
                                    'User %s (%s) has all %d answers equal to %d (acc: %.3f ndcg: %.3f)',
                                    user_id, country, len(truth), truth[0], acc, ndcg)'''
        auc_value = acc
    else:
        auc_value = roc_auc_score(truth, pred)

    metrics[f'auc_{country}'].append(auc_value)
    metrics[f'auc_{group}'].append(auc_value)

# print(sorted(metrics))
for metric in ['acc', 'auc', 'ndcg', 'mse']:
    for group in [0, 1]:
        values = metrics[f'{metric}_{group}']
        print(metric, group, avgstd(values))
        plt.hist(metrics[f'{metric}_{group}'], label=group, bins=50, alpha=0.5, density=True)
        if metric in {'auc', 'acc'}:
            nb_bad = len([value for value in values if value <= 0.6])
        elif metric == 'ndcg':
            nb_bad = len([value for value in values if value <= 0.96])
        elif metric == 'mse':
            nb_bad = len([value for value in values if value >= 0.15])
        nb_total = len(values)
        print(metric, group, f'{nb_bad} bad over {nb_total} ({nb_bad / nb_total * 100:.2f} %)')
    plt.legend()
    plt.title(metric)
    plt.show()

for group in [0, 1]:
    df_group = df.query("group == @group")
    truth = [int(value) for value in df_group['correct'].tolist()]
    pred = df_group['pred'].tolist()
    fpr, tpr, threshold = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)
    ndcg = ndcg_score([truth], [pred])
    acc = accuracy_score(truth, np.round(pred))
    plt.plot(fpr, tpr, label=f'{group} AUC={roc_auc} NDCG={ndcg} ACC={acc}')
plt.legend()
plt.show()
