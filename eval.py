from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import logging
from eval_metrics import avgstd


df = pd.read_csv('data/es_en/y_us_pred.csv')
df['group'] = df['country'].map(lambda country: 1 if country in {'US', 'CA', 'GB', 'AU'} else 0)
print(df[['user_id', 'country']].drop_duplicates()['country'].value_counts())
print(df[['user_id', 'country']].drop_duplicates().shape)

users = df['user_id'].unique()
metrics = defaultdict(list)
for user_id in users:
    df_user = df.query("user_id == @user_id")
    truth = [int(value) for value in df_user['correct'].tolist()]
    pred = df_user['pred'].tolist()
    if len(set(truth)) == 1:
        logging.warning('User %s has everything %d equal', user_id, len(truth))
        continue
    auc = roc_auc_score(truth, pred)
    ndcg = ndcg_score([truth], [pred])
    country = df_user["country"].tolist()[0]
    group = df_user["group"].tolist()[0]
    metrics[f'auc_{country}'].append(auc)
    metrics[f'ndcg_{country}'].append(ndcg)
    metrics[f'auc_{group}'].append(auc)
    metrics[f'ndcg_{group}'].append(ndcg)

print(sorted(metrics))
for metric in ['auc', 'ndcg']:
    for group in [0, 1]:
        values = metrics[f'{metric}_{group}']
        print(metric, group, avgstd(values))
        plt.hist(metrics[f'{metric}_{group}'], label=group, bins=50, alpha=0.5, density=False)
        if metric == 'auc':
            nb_bad = len([value for value in values if value <= 0.6])
        else:
            nb_bad = len([value for value in values if value <= 0.96])
        nb_total = len(values)
        print(metric, group, f'{nb_bad} bad over {nb_total} ({nb_bad / nb_total * 100:.2f} %)')
    plt.legend()
    plt.show()
