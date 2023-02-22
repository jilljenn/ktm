"""
Logistic regression on sparse features
"""
import argparse
from datetime import datetime
from collections import defaultdict
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from scipy.sparse import load_npz
import numpy as np
from eval_metrics import avgstd
from dataio import get_paths, load_folds


parser = argparse.ArgumentParser(description='Run LR')
parser.add_argument('X_file', type=str, nargs='?', default='dummy')
parser.add_argument('--test', type=str, nargs='?', default='')
parser.add_argument('--metrics', type=bool, nargs='?', const=True,
    default=False)
parser.add_argument('--folds', type=str, nargs='?', default='weak')
options = parser.parse_args()


df, X_file, folder, y_file, y_pred_file = get_paths(options, 'LR')


if '_en' in folder.name:
    df['group'] = df['country'].map(
        lambda country: 1 if country in {'US', 'CA', 'GB', 'AU'} else 0)
    group_size = df.query('fold == "train"')['group'].value_counts()
    print(group_size)
    df['weight'] = df['group'].map(lambda group: 1 / group_size.loc[group])
    print(df[['group', 'weight']].head())

X_sp = load_npz(X_file).tocsr()
nb_samples, _ = X_sp.shape
y_np = np.load(y_file).astype(np.int32)


results = defaultdict(list)
predictions = []
for i, (i_train, i_test) in enumerate(load_folds(options, df)):
    X_train, X_test, y_train, y_test = (X_sp[i_train], X_sp[i_test],
                                        y_np[i_train], y_np[i_test])
    model = LogisticRegression(solver='liblinear')
    # Has L2 regularization by default
    dt = time.time()

    nb_samples = len(y_train)

    # , df.query('fold != "test"')['weight']
    model.fit(X_train, y_train)

    print('[time] Training', time.time() - dt, 's')

    for dataset, X, y in [('Train', X_train, y_train),
                          ('Test', X_test, y_test)]:
        dt = time.time()
        y_pred = model.predict_proba(X)[:, 1]

        if dataset == 'Test' and options.metrics:

            df_test = df.iloc[i_test]
            assert len(df_test) == len(y_pred)
            df_test['pred'] = y_pred
            df_test.to_csv(y_pred_file, index=False)

        # Store predictions of the fold
        if dataset == 'Test':
            predictions.append({
                'fold': i,
                'pred': y_pred.tolist(),
                'y': y.tolist()
            })

        if len(y_pred) < 10:
            print(dataset, 'predict:', y_pred)
            print(dataset, 'was:', y)
        try:  # This may fail if there are too few classes
            nll = log_loss(y, y_pred)
            auc = roc_auc_score(y, y_pred)
        except ValueError:
            nll = auc = -1

        metrics = {'ACC': np.mean(y == np.round(y_pred)),
                   'NLL': nll,
                   'AUC': auc}
        for metric, value in metrics.items():
            results[f'{dataset} {metric}'].append(value)
            print(dataset, metric, f'on fold {i}', value)
        print('[time]', time.time() - dt, 's')

    np.save(folder / f'coef{i}.npy', model.coef_)
    break

print('# Final results')
for metric in results:
    print(f'{metric}: {avgstd(results[metric])}')

iso_date = datetime.now().isoformat()
saved_results = {
    'predictions': predictions,
    'model': 'LR',
}
with open(folder / f'results-{iso_date}.json', 'w') as f:
    json.dump(saved_results, f)
