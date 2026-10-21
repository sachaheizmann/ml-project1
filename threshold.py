"""Decision threshold used in run.py, from pooled out-of-fold predictions.

Runs configuration (c) of cv.py (special codes recoded, administrative columns
removed) on the same 5 folds. Every training row is predicted once, by the
model of the fold in which it was NOT used for training. The threshold that
maximizes F1 on all 328,135 pooled predictions is copied into run.py.
"""

import numpy as np
from cv import K_FOLD, SEED, build_k_indices, train_and_predict
from preprocessing import find_special_codes, apply_special_codes
from utils import best_f1, load_data_cached

x_train, _, y_train, _, _ = load_data_cached()
y01 = (y_train + 1) / 2
folds = build_k_indices(len(y01), K_FOLD, SEED)

oof = np.zeros(len(y01))  # out-of-fold probability of every training row
for k in range(K_FOLD):
    va = folds[k]
    tr = np.concatenate([folds[j] for j in range(K_FOLD) if j != k])
    rules = find_special_codes(x_train[tr])  # learned on this fold's training part
    x_c = apply_special_codes(x_train.copy(), rules)[:, 9:]
    oof[va] = train_and_predict(x_c, y01, tr, va)
    f1, thr = best_f1(y01[va], oof[va])
    print(f"fold {k + 1}: best F1 = {f1:.3f} (threshold {thr:.3f})", flush=True)

f1, thr = best_f1(y01, oof)
print(f"pooled out-of-fold: best F1 = {f1:.4f} at threshold {thr:.6f}")
print(f"share of training rows predicted positive: {np.mean(oof > thr):.2%}")
