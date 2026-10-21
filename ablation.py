"""Ablation of the preprocessing: reproduces every row of the ablation table.

Each row adds one change to the previous one, and all rows use regularized
logistic regression on the same 80/20 split (seed 0).
The two long-training rows (1000 and 3000 iterations) only run with --slow:
    python ablation.py          # quick rows
    python ablation.py --slow   # every row of the table
"""

import argparse
import numpy as np
from preprocessing import (
    fit_preprocessing,
    apply_preprocessing,
    find_special_codes,
    apply_special_codes,
)
from implementations import sigmoid, reg_logistic_regression
from utils import best_f1, load_data_cached

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--slow", action="store_true", help="also run 1000/3000 iterations")
args = parser.parse_args()

np.random.seed(0)
x_train, x_test, y_train, train_ids, test_ids = load_data_cached()
y01 = (y_train + 1) / 2
idx = np.random.permutation(len(y01))
cut = int(0.8 * len(y01))
tr, va = idx[:cut], idx[cut:]  # the SAME split as validate.py


def evaluate(
    x_all,
    label,
    lambda_=1e-4,
    iters=300,
    gamma=0.5,
    indicator_min=None,
    square_min_distinct=None,
):
    """Preprocess, train on the 80% part and print the best validation F1."""
    params = fit_preprocessing(
        x_all[tr], indicator_min=indicator_min, square_min_distinct=square_min_distinct
    )
    tx_tr = apply_preprocessing(x_all[tr], *params)
    tx_va = apply_preprocessing(x_all[va], *params)
    w0 = np.zeros(tx_tr.shape[1])
    w, _ = reg_logistic_regression(y01[tr], tx_tr, lambda_, w0, iters, gamma)
    f1, thr = best_f1(y01[va], sigmoid(tx_va @ w))
    print(
        f"{label:36s} columns={tx_tr.shape[1]:4d}  "
        f"best F1 = {f1:.3f}  (threshold {thr:.3f})"
    )


# row 1: baseline preprocessing on the raw data
evaluate(x_train, "baseline")

# row 2: special codes recoded (on a copy, so x_train stays raw)
rules = find_special_codes(x_train[tr])  # rules from the training part only
print("columns with special-code rules:", len(rules))
x_codes = apply_special_codes(x_train.copy(), rules)
evaluate(x_codes, "+ special codes recoded")

# row 3: additionally remove the 9 administrative columns
evaluate(x_codes[:, 9:], "+ administrative features removed")

# rows 4-5: the row-3 data trained for longer (row 3 used 300 iterations)
if args.slow:
    for iters in [1000, 3000]:
        evaluate(x_codes[:, 9:], f"  same, {iters} iterations", iters=iters)

# row 6: row 3 plus a 0/1 "was missing" column for features with >= 5% missing
evaluate(x_codes[:, 9:], "+ missing-value indicators (>= 5%)", indicator_min=0.05)

# row 7: row 3 plus squared terms of features with > 10 distinct values
evaluate(x_codes[:, 9:], "+ squared terms (> 10 distinct)", square_min_distinct=10)
