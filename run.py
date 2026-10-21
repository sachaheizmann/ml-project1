"""Train our final model and write submission.csv (our best submission).

Pipeline (configuration (c) of cv.py):
  1. special answer codes recoded (rules learned on the training set)
  2. the 9 administrative columns removed
  3. columns > 50% missing and constant columns dropped, median fill,
     standardization, bias column (statistics from the training set)
  4. regularized logistic regression, then a fixed decision threshold

Usage:
    python run.py [data_folder]   # default: data/dataset_to_release/dataset
"""

import argparse
import numpy as np
from helpers import load_csv_data, create_csv_submission
from preprocessing import (
    fit_preprocessing,
    apply_preprocessing,
    find_special_codes,
    apply_special_codes,
)
from implementations import sigmoid, reg_logistic_regression

SEED = 0
N_ADMIN_COLUMNS = 9  # survey process columns (state, date, sequence number, ...)
LAMBDA = 1e-4
GAMMA = 0.5
MAX_ITERS = 300
# Threshold that maximizes F1 on the pooled out-of-fold probabilities of the
# 5-fold cross-validation of this configuration (python threshold.py).
THRESHOLD = 0.195398

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "data_folder",
    nargs="?",
    default="data/dataset_to_release/dataset",
    help="folder with x_train.csv, y_train.csv and x_test.csv",
)
args = parser.parse_args()
np.random.seed(SEED)

print("loading the CSV files ...", flush=True)
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(args.data_folder)
y01 = (y_train + 1) / 2  # labels -1/1 -> 0/1

# special codes and administrative columns (rules from the training set only)
rules = find_special_codes(x_train)
x_train = apply_special_codes(x_train, rules)[:, N_ADMIN_COLUMNS:]
x_test = apply_special_codes(x_test, rules)[:, N_ADMIN_COLUMNS:]

# preprocessing statistics from the training set, applied unchanged to the test set
params = fit_preprocessing(x_train)
tx_train = apply_preprocessing(x_train, *params)
tx_test = apply_preprocessing(x_test, *params)

w0 = np.zeros(tx_train.shape[1])
w, loss = reg_logistic_regression(y01, tx_train, LAMBDA, w0, MAX_ITERS, GAMMA)
print(f"columns used (with bias): {tx_train.shape[1]}, training loss: {loss:.4f}")

prob = sigmoid(tx_test @ w)
y_pred = np.where(prob > THRESHOLD, 1, -1)
print(f"share of test rows predicted positive: {np.mean(y_pred == 1):.2%}")

create_csv_submission(test_ids, y_pred, "submission.csv")
print("wrote submission.csv")
