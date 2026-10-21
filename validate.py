import numpy as np
from preprocessing import fit_preprocessing, apply_preprocessing
from implementations import (
    sigmoid,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from utils import best_f1, f1_score, load_data_cached

# ---------------------------------------------------------------- 1. load + split
np.random.seed(0)  # same split every run
x_train, x_test, y_train, train_ids, test_ids = load_data_cached()
y01 = (y_train + 1) / 2  # labels -1/1 -> 0/1

idx = np.random.permutation(len(y01))  # shuffled row numbers
cut = int(0.8 * len(y01))
tr, va = idx[:cut], idx[cut:]  # 80% train, 20% validation

# ---------------------------------------------------------------- 2. preprocessing
params = fit_preprocessing(x_train[tr])  # learn from the 80% only
tx_tr = apply_preprocessing(x_train[tr], *params)
tx_va = apply_preprocessing(x_train[va], *params)  # validation uses the SAME numbers
y_tr, y_va = y01[tr], y01[va]

# ---------------------------------------------------------------- 4. train all six
D = tx_tr.shape[1]
w0 = np.zeros(D)
results = {}

w, _ = mean_squared_error_gd(y_tr, tx_tr, w0, 300, 0.05)
results["mean_squared_error_gd"] = tx_va @ w

best_sgd = None
for g in [1e-3, 1e-4, 1e-5]:
    w, _ = mean_squared_error_sgd(y_tr, tx_tr, w0, 100000, g)
    f1, _ = best_f1(y_va, tx_va @ w)
    print(f"  sgd gamma={g}: F1 = {f1:.3f}, weight norm = {np.linalg.norm(w):.1f}")
    if best_sgd is None or f1 > best_sgd[0]:
        best_sgd = (f1, tx_va @ w, g)
print(f"  sgd step size chosen (best F1): {best_sgd[2]}")
results["mean_squared_error_sgd"] = best_sgd[1]

w, _ = least_squares(y_tr, tx_tr)
results["least_squares"] = tx_va @ w

w, _ = ridge_regression(y_tr, tx_tr, 1e-4)
results["ridge_regression"] = tx_va @ w

w, _ = logistic_regression(y_tr, tx_tr, w0, 300, 0.5)
results["logistic_regression"] = sigmoid(tx_va @ w)

w, _ = reg_logistic_regression(y_tr, tx_tr, 1e-4, w0, 300, 0.5)
results["reg_logistic_regression"] = sigmoid(tx_va @ w)
f1_half = f1_score(y_va, (sigmoid(tx_va @ w) > 0.5).astype(int))
print(f"  reg_logistic_regression at fixed threshold 0.5: F1 = {f1_half:.3f}")

# ---------------------------------------------------------------- 5. results table
print(f"columns: {D}")
for name, score in results.items():
    f1, thr = best_f1(y_va, score)
    print(f"{name:26s} best F1 = {f1:.3f}  (threshold {thr:.3f})")
