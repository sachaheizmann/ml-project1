"""5-fold cross-validation of the preprocessing configurations.

Every configuration uses regularized logistic regression (lambda=1e-4,
gamma=0.5, 300 iterations) and the SAME 5 folds, so configurations can be
compared fold by fold. Everything learned from data (special-code rules, kept
columns, medians, standardization, indicator and squared-term selection) is
fitted on the training part of each fold only.

With seed 0, fold 5 is exactly the 80/20 split used in ablation.py.
"""

import numpy as np
from preprocessing import (
    fit_preprocessing,
    apply_preprocessing,
    find_special_codes,
    apply_special_codes,
)
from implementations import sigmoid, reg_logistic_regression
from utils import best_f1, f1_score, load_data_cached

K_FOLD = 5
SEED = 0
CONFIGS = [
    "(a) baseline",
    "(b) + special codes recoded",
    "(c) + administrative removed",
    "(d) (c) + missing indicators",
    "(e) (c) + squared terms",
]
REFERENCE = 2  # index of (c), our current best


def build_k_indices(n, k_fold, seed):
    """Random partition of the row numbers 0..n-1 into k_fold folds (lab 4)."""
    np.random.seed(seed)
    return np.array_split(np.random.permutation(n), k_fold)


def honest_f1(y, score):
    """F1 with a threshold that never saw the rows it is scored on.

    The rows of a fold are already in random order, so the first and second
    half are two random halves. The threshold is chosen on one half and the
    F1 computed on the other, both ways, and the two values are averaged.
    """
    half = len(y) // 2
    a, b = slice(0, half), slice(half, None)
    values = []
    for pick, test in [(a, b), (b, a)]:
        _, thr = best_f1(y[pick], score[pick])
        values.append(f1_score(y[test], (score[test] > thr).astype(int)))
    return np.mean(values)


def train_and_predict(x_all, y01, tr, va, **options):
    """Preprocess and train on tr, return the predicted probabilities for va."""
    params = fit_preprocessing(x_all[tr], **options)
    tx_tr = apply_preprocessing(x_all[tr], *params)
    tx_va = apply_preprocessing(x_all[va], *params)
    w0 = np.zeros(tx_tr.shape[1])
    w, _ = reg_logistic_regression(y01[tr], tx_tr, 1e-4, w0, 300, 0.5)
    return sigmoid(tx_va @ w)


def fit_and_score(x_all, y01, tr, va, **options):
    """Preprocess and train on tr, return best F1, its threshold and honest F1 on va."""
    score = train_and_predict(x_all, y01, tr, va, **options)
    f1, thr = best_f1(y01[va], score)
    return f1, thr, honest_f1(y01[va], score)


def main():
    x_train, _, y_train, _, _ = load_data_cached()
    y01 = (y_train + 1) / 2
    folds = build_k_indices(len(y01), K_FOLD, SEED)

    best = np.zeros((len(CONFIGS), K_FOLD))  # best F1 per configuration and fold
    honest = np.zeros((len(CONFIGS), K_FOLD))
    for k in range(K_FOLD):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(K_FOLD) if j != k])
        rules = find_special_codes(x_train[tr])  # learned on this fold's training part
        x_codes = apply_special_codes(x_train.copy(), rules)
        runs = [
            (x_train, {}),
            (x_codes, {}),
            (x_codes[:, 9:], {}),
            (x_codes[:, 9:], {"indicator_min": 0.05}),
            (x_codes[:, 9:], {"square_min_distinct": 10}),
        ]
        for c, (x_all, options) in enumerate(runs):
            f1, thr, h = fit_and_score(x_all, y01, tr, va, **options)
            best[c, k], honest[c, k] = f1, h
            print(
                f"fold {k + 1}  {CONFIGS[c]:30s} best F1 = {f1:.3f} "
                f"(threshold {thr:.3f})  honest F1 = {h:.3f}",
                flush=True,
            )

    print(f"\nmean +- std over the {K_FOLD} folds (std with ddof=1)")
    for c, name in enumerate(CONFIGS):
        print(
            f"{name:30s} best F1 = {best[c].mean():.3f} +- {best[c].std(ddof=1):.3f}"
            f"   honest F1 = {honest[c].mean():.3f} +- {honest[c].std(ddof=1):.3f}"
        )

    print(f"\npaired difference of best F1 against {CONFIGS[REFERENCE]}")
    for c, name in enumerate(CONFIGS):
        if c == REFERENCE:
            continue
        diff = best[c] - best[REFERENCE]
        print(
            f"{name:30s} diff = {diff.mean():+.4f} +- {diff.std(ddof=1):.4f}"
            f"   better in {np.sum(diff > 0)} of {K_FOLD} folds"
        )


if __name__ == "__main__":
    main()
