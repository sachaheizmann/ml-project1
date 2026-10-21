import os
import numpy as np
from helpers import load_csv_data

CACHE_NAMES = ["x_train", "x_test", "y_train", "train_ids", "test_ids"]


def load_data_cached(data_path="data/dataset_to_release/dataset", cache_dir="cache"):
    """Same output as load_csv_data, but reads fast .npy copies when they exist.

    The first call parses the CSVs and saves the five arrays in cache_dir.
    Delete cache_dir if the CSV files ever change.
    """
    paths = [os.path.join(cache_dir, name + ".npy") for name in CACHE_NAMES]
    if all(os.path.exists(p) for p in paths):  # cache complete: fast path
        return tuple(np.load(p) for p in paths)
    arrays = load_csv_data(data_path)  # slow: parse the text files
    os.makedirs(cache_dir, exist_ok=True)
    for p, a in zip(paths, arrays):
        np.save(p, a)
    return arrays


def f1_score(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))  # true positives
    fp = np.sum((y_pred == 1) & (y_true == 0))  # false alarms
    fn = np.sum((y_pred == 0) & (y_true == 1))  # missed cases
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_f1(y_true, score):
    """Try 100 thresholds between the 1st and 99th percentile of the scores."""
    best = (0.0, None)
    for thr in np.quantile(score, np.linspace(0.01, 0.99, 100)):
        f1 = f1_score(y_true, (score > thr).astype(int))
        if f1 > best[0]:
            best = (f1, thr)
    return best
