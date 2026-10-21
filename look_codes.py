import numpy as np
from helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    "data/dataset_to_release/dataset"
)

# the first line of x_train.csv holds the feature names (column 0 is the Id)
with open("data/dataset_to_release/dataset/x_train.csv") as f:
    names = f.readline().strip().split(",")[1:]

for j in range(0, 40):  # first 40 features
    col = x_train[:, j]
    vals, counts = np.unique(col[~np.isnan(col)], return_counts=True)
    top = np.argsort(-counts)[:6]  # the 6 most common values
    pairs = ", ".join(f"{vals[k]:g}:{counts[k]}" for k in top)
    print(f"{j:3d} {names[j]:14s} distinct={len(vals):5d}  {pairs}")
