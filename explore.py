import numpy as np
from utils import load_data_cached

x_train, x_test, y_train, train_ids, test_ids = load_data_cached()
# print("train:", x_train.shape, " test:", x_test.shape)

values, counts = np.unique(y_train, return_counts=True)
print("labels:", dict(zip(values, counts)))
print("share of +1:", np.mean(y_train == 1))

missing = np.isnan(x_train).mean(axis=0)  # share of NaN per feature
print("share of all values missing:", np.isnan(x_train).mean())
print("features with no missing values:", np.sum(missing == 0))
print("features with more than 50% missing:", np.sum(missing > 0.5))

n_unique = np.array([len(np.unique(col[~np.isnan(col)])) for col in x_train.T])
print("features with at most 2 distinct values:", np.sum(n_unique <= 2))
print(
    "features with 3 to 10 distinct values:", np.sum((n_unique > 2) & (n_unique <= 10))
)
print("features with more than 10 distinct values:", np.sum(n_unique > 10))

# the first line of x_train.csv holds the feature names (column 0 is the Id)
with open("data/dataset_to_release/dataset/x_train.csv") as f:
    names = f.readline().strip().split(",")[1:]
print("10 features with the most distinct values (columns 0-8 are administrative):")
for j in np.argsort(-n_unique)[:10]:
    print(
        f"  {j:3d} {names[j]:10s} distinct={n_unique[j]:7d}  missing={missing[j]:.1%}"
    )
