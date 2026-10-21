import numpy as np
from helpers import load_csv_data
from preprocessing import fit_preprocessing, apply_preprocessing

x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    "data/dataset_to_release/dataset"
)
params = fit_preprocessing(x_train)
tx_train = apply_preprocessing(x_train, *params)
tx_test = apply_preprocessing(x_test, *params)
y01 = (y_train + 1) / 2  # -1/1  ->  0/1

print("train:", x_train.shape, "-> processed:", tx_train.shape)
print("test:", x_test.shape, "-> processed:", tx_test.shape)
print("any NaN left:", np.isnan(tx_train).any(), np.isnan(tx_test).any())
print("labels now:", np.unique(y01))
