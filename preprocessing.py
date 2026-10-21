import numpy as np

SPECIAL_PAIRS = [(7, 9), (77, 99), (777, 999), (7777, 9999)]  # (don't know, refused)
NONE_CODE = {77: 88, 777: 888, 7777: 8888}  # "none" -> 0 (not for 1 digit)


def find_special_codes(x_train):
    """Decide, per column, which values are special codes. Uses training data only."""
    rules = []
    for j in range(x_train.shape[1]):
        col = x_train[:, j]
        vals = np.unique(col[~np.isnan(col)])
        to_nan, to_zero = [], []
        for dont_know, refused in SPECIAL_PAIRS:
            if len(vals) > 0 and refused in vals and vals.max() == refused:
                to_nan.append(refused)
                if dont_know in vals:
                    to_nan.append(dont_know)
                none = NONE_CODE.get(dont_know)
                if none is not None and none in vals:
                    to_zero.append(none)
        if to_nan or to_zero:
            rules.append((j, to_nan, to_zero))
    return rules


def apply_special_codes(x, rules):
    """Recode IN PLACE: special codes become NaN, 'none' codes become 0."""
    for j, to_nan, to_zero in rules:
        for c in to_nan:
            x[x[:, j] == c, j] = np.nan
        for c in to_zero:
            x[x[:, j] == c, j] = 0
    return x


def build_features(x, keep, medians, indicator):
    """Kept columns with NaN replaced by the median, followed by 0/1 indicators."""
    x = x[:, keep]
    flags = np.isnan(x[:, indicator]).astype(float)  # BEFORE the fill erases NaN
    x = np.where(np.isnan(x), medians, x)
    return np.c_[x, flags]


def squared_terms(z, square):
    """Squares of the selected standardized features.

    Only the first len(square) columns are real features; indicator columns
    come after them and are never squared.
    """
    return z[:, : len(square)][:, square] ** 2


def fit_preprocessing(
    x_train, max_missing=0.5, indicator_min=None, square_min_distinct=None
):
    """Learn everything we need from the TRAINING data only.

    indicator_min: if set, every kept column with at least this share of
    missing values gets an extra 0/1 "was missing" column.
    square_min_distinct: if set, every kept column with more than this many
    distinct values gets an extra column with its standardized value squared.
    """
    missing = np.isnan(x_train).mean(axis=0)
    keep = missing <= max_missing  # columns with enough data
    x = x_train[:, keep]
    medians = np.nanmedian(x, axis=0)
    useful = np.where(np.isnan(x), medians, x).std(axis=0) > 0  # False if constant
    keep[np.flatnonzero(keep)[~useful]] = False  # also drop them from `keep`
    medians = medians[useful]

    if indicator_min is None:
        indicator = np.zeros(keep.sum(), dtype=bool)  # no indicator columns
    else:
        indicator = missing[keep] >= indicator_min
    x = build_features(x_train, keep, medians, indicator)
    mean, std = x.mean(axis=0), x.std(axis=0)

    if square_min_distinct is None:
        square = np.zeros(keep.sum(), dtype=bool)  # no squared columns
    else:
        n_distinct = [len(np.unique(c[~np.isnan(c)])) for c in x_train[:, keep].T]
        square = np.array(n_distinct) > square_min_distinct
    sq = squared_terms((x - mean) / std, square)
    return keep, medians, indicator, mean, std, square, sq.mean(axis=0), sq.std(axis=0)


def apply_preprocessing(
    x, keep, medians, indicator, mean, std, square, sq_mean, sq_std
):
    """Apply the training-data statistics to x."""
    x = build_features(x, keep, medians, indicator)
    x = (x - mean) / std  # standardize
    sq = (squared_terms(x, square) - sq_mean) / sq_std  # same scale as the others
    return np.c_[np.ones(len(x)), x, sq]  # add the bias column of ones
