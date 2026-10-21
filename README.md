# ML Project 1: predicting heart disease (MICHD) from the BRFSS survey

We predict whether a person has coronary heart disease from 321 survey features
(labels -1/1, 8.8% positive). The final model is regularized logistic regression
after recoding the special answer codes and removing the administrative columns;
the decision threshold is chosen to maximize F1 with 5-fold cross-validation.
Only Python, NumPy and the standard library are used.

## Data

Put the three CSV files of the competition in `data/dataset_to_release/dataset/`:

```text
data/dataset_to_release/dataset/x_train.csv
data/dataset_to_release/dataset/y_train.csv
data/dataset_to_release/dataset/x_test.csv
```

`run.py` also accepts another folder as argument (see below). The `data/` folder is
not in the repository.

## Reproducing our submission

```bash
python run.py
```

or, with the data somewhere else, `python run.py path/to/dataset_folder`.
It writes `submission.csv` (Id, Prediction in -1/1) in the current folder and
takes about 75 seconds (about 50 seconds of it is reading the CSV files).
No random choices are involved, and the seed is fixed anyway.

## Files

| File | What it does |
| --- | --- |
| `implementations.py` | The 6 required methods (GD, SGD, least squares, ridge, logistic, regularized logistic regression), each returning `(w, loss)`. |
| `preprocessing.py` | Special-code recoding and the preprocessing pipeline (drop columns > 50% missing and constant columns, median fill, standardization, bias column). Optional, off by default: missing-value indicators and squared terms. All statistics are fitted on training rows only. |
| `run.py` | Trains the final model on the full training set and writes `submission.csv`. |
| `cv.py` | 5-fold cross-validation of the preprocessing configurations: mean and standard deviation of F1, paired differences, honest F1. |
| `threshold.py` | Computes the decision threshold used in `run.py` from the pooled out-of-fold predictions of `cv.py`. |
| `ablation.py` | Ablation table on a single 80/20 split (seed 0). `--slow` also runs 1000 and 3000 iterations. |
| `validate.py` | Compares the 6 required methods on the same 80/20 split. |
| `explore.py` | Exploratory statistics (labels, missing values, distinct values, features with the most distinct values). |
| `look_codes.py` | Prints the most frequent values of the first 40 features (used to find the special codes). |
| `check_preprocessing.py` | Sanity check of the preprocessed shapes and missing values. |
| `utils.py` | F1 score, threshold search, and `load_data_cached` (see below). |
| `helpers.py` | Course-provided CSV loading and submission writing. |
| `latex-example-paper/report.tex` | The report. |

## Cache for the experiment scripts

All scripts except `run.py` load the data with `utils.load_data_cached`: the first
call reads the CSV files (about 50 seconds) and saves them as `.npy` files in
`cache/` (1.1 GB, not in the repository); later calls load them in under a second.
Delete `cache/` if the CSV files change. `run.py` always reads the CSV files.

## Runtimes

Measured on our machine (Linux, 30 GB RAM), with the cache already built:

| Command | Time |
| --- | --- |
| `python run.py` | ~75 s (reads the CSVs) |
| `python ablation.py` | ~100 s |
| `python ablation.py --slow` | ~280 s |
| `python cv.py` | ~500 s |
| `python threshold.py` | ~105 s |
| `python explore.py` | ~6 s |
| `python validate.py` | not timed |

## Grading tests

The tests need `pytest`, `gitpython` and `black` (see `grading_tests/environment.yml`):

```bash
cd grading_tests
pytest --github_link <URL of this GitHub repository> .
```

During development `--github_link ..` tests the local folder; the only test that
fails then is `test_github_link_format`, which requires a real GitHub URL.
