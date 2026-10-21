"""Required ML methods for Project 1.

Conventions (from the project description):
  - y has shape (N,), tx has shape (N, D), w has shape (D,)  (1D arrays)
  - every function returns (w, loss): the LAST weight vector and its loss
  - the loss of the regularized methods does NOT include the penalty term
"""

import numpy as np


# -------------------------------- helpers --------------------------------
def compute_mse(y, tx, w):
    """Mean squared error with the 0.5 factor: 1/(2N) * sum (y - tx w)^2."""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Gradient of the MSE: -1/N * tx^T (y - tx w)."""
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(t):
    """Logistic function, maps any score to a probability in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """Mean negative log-likelihood for labels y in {0, 1}.

    log(1 + e^s) - y*s is computed with logaddexp to avoid overflow.
    """
    scores = tx @ w
    return np.mean(np.logaddexp(0, scores) - y * scores)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the mean logistic loss: 1/N * tx^T (sigmoid(tx w) - y)."""
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ------------------------------------------------- 1. linear regression, GD
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


# ------------------------------------------------ 2. linear regression, SGD
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1)."""
    w = initial_w
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(n)  # pick ONE random datapoint
        # gradient of the loss of that single sample
        grad = -tx[i] * (y[i] - tx[i] @ w)
        w = w - gamma * grad
    return w, compute_mse(y, tx, w)


# ------------------------------------------------------ 3. least squares
def least_squares(y, tx):
    """Least squares regression using the normal equations."""
    # solve (X^T X) w = X^T y ; solve() is more stable than inverting
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


# ------------------------------------------------------ 4. ridge regression
def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    The penalty is lambda_ * ||w||^2 on top of the 1/(2N)-scaled MSE, which
    gives (X^T X + 2 N lambda_ I) w = X^T y.
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, compute_mse(y, tx, w)  # loss without the penalty term


# --------------------------------------------------- 5. logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1})."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


# ------------------------------------------- 6. regularized logistic regr.
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Penalty term: lambda_ * ||w||^2, whose gradient is 2 * lambda_ * w.
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)  # loss without the penalty term
