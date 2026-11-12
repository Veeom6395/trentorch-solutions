
import numpy as np

from _load import load_solution

correlation = load_solution("math-expectation-variance-covariance").correlation


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    """
    x is (num_samples, num_features). Returns a (num_features,
    num_features) correlation matrix.
    """
    num_features = x.shape[1]

    corr_matrix = np.eye(num_features)

    # Calculate each unique pair only once
    for i in range(num_features):
        for j in range(i + 1, num_features):
            corr = correlation(x[:, i], x[:, j])

            # Exploit symmetry
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def most_correlated_pair(corr_matrix: np.ndarray) -> tuple[int, int]:
    """
    Returns the indices of the strongest off-diagonal
    correlation by absolute value.
    """
    n = corr_matrix.shape[0]

    # Ignore diagonal entries
    abs_corr = np.abs(corr_matrix.copy())
    np.fill_diagonal(abs_corr, -np.inf)

    # Find the strongest relationship
    i, j = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)

    return (i, j)
