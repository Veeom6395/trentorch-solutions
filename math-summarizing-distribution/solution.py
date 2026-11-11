import numpy as np


def skewness(x: np.ndarray) -> float:
    """
    Pearson's moment coefficient of skewness: the average CUBED
    z-score of every value.
    """
    mean = np.mean(x)
    std = np.std(x)

    if std == 0:
        return 0.0

    z_scores = (x - mean) / std
    return float(np.mean(z_scores ** 3))


def summarize_distribution(x: np.ndarray) -> dict:
    """
    Returns {"mean": ..., "median": ..., "std": ..., "skew": ...},
    the standard quick summary of a feature's distribution.
    """
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "skew": skewness(x)
    }
