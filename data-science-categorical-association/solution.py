
import numpy as np


def contingency_table(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a, b: 1D arrays of hashable labels, same length

    Returns:
        An int 2D array: rows are the sorted unique labels of a, columns
        the sorted unique labels of b, and entry (i, j) counts the rows
        where a is the i-th label and b is the j-th.
    """
    unique_a = np.unique(a)
    unique_b = np.unique(b)

    table = np.zeros((len(unique_a), len(unique_b)), dtype=int)

    for i, label_a in enumerate(unique_a):
        for j, label_b in enumerate(unique_b):
            table[i, j] = np.sum((a == label_a) & (b == label_b))

    return table


def expected_counts(table: np.ndarray) -> np.ndarray:
    """
    Returns a float array of the same shape as table with entry (i, j)
    equal to row_total_i * column_total_j / grand_total.
    """
    table = np.asarray(table, dtype=float)

    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    grand_total = table.sum()

    if grand_total == 0:
        return np.zeros(table.shape, dtype=float)

    return (row_totals * col_totals) / grand_total


def chi_square_statistic(table: np.ndarray) -> float:
    """
    Returns the sum over cells of (observed - expected) ** 2 / expected,
    skipping cells whose expected count is 0.
    """
    observed = np.asarray(table, dtype=float)
    expected = expected_counts(observed)

    mask = expected > 0

    chi2 = np.sum(
        (observed[mask] - expected[mask]) ** 2 / expected[mask]
    )

    return float(chi2)


def cramers_v(table: np.ndarray) -> float:
    """
    Returns sqrt(chi2 / (n * min(r - 1, c - 1))) for a table with r rows,
    c columns and grand total n. Returns 0.0 when min(r - 1, c - 1) is 0
    or n is 0.
    """
    table = np.asarray(table)

    r, c = table.shape
    n = table.sum()

    denominator_factor = min(r - 1, c - 1)

    if denominator_factor <= 0 or n == 0:
        return 0.0

    chi2 = chi_square_statistic(table)

    return float(np.sqrt(chi2 / (n * denominator_factor)))
