import numpy as np

def solve(x):
    """Return the unit vector in the same direction as x."""
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)

    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")

    return x / norm
