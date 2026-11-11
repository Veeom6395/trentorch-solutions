import numpy as np

def solve(x):
    """Compute the stable L2 norm of a non-empty real vector."""
    x = np.asarray(x, dtype=float)

    scale = np.max(np.abs(x))

    if scale == 0.0:
        return 0.0

    return float(scale * np.sqrt(np.sum((x / scale) ** 2)))
