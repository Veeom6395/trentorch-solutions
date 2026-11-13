import numpy as np


def linear(input: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """
    input:  shape (batch_size, in_features)
    weight: shape (out_features, in_features)
    bias:   shape (out_features,), or None

    Returns:
        output with shape (batch_size, out_features)
    """
    output = input @ weight.T

    if bias is not None:
        output = output + bias

    return output
