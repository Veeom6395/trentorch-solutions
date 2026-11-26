
import numpy as np


def mse_gradient(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:

    # Forward pass
    prediction = input @ weight.T

    if bias is not None:
        prediction = prediction + bias

    # Gradient of MSE with respect to prediction
    batch_size, out_features = target.shape

    grad_prediction = (
        2.0 * (prediction - target)
        / (batch_size * out_features)
    )

    # Gradient with respect to weight
    grad_weight = grad_prediction.T @ input

    # Gradient with respect to bias
    if bias is not None:
        grad_bias = np.sum(grad_prediction, axis=0)
    else:
        grad_bias = None

    return grad_weight, grad_bias
