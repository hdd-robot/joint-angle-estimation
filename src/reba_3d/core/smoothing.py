"""
Smoothing functions for keypoint trajectory processing.

Provides local polynomial regression for noise reduction in keypoint data.
"""

import numpy as np
from typing import Optional


def local_polynomial_regression(
    x_frames: np.ndarray,
    y_values: np.ndarray,
    order: int = 3,
    eps: float = 1e-8
) -> float:
    """
    Fit a polynomial and return the predicted value at the center of the window.

    Performs local polynomial regression on a window of data points,
    fitting a polynomial of specified order and returning the value
    at the center of the x_frames range.

    Args:
        x_frames: Array of frame indices (x values)
        y_values: Array of corresponding values (y values)
        order: Polynomial order for fitting (default: 3)
        eps: Tolerance for checking if x range is zero (default: 1e-8)

    Returns:
        Predicted value at the center of x_frames, or np.nan if insufficient data

    Example:
        >>> x = np.array([0, 1, 2, 3, 4])
        >>> y = np.array([0.1, 0.9, 2.1, 2.9, 4.1])
        >>> local_polynomial_regression(x, y, order=1)  # Linear fit
        2.02
    """
    # Masquer les valeurs NaN
    mask = ~np.isnan(y_values)
    x_clean = x_frames[mask]
    y_clean = y_values[mask]

    # Vérifier qu'il y a assez de points
    if len(x_clean) < order + 1:
        return np.nan

    # Vérifier que les x ne sont pas tous identiques
    if np.allclose(x_clean.max(), x_clean.min(), atol=eps):
        return np.nan

    # Ajuster le polynôme et évaluer au centre
    coefs = np.polyfit(x_clean, y_clean, order)
    return float(np.polyval(coefs, np.mean(x_clean)))


def smooth_keypoints_window(
    df_window,
    keypoints: list,
    poly_order: int = 2
) -> dict:
    """
    Apply polynomial smoothing to all keypoints in a window.

    Args:
        df_window: DataFrame containing keypoint data for the window
        keypoints: List of keypoint names to smooth
        poly_order: Polynomial order for regression (default: 2)

    Returns:
        Dictionary mapping keypoint names to smoothed 3D positions
    """
    x_frames = df_window["frame"].to_numpy()
    block_positions = {}

    for kp in keypoints:
        x_vals = df_window[f"{kp}_x"].to_numpy()
        y_vals = df_window[f"{kp}_y"].to_numpy()
        z_vals = df_window[f"{kp}_z"].to_numpy()

        x_pred = local_polynomial_regression(x_frames, x_vals, order=poly_order)
        y_pred = local_polynomial_regression(x_frames, y_vals, order=poly_order)
        z_pred = local_polynomial_regression(x_frames, z_vals, order=poly_order)

        block_positions[kp] = np.array([x_pred, y_pred, z_pred], dtype=float)

    return block_positions
