"""
Geometric functions for reference frame orthonormalization and angle extraction.

Provides Gram-Schmidt orthonormalization and nautical angle calculations
for body segment reference frames.
"""

import numpy as np
from typing import Optional, Tuple

# Type alias for 3D vector
Vector3D = np.ndarray


def orthonormaliser_repere(
    v1: Vector3D,
    v2: Vector3D,
    eps: float = 1e-8
) -> Optional[Tuple[Vector3D, Vector3D, Vector3D]]:
    """
    Orthonormalize two vectors to form a right-handed orthonormal basis.

    Uses Gram-Schmidt orthonormalization to create an orthonormal basis
    from a primary axis (v1) and a secondary vector (v2).

    Args:
        v1: Primary vector (becomes Z axis)
        v2: Secondary vector (used to derive X axis)
        eps: Minimum threshold for vector norms (default: 1e-8)

    Returns:
        Tuple (X, Y, Z) of unit vectors forming an orthonormal basis,
        or None if the basis cannot be constructed (vectors too small or parallel)

    Example:
        >>> v1 = np.array([0, 0, 1])  # Pointing up
        >>> v2 = np.array([1, 0, 0])  # Pointing right
        >>> X, Y, Z = orthonormaliser_repere(v1, v2)
        >>> np.allclose(np.dot(X, Y), 0)  # X perpendicular to Y
        True
    """
    norm_v1 = np.linalg.norm(v1)
    if norm_v1 < eps:
        return None

    Z = v1 / norm_v1

    # Retrait de la composante de v2 sur Z (projection orthogonale)
    v2_proj = v2 - np.dot(v2, Z) * Z
    norm_v2_proj = np.linalg.norm(v2_proj)
    if norm_v2_proj < eps:
        return None

    X = v2_proj / norm_v2_proj

    Y = np.cross(Z, X)
    norm_y = np.linalg.norm(Y)
    if norm_y < eps:
        return None
    Y = Y / norm_y

    return X, Y, Z


def extraire_angles_nautiques(
    X_t: Vector3D,
    Y_t: Vector3D,
    Z_t: Vector3D,
    X_b: Vector3D,
    Y_b: Vector3D,
    Z_b: Vector3D,
    eps: float = 1e-8
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate nautical angles (alpha, beta, gamma) between two reference frames.

    Computes the rotation matrix R from base frame (b) to target frame (t),
    then extracts Euler angles in ZYX order.

    Args:
        X_t, Y_t, Z_t: Target frame basis vectors
        X_b, Y_b, Z_b: Base frame basis vectors
        eps: Minimum threshold for cos(beta) (default: 1e-8)

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if gimbal lock occurs

    Notes:
        - beta = -arcsin(R[2,0])
        - alpha = arctan2(R[2,1]/cos(beta), R[2,2]/cos(beta))
        - gamma = arctan2(R[0,1]/cos(beta), R[0,0]/cos(beta))
    """
    R = np.array([
        [np.dot(X_t, X_b), np.dot(X_t, Y_b), np.dot(X_t, Z_b)],
        [np.dot(Y_t, X_b), np.dot(Y_t, Y_b), np.dot(Y_t, Z_b)],
        [np.dot(Z_t, X_b), np.dot(Z_t, Y_b), np.dot(Z_t, Z_b)]
    ])

    # beta
    val_sin = np.clip(R[2, 0], -1.0, 1.0)
    beta = np.arcsin(-val_sin)

    cos_b = np.cos(beta)
    if abs(cos_b) < eps:
        # Gimbal lock - cannot determine unique angles
        return None

    # alpha
    alpha = np.arctan2(R[2, 1] / cos_b, R[2, 2] / cos_b)
    # gamma
    gamma = np.arctan2(R[1, 0] / cos_b, R[0, 0] / cos_b)

    return float(np.degrees(alpha)), float(np.degrees(beta)), float(np.degrees(gamma))


def calculate_angle_between_vectors(v1: Vector3D, v2: Vector3D) -> Optional[float]:
    """
    Calculate the angle in degrees between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Angle in degrees, or None if either vector is zero
    """
    if np.allclose(v1, 0.0) or np.allclose(v2, 0.0):
        return None

    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def normaliser_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-180, 180).

    Args:
        angle: Angle in degrees

    Returns:
        Normalized angle in degrees
    """
    return (angle + 180) % 360 - 180
