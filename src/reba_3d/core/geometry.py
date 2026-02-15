"""
Geometric functions for reference frame orthonormalization and angle extraction.

Provides Gram-Schmidt orthonormalization and nautical angle calculations
for body segment reference frames.
"""

import numpy as np
from typing import Optional, Tuple

# Type alias for 3D vector
Vector3D = np.ndarray


def orthonormalize_frame(
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
        >>> X, Y, Z = orthonormalize_frame(v1, v2)
        >>> np.allclose(np.dot(X, Y), 0)  # X perpendicular to Y
        True
    """
    v1 = np.asarray(v1, dtype=float).reshape(-1)
    v2 = np.asarray(v2, dtype=float).reshape(-1)

    norm_v1 = np.linalg.norm(v1)
    if norm_v1 < eps:
        return None

    Z = v1 / norm_v1

    # Remove v2 component along Z (orthogonal projection)
    v2_proj = v2 - np.dot(v2, Z) * Z
    norm_v2_proj = np.linalg.norm(v2_proj)
    if norm_v2_proj < eps:
        return None

    X = v2_proj / norm_v2_proj

    # Force X sign consistency with v2 (avoid flips)
    if np.dot(X, v2) < 0:
        X = -X

    Y = np.cross(Z, X)
    norm_y = np.linalg.norm(Y)
    if norm_y < eps:
        return None
    Y = Y / norm_y

    return X, Y, Z


def extract_nautical_angles(
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
        - gamma = arctan2(R[1,0]/cos(beta), R[0,0]/cos(beta))
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

    # gamma (correction: R[1,0] instead of R[0,1])

    gamma = np.arctan2(R[1, 0] / cos_b, R[0, 0] / cos_b)

    return float(np.degrees(alpha)), float(np.degrees(beta)), float(np.degrees(gamma))


def detect_view_orientation(
    positions: dict,
    threshold_ratio: float = 0.4
) -> str:
    """
    Detect view orientation (frontal vs profile) based on keypoint positions.

    Analyzes the ratio of shoulder width to neck-hip height to determine
    if the view is frontal (shoulders appear wide) or profile (shoulders appear narrow).

    Args:
        positions: Dictionary of keypoint positions with keys like 'RShoulder', 'LShoulder', etc.
        threshold_ratio: Ratio below which the view is considered profile. Default: 0.4
                        (shoulder width < 40% of torso height → profile view)

    Returns:
        'xy' for frontal view, 'yz' for profile view

    Example:
        - Frontal view: shoulders span wide horizontally → use XY plane
        - Profile view: shoulders appear close together → use YZ plane
    """
    required = ["RShoulder", "LShoulder", "Neck", "MidHip"]

    # Check if all required keypoints exist and are valid
    if not all(k in positions for k in required):
        return 'xy'  # Default to frontal

    if any(np.isnan(positions[k]).any() for k in required):
        return 'xy'  # Default to frontal

    # Calculate shoulder width (horizontal span)
    shoulder_vec = positions["RShoulder"] - positions["LShoulder"]
    shoulder_width_x = abs(shoulder_vec[0])  # Width in X (left-right)

    # Calculate torso height (vertical span)
    torso_vec = positions["Neck"] - positions["MidHip"]
    torso_height_y = abs(torso_vec[1])  # Height in Y (up-down)

    # Avoid division by zero
    if torso_height_y < 1e-6:
        return 'xy'

    # Calculate ratio
    ratio = shoulder_width_x / torso_height_y

    # If shoulders appear narrow relative to height → profile view
    if ratio < threshold_ratio:
        return 'yz'  # Profile: use Y (up/down) and Z (depth)
    else:
        return 'xy'  # Frontal: use X (left/right) and Y (up/down)


def extract_nautical_angles_2d(
    X_t: Vector3D,
    X_b: Vector3D,
    plane: str = 'xy',
    eps: float = 1e-8
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate nautical angles in 2D mode.

    Projects vectors to 2D plane and calculates signed angle using atan2.

    For frontal view (xy plane): Returns (0, 0, gamma) - measures abduction/adduction
    For profile view (yz plane): Returns (alpha, 0, 0) - measures flexion/extension

    Args:
        X_t: Target frame axis vector
        X_b: Base frame axis vector
        plane: Projection plane - 'xy' for frontal view, 'yz' for profile view
        eps: Minimum threshold for vector norms (default: 1e-8)

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if vectors are too small
        - Frontal (xy): (0.0, 0.0, gamma)
        - Profile (yz): (alpha, 0.0, 0.0)
    """
    def to_plane(v, plane_type):
        """Extract plane components from vector."""
        v = np.asarray(v, dtype=float).reshape(-1)
        if plane_type == 'yz':
            return v[1:3]  # Y, Z (sagittal plane for profile view)
        return v[:2]  # X, Y (frontal plane)

    def norm2(v):
        """Normalize 2D vector or return None if too small."""
        n = np.linalg.norm(v)
        return None if n < eps else v / n

    # Normalize 2D projections in the specified plane
    Xt = norm2(to_plane(X_t, plane))
    Xb = norm2(to_plane(X_b, plane))

    if Xt is None or Xb is None:
        return None

    # Calculate signed angle: atan2(det, dot)
    det = Xb[0] * Xt[1] - Xb[1] * Xt[0]
    dot = float(np.dot(Xb, Xt))
    angle = float(np.degrees(np.arctan2(det, dot)))

    if plane == 'yz':
        # Profile view: angle represents flexion/extension (alpha)
        return angle, 0.0, 0.0
    else:
        # Frontal view: angle represents abduction/rotation (gamma)
        return 0.0, 0.0, angle


def estimate_lateral_bend(
    positions: dict,
    frontal_weight: float = 1.0,
    eps: float = 1e-8
) -> float:
    """
    Estimate lateral trunk bending (beta) from shoulder vertical asymmetry.

    In a frontal view, lateral bending shifts one shoulder up relative to the other.
    The angle is attenuated by frontal_weight so it degrades gracefully in profile views.

    Args:
        positions: Dictionary with 'RShoulder' and 'LShoulder' 3D positions.
        frontal_weight: Weight in [0, 1] indicating how frontal the view is.
        eps: Minimum shoulder distance threshold.

    Returns:
        Estimated lateral bend angle in degrees (positive = right shoulder higher).
    """
    required = ["RShoulder", "LShoulder"]
    if any(k not in positions or np.isnan(positions[k]).any() for k in required):
        return 0.0

    dy = float(positions["RShoulder"][1] - positions["LShoulder"][1])
    dist = float(np.linalg.norm(positions["RShoulder"] - positions["LShoulder"]))
    if dist < eps:
        return 0.0

    raw_beta = np.degrees(np.arcsin(np.clip(dy / dist, -1.0, 1.0)))
    return raw_beta * frontal_weight


def estimate_axial_rotation(
    positions: dict,
    neutral_shoulder_width: Optional[float] = None,
    eps: float = 1e-8
) -> float:
    """
    Estimate axial trunk rotation from projected shoulder width vs neutral width.

    When the torso rotates around the spine, the projected shoulder width
    shrinks by cos(rotation_angle).

    Args:
        positions: Dictionary with 'RShoulder' and 'LShoulder' 3D positions.
        neutral_shoulder_width: Shoulder width (in X) measured during neutral pose.
        eps: Minimum width threshold.

    Returns:
        Estimated rotation magnitude in degrees (always >= 0).
    """
    if neutral_shoulder_width is None or neutral_shoulder_width < eps:
        return 0.0

    required = ["RShoulder", "LShoulder"]
    if any(k not in positions or np.isnan(positions[k]).any() for k in required):
        return 0.0

    current_width = abs(float(positions["RShoulder"][0] - positions["LShoulder"][0]))
    ratio = np.clip(current_width / neutral_shoulder_width, 0.0, 1.0)
    return float(np.degrees(np.arccos(ratio)))


def _compute_view_ratio(
    positions: dict,
    eps: float = 1e-6
) -> float:
    """
    Compute the shoulder-width-to-torso-height ratio used for view classification.

    Args:
        positions: Dictionary of keypoint positions.
        eps: Minimum torso height threshold.

    Returns:
        Ratio value (higher = more frontal). Returns 1.0 on error (defaults to frontal).
    """
    required = ["RShoulder", "LShoulder", "Neck", "MidHip"]
    if any(k not in positions or np.isnan(positions[k]).any() for k in required):
        return 1.0

    shoulder_width_x = abs(float(positions["RShoulder"][0] - positions["LShoulder"][0]))
    torso_height_y = abs(float(positions["Neck"][1] - positions["MidHip"][1]))

    if torso_height_y < eps:
        return 1.0

    return shoulder_width_x / torso_height_y


def extract_nautical_angles_2d_enhanced(
    X_t: Vector3D,
    X_b: Vector3D,
    positions: dict,
    neutral_shoulder_width: Optional[float] = None,
    threshold: float = 0.4,
    eps: float = 1e-8
) -> Optional[Tuple[float, float, float]]:
    """
    Enhanced 2D nautical angle extraction with multi-plane fusion and inferred components.

    Improvements over extract_nautical_angles_2d:
    - Strategy 1: Blends angles from both XY (frontal) and YZ (profile) planes
      using a continuous weight instead of a binary view switch.
    - Strategy 2: Estimates lateral bending (beta) from shoulder vertical asymmetry.
    - Strategy 3: Estimates axial rotation from projected shoulder width shrinkage.

    Args:
        X_t: Target frame axis vector.
        X_b: Base frame axis vector.
        positions: Dictionary of keypoint positions (needs RShoulder, LShoulder, Neck, MidHip).
        neutral_shoulder_width: Shoulder width measured during neutral pose for rotation estimation.
        threshold: View ratio threshold for full frontal classification.
        eps: Minimum norm threshold.

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails.
    """
    # Compute view ratio and continuous frontal weight
    ratio = _compute_view_ratio(positions)
    weight_frontal = min(1.0, ratio / threshold)  # 0 = pure profile, 1 = pure frontal

    # Compute angles in both planes
    xy_result = extract_nautical_angles_2d(X_t, X_b, plane='xy', eps=eps)
    yz_result = extract_nautical_angles_2d(X_t, X_b, plane='yz', eps=eps)

    if xy_result is None and yz_result is None:
        return None

    # Extract per-plane angles (default to 0 if a plane fails)
    yz_alpha = yz_result[0] if yz_result is not None else 0.0
    xy_gamma = xy_result[2] if xy_result is not None else 0.0

    # Strategy 1: Blend alpha (profile) and gamma (frontal) by view weight
    alpha = (1.0 - weight_frontal) * yz_alpha
    gamma = weight_frontal * xy_gamma

    # Strategy 2: Estimate lateral bend from shoulder asymmetry
    beta = estimate_lateral_bend(positions, frontal_weight=weight_frontal)

    # Strategy 3: Estimate axial rotation from shoulder width
    rotation = estimate_axial_rotation(positions, neutral_shoulder_width)
    # Add rotation magnitude to gamma (unsigned — direction is unknown in 2D)
    # Use sign of gamma to orient the rotation contribution
    if abs(gamma) > eps:
        gamma = gamma + np.sign(gamma) * rotation
    else:
        gamma = gamma + rotation

    return float(alpha), float(beta), float(gamma)


def extract_shoulder_angles_dotproduct(
    Z_arm: Vector3D,
    X_b: Vector3D,
    Y_b: Vector3D,
    Z_b: Vector3D,
    eps: float = 1e-8
) -> Optional[Tuple[float, float, float]]:
    """
    Compute shoulder angles via dot products — no gimbal lock.

    Only needs the upper arm vector and the torso frame.
    No Gram-Schmidt on the shoulder, no Euler decomposition.

    alpha and beta are mathematically identical to the Euler ZYX extraction
    when cos(beta) > 0. At gimbal lock (arm perfectly lateral), alpha
    gracefully returns 0 instead of None.

    Args:
        Z_arm: Upper arm vector (Shoulder - Elbow), does not need to be normalized
        X_b, Y_b, Z_b: Torso frame basis vectors (must be orthonormal)
        eps: Minimum norm threshold for Z_arm

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if Z_arm is zero-length.
        gamma is always 0 (axial rotation is not determinable from 2 keypoints).
    """
    Z_arm = np.asarray(Z_arm, dtype=float).reshape(-1)
    norm = np.linalg.norm(Z_arm)
    if norm < eps:
        return None
    Z_arm = Z_arm / norm

    # Direction cosines in the torso frame
    c_x = float(np.dot(Z_arm, X_b))  # lateral component
    c_y = float(np.dot(Z_arm, Y_b))  # anterior component
    c_z = float(np.dot(Z_arm, Z_b))  # axial component (along spine)

    # alpha (flexion/extension) — equivalent to Euler alpha
    alpha = float(np.degrees(np.arctan2(c_y, c_z)))

    # beta (abduction) — equivalent to Euler beta
    beta = float(np.degrees(-np.arcsin(np.clip(c_x, -1.0, 1.0))))

    # gamma = 0 (axial rotation not determinable with only 2 keypoints)
    gamma = 0.0

    return alpha, beta, gamma


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


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-180, 180).

    Args:
        angle: Angle in degrees

    Returns:
        Normalized angle in degrees
    """
    return (angle + 180) % 360 - 180
