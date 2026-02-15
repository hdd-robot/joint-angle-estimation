"""
Angle calculation functions for REBA body segments.

Computes angles for neck, torso, shoulders, elbows, and knees
from 3D keypoint positions.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List

from reba_3d.core.geometry import (
    orthonormalize_frame,
    extract_nautical_angles,
    extract_nautical_angles_2d,
    extract_nautical_angles_2d_enhanced,
    extract_shoulder_angles_dotproduct,
    calculate_angle_between_vectors,
    detect_view_orientation,
)


def compute_neck_angles(
    positions: Dict[str, np.ndarray]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute neck angles (alpha, beta, gamma) relative to the torso using 3D nautical angles.

    Constructs reference frames for head and torso, then calculates
    the full 3D nautical angles between them.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails
    """
    required = ["Neck", "Nose", "REye", "LEye", "MidHip", "RShoulder", "LShoulder"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    # Repère Tête
    Z_t_raw = positions["Neck"] - positions["Nose"]
    X_t_raw = positions["REye"] - positions["LEye"]
    rep_t = orthonormalize_frame(Z_t_raw, X_t_raw)

    if rep_t is None:
        return None

    X_t, Y_t, Z_t = rep_t

    # Repère Buste
    Z_b_raw = positions["Neck"] - positions["MidHip"]
    X_b_raw = positions["RShoulder"] - positions["LShoulder"]
    rep_b = orthonormalize_frame(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Angles nautiques 3D du cou
    angles = extract_nautical_angles(X_t, Y_t, Z_t, X_b, Y_b, Z_b)
    if angles is None:
        return None

    alpha, beta, gamma = angles
    return float(abs(alpha)), float(beta), float(gamma)


def compute_neck_angles_2d(
    positions: Dict[str, np.ndarray],
    neutral_shoulder_width: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute neck angles using enhanced 2D planar projection with multi-plane fusion.

    Constructs reference frames for head and torso, then calculates
    angles using continuous blending between frontal and profile planes.

    Args:
        positions: Dictionary of smoothed keypoint positions
        neutral_shoulder_width: Shoulder width from neutral pose for rotation estimation

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails
    """
    required = ["Neck", "Nose", "REye", "LEye", "MidHip", "RShoulder", "LShoulder"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    # Repère Tête
    Z_t_raw = positions["Neck"] - positions["Nose"]
    X_t_raw = positions["REye"] - positions["LEye"]
    rep_t = orthonormalize_frame(Z_t_raw, X_t_raw)

    if rep_t is None:
        return None

    X_t, Y_t, Z_t = rep_t

    # Repère Buste
    Z_b_raw = positions["Neck"] - positions["MidHip"]
    X_b_raw = positions["RShoulder"] - positions["LShoulder"]
    rep_b = orthonormalize_frame(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Enhanced 2D angles with multi-plane fusion
    angles = extract_nautical_angles_2d_enhanced(
        X_t, X_b, positions,
        neutral_shoulder_width=neutral_shoulder_width,
    )
    if angles is None:
        return None

    alpha, beta, gamma = angles
    return float(abs(alpha)), float(beta), float(gamma)


def compute_torso_angles(
    positions: Dict[str, np.ndarray]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute torso angles (alpha, beta, gamma) relative to global frame using 3D nautical angles.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails
    """
    required = ["Neck", "MidHip", "RShoulder", "LShoulder"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    # Repère Buste
    Z_b_raw = positions["Neck"] - positions["MidHip"]
    X_b_raw = positions["RShoulder"] - positions["LShoulder"]
    rep_b = orthonormalize_frame(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Repère global
    X_g = np.array([-1, 0, 0], dtype=float)
    Y_g = np.array([0, -1, 0], dtype=float)
    Z_g = np.array([0, 0, 1], dtype=float)

    # Angles nautiques 3D du buste par rapport au global
    angles = extract_nautical_angles(X_b, Y_b, Z_b, X_g, Y_g, Z_g)
    if angles is None:
        return None

    return angles


def compute_torso_angles_2d(
    positions: Dict[str, np.ndarray],
    neutral_shoulder_width: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute torso angles using enhanced 2D planar projection with multi-plane fusion.

    Improvements over basic 2D:
    - Multi-plane fusion: blends frontal and profile angles continuously
    - Lateral bend (beta) estimated from shoulder vertical asymmetry
    - Axial rotation estimated from projected shoulder width vs neutral

    Args:
        positions: Dictionary of smoothed keypoint positions
        neutral_shoulder_width: Shoulder width from neutral pose for rotation estimation

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails
    """
    required = ["Neck", "MidHip", "RShoulder", "LShoulder"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    # Repère Buste
    Z_b_raw = positions["Neck"] - positions["MidHip"]
    X_b_raw = positions["RShoulder"] - positions["LShoulder"]
    rep_b = orthonormalize_frame(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Repère global (vue frontale XY)
    X_g = np.array([-1, 0, 0], dtype=float)

    # Enhanced 2D angles with multi-plane fusion + beta + rotation
    angles = extract_nautical_angles_2d_enhanced(
        X_b, X_g, positions,
        neutral_shoulder_width=neutral_shoulder_width,
    )
    if angles is None:
        return None

    return angles


def compute_shoulder_angles_right(
    positions: Dict[str, np.ndarray]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute right shoulder angles relative to torso using 3D nautical angles.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails
    """
    required = ["RShoulder", "RElbow", "LShoulder", "Neck", "MidHip"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    # Repère Buste
    Z_b_raw = positions["Neck"] - positions["MidHip"]
    X_b_raw = positions["RShoulder"] - positions["LShoulder"]
    rep_b = orthonormalize_frame(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Vecteur bras (pas besoin de repère complet — produits scalaires uniquement)
    Z_ed_raw = positions["RShoulder"] - positions["RElbow"]
    return extract_shoulder_angles_dotproduct(Z_ed_raw, X_b, Y_b, Z_b)


def _compute_shoulder_angles_2d(
    positions: Dict[str, np.ndarray],
    side: str,
    neutral_shoulder_width: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute shoulder angles using enhanced 2D planar projection with multi-plane fusion.

    Measures the angle between the upper arm direction (Shoulder→Elbow) and the
    torso axis (MidHip→Neck) with continuous blending between frontal and profile planes.

    Args:
        positions: Dictionary of smoothed keypoint positions
        side: 'R' for right shoulder, 'L' for left shoulder
        neutral_shoulder_width: Shoulder width from neutral pose for rotation estimation

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails
    """
    shoulder_key = f"{side}Shoulder"
    elbow_key = f"{side}Elbow"
    required = [shoulder_key, elbow_key, "Neck", "MidHip", "RShoulder", "LShoulder"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    # Raw anatomical vectors
    upper_arm = positions[elbow_key] - positions[shoulder_key]
    torso = positions["Neck"] - positions["MidHip"]

    # Enhanced 2D angles with multi-plane fusion
    return extract_nautical_angles_2d_enhanced(
        upper_arm, torso, positions,
        neutral_shoulder_width=neutral_shoulder_width,
    )


def compute_shoulder_angles_right_2d(
    positions: Dict[str, np.ndarray],
    neutral_shoulder_width: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute right shoulder angles using enhanced 2D planar projection.

    Args:
        positions: Dictionary of smoothed keypoint positions
        neutral_shoulder_width: Shoulder width from neutral pose for rotation estimation

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails
    """
    return _compute_shoulder_angles_2d(positions, side='R', neutral_shoulder_width=neutral_shoulder_width)


def compute_shoulder_angles_left(
    positions: Dict[str, np.ndarray]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute left shoulder angles relative to torso using 3D nautical angles.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails
    """
    required = ["LShoulder", "LElbow", "RShoulder", "Neck", "MidHip"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    # Repère Buste
    Z_b_raw = positions["Neck"] - positions["MidHip"]
    X_b_raw = positions["RShoulder"] - positions["LShoulder"]
    rep_b = orthonormalize_frame(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Vecteur bras (pas besoin de repère complet — produits scalaires uniquement)
    Z_eg_raw = positions["LShoulder"] - positions["LElbow"]
    return extract_shoulder_angles_dotproduct(Z_eg_raw, X_b, Y_b, Z_b)


def compute_shoulder_angles_left_2d(
    positions: Dict[str, np.ndarray],
    neutral_shoulder_width: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute left shoulder angles using enhanced 2D planar projection.

    Args:
        positions: Dictionary of smoothed keypoint positions
        neutral_shoulder_width: Shoulder width from neutral pose for rotation estimation

    Returns:
        Tuple (alpha, beta, gamma) in degrees, or None if calculation fails
    """
    return _compute_shoulder_angles_2d(positions, side='L', neutral_shoulder_width=neutral_shoulder_width)


def compute_shoulder_elevation_right(
    positions: Dict[str, np.ndarray]
) -> Optional[float]:
    """
    Compute right shoulder elevation angle.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Elevation angle in degrees, or None if calculation fails
    """
    required = ["RShoulder", "Neck", "MidHip"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    v_epaule = positions["RShoulder"] - positions["Neck"]
    v_vertical = positions["MidHip"] - positions["Neck"]

    return calculate_angle_between_vectors(v_epaule, v_vertical)


def compute_shoulder_elevation_left(
    positions: Dict[str, np.ndarray]
) -> Optional[float]:
    """
    Compute left shoulder elevation angle.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Elevation angle in degrees, or None if calculation fails
    """
    required = ["LShoulder", "Neck", "MidHip"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    v_epaule = positions["LShoulder"] - positions["Neck"]
    v_vertical = positions["MidHip"] - positions["Neck"]

    return calculate_angle_between_vectors(v_epaule, v_vertical)


def compute_elbow_angle_right(
    positions: Dict[str, np.ndarray]
) -> Optional[float]:
    """
    Compute right elbow angle.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Elbow angle in degrees, or None if calculation fails
    """
    required = ["RShoulder", "RElbow", "RWrist"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    v_avantbras = positions["RWrist"] - positions["RElbow"]
    v_bras = positions["RShoulder"] - positions["RElbow"]

    return calculate_angle_between_vectors(v_avantbras, v_bras)


def compute_elbow_angle_left(
    positions: Dict[str, np.ndarray]
) -> Optional[float]:
    """
    Compute left elbow angle.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Elbow angle in degrees, or None if calculation fails
    """
    required = ["LShoulder", "LElbow", "LWrist"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    v_avantbras = positions["LWrist"] - positions["LElbow"]
    v_bras = positions["LShoulder"] - positions["LElbow"]

    return calculate_angle_between_vectors(v_avantbras, v_bras)


def compute_knee_angle_right(
    positions: Dict[str, np.ndarray]
) -> Optional[float]:
    """
    Compute right knee angle.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Knee angle in degrees, or None if calculation fails
    """
    required = ["RHip", "RKnee", "RAnkle"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    v_knee_to_hip = positions["RHip"] - positions["RKnee"]
    v_knee_to_ankle = positions["RAnkle"] - positions["RKnee"]

    return calculate_angle_between_vectors(v_knee_to_hip, v_knee_to_ankle)


def compute_knee_angle_left(
    positions: Dict[str, np.ndarray]
) -> Optional[float]:
    """
    Compute left knee angle.

    Args:
        positions: Dictionary of smoothed keypoint positions

    Returns:
        Knee angle in degrees, or None if calculation fails
    """
    required = ["LHip", "LKnee", "LAnkle"]
    if any(np.isnan(positions.get(kp, [np.nan])).any() for kp in required):
        return None

    v_knee_to_hip = positions["LHip"] - positions["LKnee"]
    v_knee_to_ankle = positions["LAnkle"] - positions["LKnee"]

    return calculate_angle_between_vectors(v_knee_to_hip, v_knee_to_ankle)


def compute_feet_contact(
    positions: Dict[str, np.ndarray],
    r_ankle_mean: float,
    l_ankle_mean: float,
    threshold: float = 0.10
) -> str:
    """
    Determine which feet are in contact with the ground.

    Args:
        positions: Dictionary of smoothed keypoint positions
        r_ankle_mean: Mean Y position of right ankle from calibration
        l_ankle_mean: Mean Y position of left ankle from calibration
        threshold: Distance threshold for contact detection (meters)

    Returns:
        Contact state: "OK" (both), "DROIT" (right only), "GAUCHE" (left only), or "404" (error)
    """
    if (np.isnan(positions.get("RAnkle", [np.nan])).any() or
        np.isnan(positions.get("LAnkle", [np.nan])).any()):
        return "404"

    r_y = float(positions["RAnkle"][1])
    l_y = float(positions["LAnkle"][1])

    r_contact = abs(r_y - r_ankle_mean) < threshold
    l_contact = abs(l_y - l_ankle_mean) < threshold

    if r_contact and l_contact:
        return "OK"
    elif r_contact:
        return "DROIT"
    elif l_contact:
        return "GAUCHE"
    return "404"
