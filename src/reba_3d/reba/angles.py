"""
Angle calculation functions for REBA body segments.

Computes angles for neck, torso, shoulders, elbows, and knees
from 3D keypoint positions.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List

from reba_3d.core.geometry import (
    orthonormaliser_repere,
    extraire_angles_nautiques,
    calculate_angle_between_vectors,
)


def compute_neck_angles(
    positions: Dict[str, np.ndarray]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute neck angles (alpha, beta, gamma) relative to the torso.

    Constructs reference frames for head and torso, then calculates
    the nautical angles between them.

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
    rep_t = orthonormaliser_repere(Z_t_raw, X_t_raw)

    if rep_t is None:
        return None

    X_t, Y_t, Z_t = rep_t

    # Repère Buste
    Z_b_raw = positions["Neck"] - positions["MidHip"]
    X_b_raw = positions["RShoulder"] - positions["LShoulder"]
    rep_b = orthonormaliser_repere(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Angles nautiques du cou
    angles = extraire_angles_nautiques(X_t, Y_t, Z_t, X_b, Y_b, Z_b)
    if angles is None:
        return None

    alpha, beta, gamma = angles
    return float(abs(alpha)), float(beta), float(gamma)


def compute_torso_angles(
    positions: Dict[str, np.ndarray]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute torso angles (alpha, beta, gamma) relative to global frame.

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
    rep_b = orthonormaliser_repere(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Repère global
    X_g = np.array([-1, 0, 0], dtype=float)
    Y_g = np.array([0, -1, 0], dtype=float)
    Z_g = np.array([0, 0, 1], dtype=float)

    # Angles nautiques du buste par rapport au global
    angles = extraire_angles_nautiques(X_b, Y_b, Z_b, X_g, Y_g, Z_g)
    if angles is None:
        return None

    return angles


def compute_shoulder_angles_right(
    positions: Dict[str, np.ndarray]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute right shoulder angles relative to torso.

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
    rep_b = orthonormaliser_repere(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Repère Épaule Droite
    Z_ed_raw = positions["RShoulder"] - positions["RElbow"]
    X_ed_raw = positions["RShoulder"] - positions["LShoulder"]
    rep_ed = orthonormaliser_repere(Z_ed_raw, X_ed_raw)

    if rep_ed is None:
        return None

    X_ed, Y_ed, Z_ed = rep_ed

    # Angles nautiques de l'épaule
    return extraire_angles_nautiques(X_ed, Y_ed, Z_ed, X_b, Y_b, Z_b)


def compute_shoulder_angles_left(
    positions: Dict[str, np.ndarray]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute left shoulder angles relative to torso.

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
    rep_b = orthonormaliser_repere(Z_b_raw, X_b_raw)

    if rep_b is None:
        return None

    X_b, Y_b, Z_b = rep_b

    # Repère Épaule Gauche
    Z_eg_raw = positions["LShoulder"] - positions["LElbow"]
    X_eg_raw = positions["LShoulder"] - positions["RShoulder"]
    rep_eg = orthonormaliser_repere(Z_eg_raw, X_eg_raw)

    if rep_eg is None:
        return None

    X_eg, Y_eg, Z_eg = rep_eg

    return extraire_angles_nautiques(X_eg, Y_eg, Z_eg, X_b, Y_b, Z_b)


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
    v_vertical = positions["Neck"] - positions["MidHip"]

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
