"""
Simple angle calculations from OpenPose keypoints.

Calculates joint angles using vector dot products, similar to the original code.
Used for calibration and real-time REBA assessment.
"""

import numpy as np
from typing import Dict, List, Optional, Any

# Indices des keypoints OpenPose BODY_25
KEYPOINT_INDICES = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "MidHip": 8,
    "RHip": 9,
    "RKnee": 10,
    "RAnkle": 11,
    "LHip": 12,
    "LKnee": 13,
    "LAnkle": 14,
    "REye": 15,
    "LEye": 16,
    "REar": 17,
    "LEar": 18,
    "LBigToe": 19,
    "LSmallToe": 20,
    "LHeel": 21,
    "RBigToe": 22,
    "RSmallToe": 23,
    "RHeel": 24,
}


def get_keypoint_3d(keypoints: np.ndarray, name: str) -> Optional[np.ndarray]:
    """
    Extract 3D coordinates for a named keypoint.

    Args:
        keypoints: OpenPose keypoints array (25x3 or 25x4 with confidence)
        name: Keypoint name (e.g., "Nose", "Neck")

    Returns:
        3D coordinates [x, y, z] or None if invalid
    """
    if name not in KEYPOINT_INDICES:
        return None

    idx = KEYPOINT_INDICES[name]
    if idx >= len(keypoints):
        return None

    kp = keypoints[idx]

    # Handle different formats (x,y,conf) or (x,y,z) or (x,y,z,conf)
    if len(kp) >= 3:
        # Check if it's 2D with confidence or 3D
        if len(kp) == 3:
            # Could be (x, y, conf) or (x, y, z)
            # If z value is between 0 and 1, it's probably confidence
            if 0 <= kp[2] <= 1:
                return None  # 2D only
            return np.array([kp[0], kp[1], kp[2]])
        else:
            # (x, y, z, conf) format
            return np.array([kp[0], kp[1], kp[2]])

    return None


def calculate_angle(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate angle at vertex between vectors (vertex->p1) and (vertex->p2).

    Args:
        p1: First point
        vertex: Vertex point (center of angle)
        p2: Second point

    Returns:
        Angle in degrees (0-180)
    """
    v1 = p1 - vertex
    v2 = p2 - vertex

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-8 or norm_v2 < 1e-8:
        return 0.0

    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Use atan2 for better numerical stability
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    angle_rad = np.arctan2(sin_theta, cos_theta)

    return float(np.degrees(angle_rad))


def calculate_angles_from_keypoints(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Calculate all body segment angles from OpenPose keypoints.

    Based on original code: CALCUL ANGLE GENERAL 3D.py

    Args:
        keypoints: OpenPose keypoints array

    Returns:
        Dictionary of angles:
        - cou: Neck angle (Nose-Neck-MidHip)
        - epaule_droite: Right shoulder (MidHip-RShoulder-RElbow)
        - epaule_gauche: Left shoulder (MidHip-LShoulder-LElbow)
        - coude_droit: Right elbow (RShoulder-RElbow-RWrist)
        - coude_gauche: Left elbow (LShoulder-LElbow-LWrist)
        - genou_droit: Right knee (RHip-RKnee-RAnkle)
        - genou_gauche: Left knee (LHip-LKnee-LAnkle)
        - hanche: Hip angle (Neck-MidHip-LHip)
    """
    angles = {}

    # Extract keypoints
    nose = get_keypoint_3d(keypoints, "Nose")
    neck = get_keypoint_3d(keypoints, "Neck")
    mid_hip = get_keypoint_3d(keypoints, "MidHip")

    r_shoulder = get_keypoint_3d(keypoints, "RShoulder")
    r_elbow = get_keypoint_3d(keypoints, "RElbow")
    r_wrist = get_keypoint_3d(keypoints, "RWrist")
    r_hip = get_keypoint_3d(keypoints, "RHip")
    r_knee = get_keypoint_3d(keypoints, "RKnee")
    r_ankle = get_keypoint_3d(keypoints, "RAnkle")

    l_shoulder = get_keypoint_3d(keypoints, "LShoulder")
    l_elbow = get_keypoint_3d(keypoints, "LElbow")
    l_wrist = get_keypoint_3d(keypoints, "LWrist")
    l_hip = get_keypoint_3d(keypoints, "LHip")
    l_knee = get_keypoint_3d(keypoints, "LKnee")
    l_ankle = get_keypoint_3d(keypoints, "LAnkle")

    # COU (Neck): angle Nose-Neck-MidHip
    if nose is not None and neck is not None and mid_hip is not None:
        angles["cou"] = calculate_angle(nose, neck, mid_hip)

    # ÉPAULE DROITE: angle MidHip-RShoulder-RElbow
    if mid_hip is not None and r_shoulder is not None and r_elbow is not None:
        angles["epaule_droite"] = calculate_angle(mid_hip, r_shoulder, r_elbow)

    # ÉPAULE GAUCHE: angle MidHip-LShoulder-LElbow
    if mid_hip is not None and l_shoulder is not None and l_elbow is not None:
        angles["epaule_gauche"] = calculate_angle(mid_hip, l_shoulder, l_elbow)

    # COUDE DROIT: angle RShoulder-RElbow-RWrist
    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        angles["coude_droit"] = calculate_angle(r_shoulder, r_elbow, r_wrist)

    # COUDE GAUCHE: angle LShoulder-LElbow-LWrist
    if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
        angles["coude_gauche"] = calculate_angle(l_shoulder, l_elbow, l_wrist)

    # GENOU DROIT: angle RHip-RKnee-RAnkle
    if r_hip is not None and r_knee is not None and r_ankle is not None:
        angles["genou_droit"] = calculate_angle(r_hip, r_knee, r_ankle)

    # GENOU GAUCHE: angle LHip-LKnee-LAnkle
    if l_hip is not None and l_knee is not None and l_ankle is not None:
        angles["genou_gauche"] = calculate_angle(l_hip, l_knee, l_ankle)

    # HANCHE: angle Neck-MidHip-LHip
    if neck is not None and mid_hip is not None and l_hip is not None:
        angles["hanche"] = calculate_angle(neck, mid_hip, l_hip)

    return angles


def calculate_angles_from_keypoints_2d(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Calculate angles from 2D keypoints (x, y, confidence).

    Falls back to 2D calculation when depth is not available.

    Args:
        keypoints: OpenPose keypoints array (25x3 with x, y, confidence)

    Returns:
        Dictionary of angles (same as calculate_angles_from_keypoints)
    """
    angles = {}

    def get_2d(name: str) -> Optional[np.ndarray]:
        if name not in KEYPOINT_INDICES:
            return None
        idx = KEYPOINT_INDICES[name]
        if idx >= len(keypoints):
            return None
        kp = keypoints[idx]
        if len(kp) >= 2 and kp[0] != 0 and kp[1] != 0:
            # Use x, y only, set z to 0
            return np.array([kp[0], kp[1], 0.0])
        return None

    # Extract 2D keypoints
    nose = get_2d("Nose")
    neck = get_2d("Neck")
    mid_hip = get_2d("MidHip")

    r_shoulder = get_2d("RShoulder")
    r_elbow = get_2d("RElbow")
    r_wrist = get_2d("RWrist")
    r_hip = get_2d("RHip")
    r_knee = get_2d("RKnee")
    r_ankle = get_2d("RAnkle")

    l_shoulder = get_2d("LShoulder")
    l_elbow = get_2d("LElbow")
    l_wrist = get_2d("LWrist")
    l_hip = get_2d("LHip")
    l_knee = get_2d("LKnee")
    l_ankle = get_2d("LAnkle")

    # Calculate angles (same as 3D version)
    if nose is not None and neck is not None and mid_hip is not None:
        angles["cou"] = calculate_angle(nose, neck, mid_hip)

    if mid_hip is not None and r_shoulder is not None and r_elbow is not None:
        angles["epaule_droite"] = calculate_angle(mid_hip, r_shoulder, r_elbow)

    if mid_hip is not None and l_shoulder is not None and l_elbow is not None:
        angles["epaule_gauche"] = calculate_angle(mid_hip, l_shoulder, l_elbow)

    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        angles["coude_droit"] = calculate_angle(r_shoulder, r_elbow, r_wrist)

    if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
        angles["coude_gauche"] = calculate_angle(l_shoulder, l_elbow, l_wrist)

    if r_hip is not None and r_knee is not None and r_ankle is not None:
        angles["genou_droit"] = calculate_angle(r_hip, r_knee, r_ankle)

    if l_hip is not None and l_knee is not None and l_ankle is not None:
        angles["genou_gauche"] = calculate_angle(l_hip, l_knee, l_ankle)

    if neck is not None and mid_hip is not None and l_hip is not None:
        angles["hanche"] = calculate_angle(neck, mid_hip, l_hip)

    return angles


def moyenne_sans_zero(liste: List[float], skip_n: int = 1) -> float:
    """
    Calculate mean excluding initial values, NaN and zeros.

    Args:
        liste: List of values
        skip_n: Number of initial values to skip (default: 1)

    Returns:
        Mean value or 0.0 if all values filtered
    """
    if len(liste) <= skip_n:
        return 0.0

    liste_filtree = liste[skip_n:]
    valeurs_valides = [v for v in liste_filtree if v != 0.0 and not np.isnan(v)]

    if valeurs_valides:
        return float(np.mean(valeurs_valides))
    return 0.0


def compute_calibration_offsets(
    angles_list: List[Dict[str, float]],
    window_size: int = 30,
    skip_windows: int = 1
) -> Dict[str, float]:
    """
    Compute calibration offsets from a list of angle measurements.

    Groups angles into windows, calculates mean per window,
    then averages windows (skipping first ones).

    Args:
        angles_list: List of angle dictionaries from each frame
        window_size: Number of frames per window (default: 30)
        skip_windows: Number of initial windows to skip (default: 1)

    Returns:
        Dictionary of offset values for each angle type
    """
    if not angles_list:
        return {}

    # Collect all angle types present
    all_keys = set()
    for angles in angles_list:
        all_keys.update(angles.keys())

    # Group by angle type
    angle_series = {key: [] for key in all_keys}
    for angles in angles_list:
        for key in all_keys:
            angle_series[key].append(angles.get(key, 0.0))

    # Calculate window means
    offsets = {}
    for key, values in angle_series.items():
        window_means = []

        for start in range(0, len(values) - window_size + 1, window_size):
            window = values[start:start + window_size]
            # Filter out zeros
            valid = [v for v in window if v != 0.0]
            if valid:
                window_means.append(np.mean(valid))

        # Average windows (skip first N)
        offsets[key] = moyenne_sans_zero(window_means, skip_n=skip_windows)

    return offsets


def apply_calibration(angle: float, offset: float) -> float:
    """
    Apply calibration offset to an angle.

    Args:
        angle: Raw measured angle
        offset: Calibration offset (neutral position angle)

    Returns:
        Calibrated angle (effective deviation from neutral)
    """
    if angle == 0.0:
        return 0.0
    return abs(angle - offset)


def project_keypoint_to_3d(
    x: float, y: float,
    depth_frame,
    intrinsics
) -> Optional[np.ndarray]:
    """
    Project a 2D keypoint to 3D using depth data.

    Uses RealSense rs2_deproject_pixel_to_point to convert
    pixel coordinates + depth to 3D point in meters.

    Args:
        x: Pixel x coordinate
        y: Pixel y coordinate
        depth_frame: RealSense depth frame
        intrinsics: Camera intrinsics

    Returns:
        3D point [x, y, z] in meters or None if invalid
    """
    import pyrealsense2 as rs

    # Ensure coordinates are within frame bounds
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    px = int(round(x))
    py = int(round(y))

    if px < 0 or px >= width or py < 0 or py >= height:
        return None

    # Get depth value at pixel
    depth = depth_frame.get_distance(px, py)

    if depth <= 0 or depth > 10.0:  # Invalid or too far
        return None

    # Deproject pixel to 3D point
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)

    return np.array(point_3d)


def calculate_angles_from_keypoints_3d(
    keypoints: np.ndarray,
    depth_frame,
    intrinsics
) -> Dict[str, float]:
    """
    Calculate all body segment angles from OpenPose keypoints using depth.

    Projects 2D keypoints to 3D using depth data, then calculates
    angles between body segments in true 3D space.

    Args:
        keypoints: OpenPose keypoints array (25x3 with x, y, confidence)
        depth_frame: RealSense depth frame
        intrinsics: Camera intrinsics

    Returns:
        Dictionary of angles (same keys as calculate_angles_from_keypoints_2d)
    """
    angles = {}

    def get_3d(name: str) -> Optional[np.ndarray]:
        """Get 3D point for a keypoint using depth projection."""
        if name not in KEYPOINT_INDICES:
            return None
        idx = KEYPOINT_INDICES[name]
        if idx >= len(keypoints):
            return None
        kp = keypoints[idx]
        # Check if keypoint is valid (x, y, confidence)
        if len(kp) >= 2 and kp[0] != 0 and kp[1] != 0:
            return project_keypoint_to_3d(kp[0], kp[1], depth_frame, intrinsics)
        return None

    # Extract 3D keypoints
    nose = get_3d("Nose")
    neck = get_3d("Neck")
    mid_hip = get_3d("MidHip")

    r_shoulder = get_3d("RShoulder")
    r_elbow = get_3d("RElbow")
    r_wrist = get_3d("RWrist")
    r_hip = get_3d("RHip")
    r_knee = get_3d("RKnee")
    r_ankle = get_3d("RAnkle")

    l_shoulder = get_3d("LShoulder")
    l_elbow = get_3d("LElbow")
    l_wrist = get_3d("LWrist")
    l_hip = get_3d("LHip")
    l_knee = get_3d("LKnee")
    l_ankle = get_3d("LAnkle")

    # Calculate angles (same as 2D version but with real 3D coordinates)
    if nose is not None and neck is not None and mid_hip is not None:
        angles["cou"] = calculate_angle(nose, neck, mid_hip)

    if mid_hip is not None and r_shoulder is not None and r_elbow is not None:
        angles["epaule_droite"] = calculate_angle(mid_hip, r_shoulder, r_elbow)

    if mid_hip is not None and l_shoulder is not None and l_elbow is not None:
        angles["epaule_gauche"] = calculate_angle(mid_hip, l_shoulder, l_elbow)

    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        angles["coude_droit"] = calculate_angle(r_shoulder, r_elbow, r_wrist)

    if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
        angles["coude_gauche"] = calculate_angle(l_shoulder, l_elbow, l_wrist)

    if r_hip is not None and r_knee is not None and r_ankle is not None:
        angles["genou_droit"] = calculate_angle(r_hip, r_knee, r_ankle)

    if l_hip is not None and l_knee is not None and l_ankle is not None:
        angles["genou_gauche"] = calculate_angle(l_hip, l_knee, l_ankle)

    if neck is not None and mid_hip is not None and l_hip is not None:
        angles["hanche"] = calculate_angle(neck, mid_hip, l_hip)

    return angles


def calculate_angles_both_2d_3d(
    keypoints: np.ndarray,
    depth_frame=None,
    intrinsics=None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate both 2D and 3D angles for comparison.

    Args:
        keypoints: OpenPose keypoints array
        depth_frame: RealSense depth frame (optional)
        intrinsics: Camera intrinsics (optional)

    Returns:
        Dictionary with "2d" and "3d" keys, each containing angle dict
    """
    result = {
        "2d": calculate_angles_from_keypoints_2d(keypoints),
        "3d": {}
    }

    if depth_frame is not None and intrinsics is not None:
        result["3d"] = calculate_angles_from_keypoints_3d(
            keypoints, depth_frame, intrinsics
        )

    return result
