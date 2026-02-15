"""
Simple angle calculations from OpenPose keypoints.

Calculates joint angles using vector dot products, similar to the original code.
Used for calibration and real-time REBA assessment.
"""

import numpy as np
from typing import Dict, List, Optional, Any

# OpenPose BODY_25 keypoint indices
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


def calculate_angle_to_vertical_2d(vector: np.ndarray, image_y_down: bool = True) -> float:
    """
    Calculate angle of a 2D vector relative to vertical axis.

    In image coordinates, Y typically points downward. This function calculates
    the angle between a vector and the vertical (Y) axis.

    Args:
        vector: 2D vector [x, y] (z component ignored if present)
        image_y_down: If True, Y axis points down (image coordinates).
                      If False, Y axis points up (standard math coordinates).

    Returns:
        Signed angle in degrees:
        - 0° = vertical (aligned with Y axis)
        - Positive = tilted to the right (in image coords) or forward (profile view)
        - Negative = tilted to the left (in image coords) or backward (profile view)
    """
    # Extract 2D components
    vx = vector[0]
    vy = vector[1]

    norm = np.sqrt(vx**2 + vy**2)
    if norm < 1e-8:
        return 0.0

    # Vertical reference: pointing down in image coordinates (positive Y)
    # For a person standing upright, neck-to-hip vector points downward (+Y)
    if image_y_down:
        # Angle from vertical (Y-down): use atan2(x, y)
        # 0° when vector points straight down (+Y)
        # Positive angle when tilted right/forward (+X)
        angle_rad = np.arctan2(vx, vy)
    else:
        # Standard math coordinates (Y-up)
        angle_rad = np.arctan2(vx, -vy)

    return float(np.degrees(angle_rad))


def detect_view_from_keypoints_2d(
    positions: Dict[str, np.ndarray],
    threshold_ratio: float = 0.4
) -> str:
    """
    Detect view orientation (frontal vs profile) from 2D keypoint positions.

    Analyzes the ratio of shoulder width to torso height to determine
    if the camera view is frontal or profile.

    Args:
        positions: Dictionary mapping keypoint names to 2D/3D coordinates
        threshold_ratio: Shoulder width / torso height ratio below which
                        the view is considered profile. Default: 0.4

    Returns:
        'frontal': Person facing camera (shoulders appear wide)
        'profile': Person sideways to camera (shoulders appear narrow)

    Note:
        In profile view, flexion angles (trunk, shoulder) are more accurate.
        In frontal view, lateral angles are more visible.
    """
    required = ["RShoulder", "LShoulder", "Neck", "MidHip"]

    # Check if all required keypoints exist and are valid
    for kp in required:
        if kp not in positions or positions[kp] is None:
            return 'frontal'  # Default to frontal
        if np.isnan(positions[kp]).any():
            return 'frontal'

    # Calculate shoulder width (horizontal span in X)
    shoulder_vec = positions["RShoulder"] - positions["LShoulder"]
    shoulder_width = abs(shoulder_vec[0])

    # Calculate torso height (vertical span in Y)
    torso_vec = positions["Neck"] - positions["MidHip"]
    torso_height = abs(torso_vec[1])

    # Avoid division by zero
    if torso_height < 1e-6:
        return 'frontal'

    # Calculate ratio
    ratio = shoulder_width / torso_height

    # If shoulders appear narrow relative to height → profile view
    if ratio < threshold_ratio:
        return 'profile'
    else:
        return 'frontal'


def calculate_shoulder_angle_profile(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    neck: np.ndarray,
    mid_hip: np.ndarray
) -> float:
    """
    Calculate shoulder flexion/extension angle in profile view.

    Measures the angle between the upper arm and the trunk axis.
    This is the relevant angle for REBA upper arm scoring.

    Args:
        shoulder: Shoulder keypoint position
        elbow: Elbow keypoint position
        neck: Neck keypoint position
        mid_hip: MidHip keypoint position

    Returns:
        Angle in degrees:
        - 0° = arm hanging straight down alongside trunk
        - Positive = arm raised forward (flexion)
        - Negative = arm behind (extension)
    """
    # Trunk axis (pointing upward from hip to neck)
    trunk_vec = neck - mid_hip

    # Upper arm vector (from shoulder to elbow)
    arm_vec = elbow - shoulder

    # Project to 2D (X, Y plane for profile view)
    trunk_2d = np.array([trunk_vec[0], trunk_vec[1]])
    arm_2d = np.array([arm_vec[0], arm_vec[1]])

    # Calculate angle of arm relative to trunk
    # First, get angle of trunk from vertical
    trunk_angle = calculate_angle_to_vertical_2d(trunk_2d)

    # Get angle of arm from vertical
    arm_angle = calculate_angle_to_vertical_2d(arm_2d)

    # Relative angle: arm angle minus trunk angle
    # This gives flexion relative to the trunk, not absolute
    relative_angle = arm_angle - trunk_angle

    # Normalize to -180 to 180
    while relative_angle > 180:
        relative_angle -= 360
    while relative_angle < -180:
        relative_angle += 360

    return abs(relative_angle)


def calculate_shoulder_angle_frontal(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    neck: np.ndarray,
    mid_hip: np.ndarray
) -> float:
    """
    Calculate shoulder abduction angle in frontal view.

    Measures how far the arm is raised sideways from the body.
    Less relevant for REBA flexion but useful for lateral assessment.

    Args:
        shoulder: Shoulder keypoint position
        elbow: Elbow keypoint position
        neck: Neck keypoint position
        mid_hip: MidHip keypoint position

    Returns:
        Angle in degrees (0° = arm at side, 90° = arm horizontal)
    """
    # Trunk axis (vertical reference)
    trunk_vec = neck - mid_hip

    # Upper arm vector
    arm_vec = elbow - shoulder

    # Use standard 3-point angle calculation
    # This measures abduction in the frontal plane
    return calculate_angle(neck, shoulder, elbow)


def calculate_angles_from_keypoints(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Calculate all body segment angles from OpenPose keypoints.

    Based on original code: CALCUL ANGLE GENERAL 3D.py

    Args:
        keypoints: OpenPose keypoints array

    Returns:
        Dictionary of angles:
        - neck: Neck angle (Nose-Neck-MidHip)
        - right_shoulder: Right shoulder (MidHip-RShoulder-RElbow)
        - left_shoulder: Left shoulder (MidHip-LShoulder-LElbow)
        - right_elbow: Right elbow (RShoulder-RElbow-RWrist)
        - left_elbow: Left elbow (LShoulder-LElbow-LWrist)
        - right_knee: Right knee (RHip-RKnee-RAnkle)
        - left_knee: Left knee (LHip-LKnee-LAnkle)
        - hip: Hip angle (Neck-MidHip-LHip)
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
        angles["neck"] = calculate_angle(nose, neck, mid_hip)

    # ÉPAULE DROITE: angle MidHip-RShoulder-RElbow
    if mid_hip is not None and r_shoulder is not None and r_elbow is not None:
        angles["right_shoulder"] = calculate_angle(mid_hip, r_shoulder, r_elbow)

    # ÉPAULE GAUCHE: angle MidHip-LShoulder-LElbow
    if mid_hip is not None and l_shoulder is not None and l_elbow is not None:
        angles["left_shoulder"] = calculate_angle(mid_hip, l_shoulder, l_elbow)

    # COUDE DROIT: angle RShoulder-RElbow-RWrist
    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        angles["right_elbow"] = calculate_angle(r_shoulder, r_elbow, r_wrist)

    # COUDE GAUCHE: angle LShoulder-LElbow-LWrist
    if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
        angles["left_elbow"] = calculate_angle(l_shoulder, l_elbow, l_wrist)

    # GENOU DROIT: angle RHip-RKnee-RAnkle
    if r_hip is not None and r_knee is not None and r_ankle is not None:
        angles["right_knee"] = calculate_angle(r_hip, r_knee, r_ankle)

    # GENOU GAUCHE: angle LHip-LKnee-LAnkle
    if l_hip is not None and l_knee is not None and l_ankle is not None:
        angles["left_knee"] = calculate_angle(l_hip, l_knee, l_ankle)

    # HANCHE: angle Neck-MidHip-LHip
    if neck is not None and mid_hip is not None and l_hip is not None:
        angles["hip"] = calculate_angle(neck, mid_hip, l_hip)

    return angles


def calculate_angles_from_keypoints_2d(keypoints: np.ndarray) -> Dict[str, Any]:
    """
    Calculate angles from 2D keypoints with automatic view detection.

    Detects whether the camera view is frontal or profile, and calculates
    angles appropriate for each view. Returns REBA-relevant angles.

    Args:
        keypoints: OpenPose keypoints array (25x3 with x, y, confidence)

    Returns:
        Dictionary containing:
        - view: Detected view ('frontal' or 'profile')
        - neck: Neck flexion angle
        - right_shoulder: Right shoulder angle (flexion in profile, abduction in frontal)
        - left_shoulder: Left shoulder angle
        - right_elbow: Right elbow flexion
        - left_elbow: Left elbow flexion
        - right_knee: Right knee flexion
        - left_knee: Left knee flexion
        - trunk_flexion: Trunk flexion relative to vertical (profile view)
                        or lateral bend (frontal view)
    """
    angles: Dict[str, Any] = {}

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

    # Build positions dict for view detection
    positions = {
        "Nose": nose, "Neck": neck, "MidHip": mid_hip,
        "RShoulder": r_shoulder, "RElbow": r_elbow, "RWrist": r_wrist,
        "RHip": r_hip, "RKnee": r_knee, "RAnkle": r_ankle,
        "LShoulder": l_shoulder, "LElbow": l_elbow, "LWrist": l_wrist,
        "LHip": l_hip, "LKnee": l_knee, "LAnkle": l_ankle,
    }

    # Detect view orientation
    view = detect_view_from_keypoints_2d(positions)
    angles["view"] = view

    # === COU (Neck): angle Nose-Neck-MidHip ===
    # Valid in both views - measures head tilt relative to trunk
    if nose is not None and neck is not None and mid_hip is not None:
        angles["neck"] = calculate_angle(nose, neck, mid_hip)

    # === TRUNK FLEXION (replaces old "hip") ===
    # Measures trunk angle relative to vertical - key for REBA
    if neck is not None and mid_hip is not None:
        trunk_vec = neck - mid_hip  # Vector from hip to neck
        trunk_angle = calculate_angle_to_vertical_2d(trunk_vec)
        angles["trunk_flexion"] = abs(trunk_angle)

        if view == 'profile':
            # In profile view, this is the actual forward/backward flexion
            angles["trunk_flexion_type"] = "sagittal"  # Forward/backward bend
        else:
            # In frontal view, this measures lateral bend
            angles["trunk_flexion_type"] = "lateral"  # Side bend

    # === ÉPAULES (Shoulders) ===
    # Calculation depends on view for accuracy
    if view == 'profile':
        # Profile view: calculate flexion/extension (forward/backward arm movement)
        if r_shoulder is not None and r_elbow is not None and neck is not None and mid_hip is not None:
            angles["right_shoulder"] = calculate_shoulder_angle_profile(
                r_shoulder, r_elbow, neck, mid_hip
            )
        if l_shoulder is not None and l_elbow is not None and neck is not None and mid_hip is not None:
            angles["left_shoulder"] = calculate_shoulder_angle_profile(
                l_shoulder, l_elbow, neck, mid_hip
            )
    else:
        # Frontal view: calculate abduction (sideways arm raise)
        if r_shoulder is not None and r_elbow is not None and neck is not None and mid_hip is not None:
            angles["right_shoulder"] = calculate_shoulder_angle_frontal(
                r_shoulder, r_elbow, neck, mid_hip
            )
        if l_shoulder is not None and l_elbow is not None and neck is not None and mid_hip is not None:
            angles["left_shoulder"] = calculate_shoulder_angle_frontal(
                l_shoulder, l_elbow, neck, mid_hip
            )

    # === COUDES (Elbows) ===
    # Valid in both views - measures elbow flexion angle
    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        angles["right_elbow"] = calculate_angle(r_shoulder, r_elbow, r_wrist)

    if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
        angles["left_elbow"] = calculate_angle(l_shoulder, l_elbow, l_wrist)

    # === GENOUX (Knees) ===
    # Better accuracy in profile view, but usable in both
    if r_hip is not None and r_knee is not None and r_ankle is not None:
        # Calculate knee flexion (180° = straight leg, <180° = bent)
        knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        # Convert to flexion from straight (0° = straight, positive = bent)
        angles["right_knee"] = 180.0 - knee_angle

    if l_hip is not None and l_knee is not None and l_ankle is not None:
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        angles["left_knee"] = 180.0 - knee_angle

    # === Legacy compatibility: keep "hip" as alias for trunk_flexion ===
    if "trunk_flexion" in angles:
        angles["hip"] = angles["trunk_flexion"]

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
        angles["neck"] = calculate_angle(nose, neck, mid_hip)

    if mid_hip is not None and r_shoulder is not None and r_elbow is not None:
        angles["right_shoulder"] = calculate_angle(mid_hip, r_shoulder, r_elbow)

    if mid_hip is not None and l_shoulder is not None and l_elbow is not None:
        angles["left_shoulder"] = calculate_angle(mid_hip, l_shoulder, l_elbow)

    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        angles["right_elbow"] = calculate_angle(r_shoulder, r_elbow, r_wrist)

    if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
        angles["left_elbow"] = calculate_angle(l_shoulder, l_elbow, l_wrist)

    if r_hip is not None and r_knee is not None and r_ankle is not None:
        angles["right_knee"] = calculate_angle(r_hip, r_knee, r_ankle)

    if l_hip is not None and l_knee is not None and l_ankle is not None:
        angles["left_knee"] = calculate_angle(l_hip, l_knee, l_ankle)

    if neck is not None and mid_hip is not None and l_hip is not None:
        angles["hip"] = calculate_angle(neck, mid_hip, l_hip)

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


def calculate_nautical_angles_3d(
    keypoints: np.ndarray,
    depth_frame,
    intrinsics
) -> Dict[str, Dict[str, float]]:
    """
    Calculate full 3D nautical angles (alpha, beta, gamma) from OpenPose keypoints using depth.

    This function uses the advanced nautical angle computation from reba.angles
    to provide complete 3D orientation information for each body segment.

    Args:
        keypoints: OpenPose keypoints array (25x3 with x, y, confidence)
        depth_frame: RealSense depth frame
        intrinsics: Camera intrinsics

    Returns:
        Nested dictionary structure:
        {
            "neck": {"alpha": float, "beta": float, "gamma": float},
            "torso": {"alpha": float, "beta": float, "gamma": float},
            "right_shoulder": {"alpha": float, "beta": float, "gamma": float, "elevation": float},
            "left_shoulder": {"alpha": float, "beta": float, "gamma": float, "elevation": float},
            "right_elbow": {"angle": float},
            "left_elbow": {"angle": float},
            "right_knee": {"angle": float},
            "left_knee": {"angle": float}
        }
    """
    # Lazy import to avoid circular dependency
    from reba_3d.reba.angles import (
        compute_neck_angles,
        compute_torso_angles,
        compute_shoulder_angles_right,
        compute_shoulder_angles_left,
        compute_shoulder_elevation_right,
        compute_shoulder_elevation_left,
        compute_elbow_angle_right,
        compute_elbow_angle_left,
        compute_knee_angle_right,
        compute_knee_angle_left,
    )

    def get_3d(name: str) -> Optional[np.ndarray]:
        """Get 3D point for a keypoint using depth projection."""
        if name not in KEYPOINT_INDICES:
            return None
        idx = KEYPOINT_INDICES[name]
        if idx >= len(keypoints):
            return None
        kp = keypoints[idx]
        if len(kp) >= 2 and kp[0] != 0 and kp[1] != 0:
            return project_keypoint_to_3d(kp[0], kp[1], depth_frame, intrinsics)
        return None

    # Build positions dictionary compatible with reba.angles functions
    positions = {}
    keypoint_names = [
        "Nose", "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow",
        "RWrist", "LWrist", "MidHip", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "REye", "LEye"
    ]

    for name in keypoint_names:
        point_3d = get_3d(name)
        if point_3d is not None:
            positions[name] = point_3d
        else:
            positions[name] = np.array([np.nan, np.nan, np.nan])

    # Calculate nautical angles using reba.angles functions
    result = {}

    # Neck angles (alpha, beta, gamma)
    neck_angles = compute_neck_angles(positions)
    if neck_angles:
        result["neck"] = {
            "alpha": neck_angles[0],
            "beta": neck_angles[1],
            "gamma": neck_angles[2]
        }
    else:
        result["neck"] = {"alpha": np.nan, "beta": np.nan, "gamma": np.nan}

    # Torso angles (alpha, beta, gamma)
    torso_angles = compute_torso_angles(positions)
    if torso_angles:
        result["torso"] = {
            "alpha": torso_angles[0],
            "beta": torso_angles[1],
            "gamma": torso_angles[2]
        }
    else:
        result["torso"] = {"alpha": np.nan, "beta": np.nan, "gamma": np.nan}

    # Right shoulder angles (alpha, beta, gamma) + elevation
    shoulder_r = compute_shoulder_angles_right(positions)
    elev_r = compute_shoulder_elevation_right(positions)
    if shoulder_r:
        result["right_shoulder"] = {
            "alpha": shoulder_r[0],
            "beta": shoulder_r[1],
            "gamma": shoulder_r[2],
            "elevation": elev_r if elev_r is not None else np.nan
        }
    else:
        result["right_shoulder"] = {
            "alpha": np.nan, "beta": np.nan, "gamma": np.nan, "elevation": np.nan
        }

    # Left shoulder angles (alpha, beta, gamma) + elevation
    shoulder_l = compute_shoulder_angles_left(positions)
    elev_l = compute_shoulder_elevation_left(positions)
    if shoulder_l:
        result["left_shoulder"] = {
            "alpha": shoulder_l[0],
            "beta": shoulder_l[1],
            "gamma": shoulder_l[2],
            "elevation": elev_l if elev_l is not None else np.nan
        }
    else:
        result["left_shoulder"] = {
            "alpha": np.nan, "beta": np.nan, "gamma": np.nan, "elevation": np.nan
        }

    # Elbows (single angle value)
    elbow_r = compute_elbow_angle_right(positions)
    result["right_elbow"] = {"angle": elbow_r if elbow_r is not None else np.nan}

    elbow_l = compute_elbow_angle_left(positions)
    result["left_elbow"] = {"angle": elbow_l if elbow_l is not None else np.nan}

    # Knees (single angle value)
    knee_r = compute_knee_angle_right(positions)
    result["right_knee"] = {"angle": knee_r if knee_r is not None else np.nan}

    knee_l = compute_knee_angle_left(positions)
    result["left_knee"] = {"angle": knee_l if knee_l is not None else np.nan}

    return result


def calculate_nautical_angles_2d(
    keypoints: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Calculate 2D nautical angles from OpenPose keypoints with automatic view detection.

    Harmonized version that:
    - Detects camera view (frontal vs profile) automatically
    - Uses REBA-relevant angle calculations based on view
    - In profile view: trunk_flexion = angle from vertical (sagittal plane)
    - In frontal view: trunk_flexion = lateral bend angle
    - Maintains nested dict format for compatibility with REBA scorer

    Args:
        keypoints: OpenPose keypoints array (25x3 with x, y, confidence)

    Returns:
        Nested dictionary structure compatible with calculate_reba_score_nautical:
        {
            "view": {"detected": "profile" or "frontal"},
            "neck": {"alpha": 0.0, "beta": 0.0, "gamma": float},
            "torso": {"alpha": 0.0, "beta": 0.0, "gamma": float},
            "right_shoulder": {"alpha": 0.0, "beta": 0.0, "gamma": float, "elevation": float},
            "left_shoulder": {"alpha": 0.0, "beta": 0.0, "gamma": float, "elevation": float},
            "right_elbow": {"angle": float},
            "left_elbow": {"angle": float},
            "right_knee": {"angle": float},
            "left_knee": {"angle": float},
        }

    Notes:
        - gamma values are now REBA-relevant angles (flexion from vertical or trunk)
        - In profile view, gamma represents forward/backward flexion
        - In frontal view, gamma represents lateral movement
        - Knee angles are converted to flexion (0° = straight, positive = bent)
    """
    # Lazy import to avoid circular dependency
    from reba_3d.reba.angles import (
        compute_shoulder_elevation_right,
        compute_shoulder_elevation_left,
    )

    def get_2d(name: str) -> Optional[np.ndarray]:
        """Get 2D point for a keypoint (x, y, z=0)."""
        if name not in KEYPOINT_INDICES:
            return None
        idx = KEYPOINT_INDICES[name]
        if idx >= len(keypoints):
            return None
        kp = keypoints[idx]
        if len(kp) >= 2 and kp[0] != 0 and kp[1] != 0:
            # Use x, y only, set z to 0 for 2D mode
            return np.array([kp[0], kp[1], 0.0])
        return None

    # Build positions dictionary
    positions: Dict[str, Optional[np.ndarray]] = {}
    keypoint_names = [
        "Nose", "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow",
        "RWrist", "LWrist", "MidHip", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "REye", "LEye"
    ]

    for name in keypoint_names:
        positions[name] = get_2d(name)

    # Build positions dict with NaN for missing (for elevation functions)
    positions_with_nan = {}
    for name in keypoint_names:
        if positions[name] is not None:
            positions_with_nan[name] = positions[name]
        else:
            positions_with_nan[name] = np.array([np.nan, np.nan, np.nan])

    # Detect view orientation using harmonized function
    view = detect_view_from_keypoints_2d(positions_with_nan)

    result: Dict[str, Dict[str, float]] = {}

    # Store view metadata
    result["view"] = {"detected": view}

    # === COU (Neck) ===
    # Angle Nose-Neck-MidHip (head tilt relative to trunk)
    nose = positions["Nose"]
    neck = positions["Neck"]
    mid_hip = positions["MidHip"]

    if nose is not None and neck is not None and mid_hip is not None:
        neck_angle = calculate_angle(nose, neck, mid_hip)
        # For REBA, we want deviation from neutral (~180° when aligned)
        # Convert to flexion angle (0° = neutral, positive = flexed)
        neck_flexion = abs(180.0 - neck_angle)
        result["neck"] = {
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": neck_flexion
        }
    else:
        result["neck"] = {"alpha": 0.0, "beta": 0.0, "gamma": np.nan}

    # === BUSTE (Trunk) ===
    # Use trunk_flexion: angle from vertical (REBA-relevant)
    if neck is not None and mid_hip is not None:
        trunk_vec = neck - mid_hip
        trunk_flexion = abs(calculate_angle_to_vertical_2d(trunk_vec))
        result["torso"] = {
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": trunk_flexion  # This is now the actual flexion from vertical
        }
    else:
        result["torso"] = {"alpha": 0.0, "beta": 0.0, "gamma": np.nan}

    # === ÉPAULES (Shoulders) ===
    r_shoulder = positions["RShoulder"]
    l_shoulder = positions["LShoulder"]
    r_elbow = positions["RElbow"]
    l_elbow = positions["LElbow"]

    # Get shoulder elevations (works in both views)
    elev_r = compute_shoulder_elevation_right(positions_with_nan)
    elev_l = compute_shoulder_elevation_left(positions_with_nan)

    if view == 'profile':
        # Profile view: use flexion/extension calculation
        if r_shoulder is not None and r_elbow is not None and neck is not None and mid_hip is not None:
            shoulder_angle_r = calculate_shoulder_angle_profile(
                r_shoulder, r_elbow, neck, mid_hip
            )
            result["right_shoulder"] = {
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": shoulder_angle_r,
                "elevation": elev_r if elev_r is not None else np.nan
            }
        else:
            result["right_shoulder"] = {
                "alpha": 0.0, "beta": 0.0, "gamma": np.nan, "elevation": np.nan
            }

        if l_shoulder is not None and l_elbow is not None and neck is not None and mid_hip is not None:
            shoulder_angle_l = calculate_shoulder_angle_profile(
                l_shoulder, l_elbow, neck, mid_hip
            )
            result["left_shoulder"] = {
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": shoulder_angle_l,
                "elevation": elev_l if elev_l is not None else np.nan
            }
        else:
            result["left_shoulder"] = {
                "alpha": 0.0, "beta": 0.0, "gamma": np.nan, "elevation": np.nan
            }
    else:
        # Frontal view: use abduction calculation
        if r_shoulder is not None and r_elbow is not None and neck is not None and mid_hip is not None:
            shoulder_angle_r = calculate_shoulder_angle_frontal(
                r_shoulder, r_elbow, neck, mid_hip
            )
            result["right_shoulder"] = {
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": shoulder_angle_r,
                "elevation": elev_r if elev_r is not None else np.nan
            }
        else:
            result["right_shoulder"] = {
                "alpha": 0.0, "beta": 0.0, "gamma": np.nan, "elevation": np.nan
            }

        if l_shoulder is not None and l_elbow is not None and neck is not None and mid_hip is not None:
            shoulder_angle_l = calculate_shoulder_angle_frontal(
                l_shoulder, l_elbow, neck, mid_hip
            )
            result["left_shoulder"] = {
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": shoulder_angle_l,
                "elevation": elev_l if elev_l is not None else np.nan
            }
        else:
            result["left_shoulder"] = {
                "alpha": 0.0, "beta": 0.0, "gamma": np.nan, "elevation": np.nan
            }

    # === COUDES (Elbows) ===
    # Elbow flexion works in both views
    r_wrist = positions["RWrist"]
    l_wrist = positions["LWrist"]

    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        elbow_angle_r = calculate_angle(r_shoulder, r_elbow, r_wrist)
        result["right_elbow"] = {"angle": elbow_angle_r}
    else:
        result["right_elbow"] = {"angle": np.nan}

    if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
        elbow_angle_l = calculate_angle(l_shoulder, l_elbow, l_wrist)
        result["left_elbow"] = {"angle": elbow_angle_l}
    else:
        result["left_elbow"] = {"angle": np.nan}

    # === GENOUX (Knees) ===
    # Convert to flexion angle (0° = straight leg, positive = bent)
    r_hip = positions["RHip"]
    l_hip = positions["LHip"]
    r_knee = positions["RKnee"]
    l_knee = positions["LKnee"]
    r_ankle = positions["RAnkle"]
    l_ankle = positions["LAnkle"]

    if r_hip is not None and r_knee is not None and r_ankle is not None:
        knee_raw = calculate_angle(r_hip, r_knee, r_ankle)
        # Convert: 180° (straight) -> 0° flexion, 90° -> 90° flexion
        knee_flexion_r = 180.0 - knee_raw
        result["right_knee"] = {"angle": knee_flexion_r}
    else:
        result["right_knee"] = {"angle": np.nan}

    if l_hip is not None and l_knee is not None and l_ankle is not None:
        knee_raw = calculate_angle(l_hip, l_knee, l_ankle)
        knee_flexion_l = 180.0 - knee_raw
        result["left_knee"] = {"angle": knee_flexion_l}
    else:
        result["left_knee"] = {"angle": np.nan}

    return result


def compute_calibration_offsets_nested(
    angles_list: List[Dict[str, Dict[str, float]]],
    window_size: int = 30,
    skip_windows: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Compute calibration offsets from a list of nested angle measurements.

    Groups angles into windows, calculates mean per window,
    then averages windows (skipping first ones for stability).

    Args:
        angles_list: List of angle dictionaries from each frame
        window_size: Number of frames per window (default: 30)
        skip_windows: Number of initial windows to skip (default: 1)

    Returns:
        Dictionary of offset values for each angle component:
        {
            "neck": {"alpha": float, "beta": float, "gamma": float},
            "torso": {"alpha": float, "beta": float, "gamma": float},
            ...
        }
    """
    if not angles_list:
        # Return default offsets if no data
        return {
            "neck": {"alpha": 180.0, "beta": 3.5, "gamma": 0.0},
            "torso": {"alpha": 90.0, "beta": 2.8, "gamma": 2.8},
            "right_shoulder": {"alpha": 0.0, "beta": 9.6, "gamma": 0.0, "elevation": 94.0},
            "left_shoulder": {"alpha": 21.0, "beta": -15.3, "gamma": 174.2, "elevation": 89.5},
            "right_elbow": {"angle": 170.5},
            "left_elbow": {"angle": 170.5},
            "right_knee": {"angle": 178.0},
            "left_knee": {"angle": 178.0},
        }

    # Segments to skip (metadata, not angle data)
    skip_segments = {"view"}

    # Collect all angle components by segment
    segments = {}
    for angles in angles_list:
        for segment, angle_dict in angles.items():
            # Skip metadata segments that don't contain numeric angles
            if segment in skip_segments:
                continue
            if segment not in segments:
                segments[segment] = {k: [] for k in angle_dict.keys()}
            for angle_name, value in angle_dict.items():
                segments[segment][angle_name].append(value)

    # Calculate window means for each component
    offsets = {}
    for segment, angle_series in segments.items():
        offsets[segment] = {}
        for angle_name, values in angle_series.items():
            window_means = []

            # Group into windows
            for start in range(0, len(values) - window_size + 1, window_size):
                window = values[start:start + window_size]
                # Filter out NaN values
                valid = [v for v in window if not np.isnan(v)]
                if valid:
                    window_means.append(np.mean(valid))

            # Average windows (skip first N for stability)
            if len(window_means) > skip_windows:
                offsets[segment][angle_name] = float(np.mean(window_means[skip_windows:]))
            elif window_means:
                offsets[segment][angle_name] = float(np.mean(window_means))
            else:
                offsets[segment][angle_name] = 0.0

    return offsets


def convert_calibration_data_to_robust_format(
    frames_list: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Convert GUI calibration data to robust calibration format.

    Transposes the temporal sequence of angle measurements into the format
    expected by robust calibration functions (MAD filtering).

    Args:
        frames_list: List of angle dictionaries from each frame
            [
                {"neck": {"alpha": 180.0, "beta": 3.5, ...}, "torso": {...}, ...},
                {"neck": {"alpha": 180.5, "beta": 3.2, ...}, "torso": {...}, ...},
                ...
            ]

    Returns:
        Dictionary organized by segment and angle with time series
            {
                "neck": {"alpha": [180.0, 180.5, ...], "beta": [3.5, 3.2, ...], ...},
                "torso": {...},
                ...
            }

    Raises:
        ValueError: If frames_list is empty or has inconsistent structure
    """
    if not frames_list:
        raise ValueError("Cannot convert empty calibration data")

    # Initialize result structure
    result = {}

    # Segments to skip (metadata, not angle data)
    skip_segments = {"view"}

    # Process each frame
    for frame_idx, frame_angles in enumerate(frames_list):
        if not isinstance(frame_angles, dict):
            raise ValueError(f"Frame {frame_idx} is not a dictionary")

        for segment, angle_dict in frame_angles.items():
            # Skip metadata segments that don't contain numeric angles
            if segment in skip_segments:
                continue

            if not isinstance(angle_dict, dict):
                raise ValueError(
                    f"Frame {frame_idx}, segment '{segment}' angles not a dict"
                )

            # Initialize segment if first encounter
            if segment not in result:
                result[segment] = {angle_name: [] for angle_name in angle_dict.keys()}

            # Append each angle value
            for angle_name, value in angle_dict.items():
                if angle_name not in result[segment]:
                    # Handle case where later frames have new angles
                    result[segment][angle_name] = [np.nan] * frame_idx
                result[segment][angle_name].append(value)

    # Pad any missing values at the end
    num_frames = len(frames_list)
    for segment, angles in result.items():
        for angle_name, values in angles.items():
            if len(values) < num_frames:
                values.extend([np.nan] * (num_frames - len(values)))

    return result


def compute_calibration_offsets_robust(
    angles_list: List[Dict[str, Dict[str, float]]],
    n_neutre: int = 60,
    k_mad: float = 3.5
) -> Dict[str, Dict[str, float]]:
    """
    Compute calibration offsets using robust MAD-based filtering.

    This function wraps the robust calibration system and provides a drop-in
    replacement for compute_calibration_offsets_nested() with improved
    outlier resistance using Median Absolute Deviation (MAD) filtering.

    The MAD method is more robust to outliers and uses circular statistics
    for proper handling of angular measurements near ±180°.

    Args:
        angles_list: List of angle dictionaries from each frame (from GUI)
        n_neutre: Number of neutral frames to use for offset calculation (default: 60)
        k_mad: MAD threshold multiplier for outlier filtering (default: 3.5)
               Higher values = more permissive (keep more data)
               Lower values = more strict (reject more outliers)

    Returns:
        Dictionary of offset values for each angle component:
        {
            "neck": {"alpha": float, "beta": float, "gamma": float},
            "torso": {"alpha": float, "beta": float, "gamma": float},
            ...
        }

    Raises:
        ValueError: If angles_list is empty or has insufficient frames

    Example:
        >>> frames = [{"neck": {"alpha": 180.0}, ...}, ...]
        >>> offsets = compute_calibration_offsets_robust(frames, n_neutre=60)
        >>> print(offsets["neck"]["alpha"])
        180.0
    """
    from reba_3d.core.robust_calibration import calibrate_all_segments

    if not angles_list:
        raise ValueError("Cannot compute offsets from empty angle list")

    if len(angles_list) < n_neutre:
        raise ValueError(
            f"Insufficient frames for calibration: {len(angles_list)} < {n_neutre}. "
            f"Need at least {n_neutre} frames."
        )

    # Convert format from GUI (frames → segments) to robust format (segments → frames)
    try:
        robust_format = convert_calibration_data_to_robust_format(angles_list)
    except ValueError as e:
        raise ValueError(f"Failed to convert calibration data: {e}")

    # Apply robust calibration with MAD filtering
    # Note: calibrate_all_segments returns (offsets, calibrated_angles)
    # We only need the offsets for saving
    offsets, _ = calibrate_all_segments(
        robust_format,
        n_neutral=n_neutre,
        config_segments=None  # Use default configuration
    )

    return offsets
