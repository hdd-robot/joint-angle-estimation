"""
Calibration functions for REBA angle measurements.

Applies calibration offsets to convert raw measured angles to
effective angles relative to a neutral standing position.

Includes both legacy methods and new robust calibration using MAD filtering.
"""

import numpy as np
from typing import List, Dict, Set

from reba_3d.config.calibration_data import CALIBRATION_OFFSETS
from reba_3d.core.geometry import normalize_angle
from reba_3d.core.robust_calibration import (
    calculate_offsets_from_neutral,
    apply_calibration as apply_calibration_torso,
    calibrate_all_segments,
)


def apply_calibration_neck(angles: List[float], offset: float) -> List[float]:
    """
    Apply calibration to neck angles.

    Uses normalization to keep angles in [-180, 180) range.

    Args:
        angles: List of raw angle values
        offset: Calibration offset from neutral position

    Returns:
        List of calibrated angles
    """
    return [
        normalize_angle(angle - offset) if angle != 0.0 else 0.0
        for angle in angles
    ]


def apply_calibration(angles: List[float], offset: float) -> List[float]:
    """
    Apply calibration with absolute value.

    Standard calibration method for most body segments.

    Args:
        angles: List of raw angle values
        offset: Calibration offset from neutral position

    Returns:
        List of calibrated angles (absolute values)
    """
    return [
        abs(angle - offset) if angle != 0.0 else 0.0
        for angle in angles
    ]


def apply_calibration_torso(angles: List[float], offset: float) -> List[float]:
    """
    Apply calibration to torso angles.

    Does not use absolute value to preserve direction information.

    Args:
        angles: List of raw angle values
        offset: Calibration offset from neutral position

    Returns:
        List of calibrated angles
    """
    return [
        angle - offset if angle != 0.0 else 0.0
        for angle in angles
    ]


def apply_calibration_epaule(angles: List[float], offset: float) -> List[float]:
    """
    Apply calibration to shoulder angles.

    Uses inverted subtraction (offset - angle).

    Args:
        angles: List of raw angle values
        offset: Calibration offset from neutral position

    Returns:
        List of calibrated angles
    """
    return [
        offset - angle if angle != 0.0 else 0.0
        for angle in angles
    ]


def apply_calibration_genou(angles: List[float], offset: float) -> List[float]:
    """
    Apply calibration to knee angles.

    Adds 180 degrees to account for knee angle convention.

    Args:
        angles: List of raw angle values
        offset: Calibration offset from neutral position

    Returns:
        List of calibrated angles
    """
    return [
        abs(180 + (angle - offset)) if angle != 0.0 else 0.0
        for angle in angles
    ]


def calibrate_all_angles(raw_angles: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Apply calibration to all body segment angles.

    Args:
        raw_angles: Dictionary of raw angles organized by body segment
            {
                "neck": {"alpha": [...], "beta": [...], "gamma": [...]},
                "torso": {"alpha": [...], "beta": [...], "gamma": [...]},
                "right_shoulder": {"alpha": [...], "beta": [...], "gamma": [...], "elevation": [...]},
                "right_elbow": {"angle": [...]},
                "right_knee": {"angle": [...]},
                "left_shoulder": {"alpha": [...], "beta": [...], "gamma": [...], "elevation": [...]},
                "left_elbow": {"angle": [...]},
                "left_knee": {"angle": [...]}
            }

    Returns:
        Dictionary of calibrated angles with same structure
    """
    offsets = CALIBRATION_OFFSETS
    calibrated = {}

    # Cou (neck)
    if "neck" in raw_angles:
        calibrated["neck"] = {
            "alpha": apply_calibration_neck(
                raw_angles["neck"].get("alpha", []),
                offsets["neck"]["alpha"]
            ),
            "beta": apply_calibration_neck(
                raw_angles["neck"].get("beta", []),
                offsets["neck"]["beta"]
            ),
            "gamma": apply_calibration_neck(
                raw_angles["neck"].get("gamma", []),
                offsets["neck"]["gamma"]
            ),
        }

    # Buste (torso)
    if "torso" in raw_angles:
        calibrated["torso"] = {
            "alpha": apply_calibration_torso(
                raw_angles["torso"].get("alpha", []),
                offsets["torso"]["alpha"]
            ),
            "beta": apply_calibration_torso(
                raw_angles["torso"].get("beta", []),
                offsets["torso"]["beta"]
            ),
            "gamma": apply_calibration_torso(
                raw_angles["torso"].get("gamma", []),
                offsets["torso"]["gamma"]
            ),
        }

    # Épaule droite (right shoulder)
    if "right_shoulder" in raw_angles:
        calibrated["right_shoulder"] = {
            "alpha": apply_calibration_epaule(
                raw_angles["right_shoulder"].get("alpha", []),
                offsets["right_shoulder"]["alpha"]
            ),
            "beta": apply_calibration_epaule(
                raw_angles["right_shoulder"].get("beta", []),
                offsets["right_shoulder"]["beta"]
            ),
            "gamma": apply_calibration_epaule(
                raw_angles["right_shoulder"].get("gamma", []),
                offsets["right_shoulder"]["gamma"]
            ),
            "elevation": apply_calibration_epaule(
                raw_angles["right_shoulder"].get("elevation", []),
                offsets["right_shoulder"]["elevation"]
            ),
        }

    # Coude droit (right elbow)
    if "right_elbow" in raw_angles:
        calibrated["right_elbow"] = {
            "angle": apply_calibration(
                raw_angles["right_elbow"].get("angle", []),
                offsets["right_elbow"]["angle"]
            ),
        }

    # Genou droit (right knee)
    if "right_knee" in raw_angles:
        calibrated["right_knee"] = {
            "angle": apply_calibration(
                raw_angles["right_knee"].get("angle", []),
                offsets["right_knee"]["angle"]
            ),
        }

    # Épaule gauche (left shoulder)
    if "left_shoulder" in raw_angles:
        calibrated["left_shoulder"] = {
            "alpha": apply_calibration_epaule(
                raw_angles["left_shoulder"].get("alpha", []),
                offsets["left_shoulder"]["alpha"]
            ),
            "beta": apply_calibration_epaule(
                raw_angles["left_shoulder"].get("beta", []),
                offsets["left_shoulder"]["beta"]
            ),
            "gamma": apply_calibration_epaule(
                raw_angles["left_shoulder"].get("gamma", []),
                offsets["left_shoulder"]["gamma"]
            ),
            "elevation": apply_calibration_epaule(
                raw_angles["left_shoulder"].get("elevation", []),
                offsets["left_shoulder"]["elevation"]
            ),
        }

    # Coude gauche (left elbow)
    if "left_elbow" in raw_angles:
        calibrated["left_elbow"] = {
            "angle": apply_calibration(
                raw_angles["left_elbow"].get("angle", []),
                offsets["left_elbow"]["angle"]
            ),
        }

    # Genou gauche (left knee)
    if "left_knee" in raw_angles:
        calibrated["left_knee"] = {
            "angle": apply_calibration(
                raw_angles["left_knee"].get("angle", []),
                offsets["left_knee"]["angle"]
            ),
        }

    return calibrated


def compute_offsets_from_neutral(
    neutral_angles: Dict[str, Dict[str, List[float]]],
    skip_n: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Compute calibration offsets from a neutral pose recording.

    Args:
        neutral_angles: Dictionary of angles recorded in neutral standing pose
        skip_n: Number of initial measurements to skip (default: 1)

    Returns:
        Dictionary of offset values for each body segment
    """
    from reba_3d.config.calibration_data import moyenne_sans_zero

    offsets = {}

    for segment, angles_dict in neutral_angles.items():
        offsets[segment] = {}
        for angle_name, values in angles_dict.items():
            offsets[segment][angle_name] = moyenne_sans_zero(values, skip_n)

    return offsets


def compute_offsets_from_neutral_robust(
    neutral_angles: Dict[str, Dict[str, List[float]]],
    N_neutre: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Compute calibration offsets using robust MAD filtering.

    Uses Median Absolute Deviation (MAD) to identify and remove outliers
    before computing offset values. More resistant to noise and outliers
    than simple averaging.

    Args:
        neutral_angles: Dictionary of angles recorded in neutral standing pose
            {
                "neck": {"alpha": [...], "beta": [...], "gamma": [...]},
                "torso": {...},
                ...
            }
        N_neutre: Number of neutral frames to use (default: 30)

    Returns:
        Dictionary of offset values for each body segment
    """
    # Define which angles are circular (need wraparound handling)
    circular_angles = {
        "neck": {"alpha", "beta", "gamma"},
        "torso": {"alpha", "beta", "gamma"},
        "right_shoulder": {"alpha", "beta", "gamma"},
        "left_shoulder": {"alpha", "beta", "gamma"},
        "right_elbow": set(),
        "left_elbow": set(),
        "right_knee": set(),
        "left_knee": set(),
    }

    offsets = {}

    for segment, angles_dict in neutral_angles.items():
        circulaires = circular_angles.get(segment, set())
        offsets[segment] = calculate_offsets_from_neutral(
            angles_dict,
            N_neutre,
            keys_circulaires=circulaires
        )

    return offsets


def calibrate_all_angles_robust(
    raw_angles: Dict[str, Dict[str, List[float]]],
    N_neutre: int = 1
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, List[float]]]]:
    """
    Apply robust calibration to all body segment angles.

    This function:
    1. Computes robust offsets from the first N_neutre frames
    2. Applies calibration with appropriate settings for each segment

    Args:
        raw_angles: Dictionary of raw angles organized by body segment
            {
                "neck": {"alpha": [...], "beta": [...], "gamma": [...]},
                "torso": {"alpha": [...], "beta": [...], "gamma": [...]},
                "right_shoulder": {"alpha": [...], "beta": [...], "gamma": [...], "elevation": [...]},
                ...
            }
        N_neutre: Number of neutral frames to use for offset computation (default: 30)

    Returns:
        Tuple of (offsets_dict, calibrated_angles_dict)
        - offsets_dict: Computed offset values for each segment
        - calibrated_angles_dict: Calibrated angle sequences
    """
    # Configuration for each segment
    config_segments = {
        "neck": {
            "circulaires": {"alpha", "beta", "gamma"},
            "absval": False,
            "invert": False
        },
        "torso": {
            "circulaires": {"alpha", "beta", "gamma"},
            "absval": False,
            "invert": False
        },
        "right_shoulder": {
            "circulaires": {"alpha", "beta", "gamma"},
            "absval": False,
            "invert": True
        },
        "left_shoulder": {
            "circulaires": {"alpha", "beta", "gamma"},
            "absval": False,
            "invert": True
        },
        "right_elbow": {
            "circulaires": set(),
            "absval": True,
            "invert": False
        },
        "left_elbow": {
            "circulaires": set(),
            "absval": True,
            "invert": False
        },
        "right_knee": {
            "circulaires": set(),
            "absval": True,
            "invert": False
        },
        "left_knee": {
            "circulaires": set(),
            "absval": True,
            "invert": False
        },
    }

    return calibrate_all_segments(raw_angles, N_neutre, config_segments)
