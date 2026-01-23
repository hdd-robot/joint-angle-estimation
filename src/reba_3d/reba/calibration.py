"""
Calibration functions for REBA angle measurements.

Applies calibration offsets to convert raw measured angles to
effective angles relative to a neutral standing position.
"""

import numpy as np
from typing import List, Dict

from reba_3d.config.calibration_data import CALIBRATION_OFFSETS
from reba_3d.core.geometry import normaliser_angle


def appliquer_calibration_cou(angles: List[float], offset: float) -> List[float]:
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
        normaliser_angle(angle - offset) if angle != 0.0 else 0.0
        for angle in angles
    ]


def appliquer_calibration(angles: List[float], offset: float) -> List[float]:
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


def appliquer_calibration_buste(angles: List[float], offset: float) -> List[float]:
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


def appliquer_calibration_epaule(angles: List[float], offset: float) -> List[float]:
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


def appliquer_calibration_genou(angles: List[float], offset: float) -> List[float]:
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
                "cou": {"alpha": [...], "beta": [...], "gamma": [...]},
                "buste": {"alpha": [...], "beta": [...], "gamma": [...]},
                "epaule_droite": {"alpha": [...], "beta": [...], "gamma": [...], "elevation": [...]},
                "coude_droit": {"angle": [...]},
                "genou_droit": {"angle": [...]},
                "epaule_gauche": {"alpha": [...], "beta": [...], "gamma": [...], "elevation": [...]},
                "coude_gauche": {"angle": [...]},
                "genou_gauche": {"angle": [...]}
            }

    Returns:
        Dictionary of calibrated angles with same structure
    """
    offsets = CALIBRATION_OFFSETS
    calibrated = {}

    # Cou (neck)
    if "cou" in raw_angles:
        calibrated["cou"] = {
            "alpha": appliquer_calibration_cou(
                raw_angles["cou"].get("alpha", []),
                offsets["cou"]["alpha"]
            ),
            "beta": appliquer_calibration_cou(
                raw_angles["cou"].get("beta", []),
                offsets["cou"]["beta"]
            ),
            "gamma": appliquer_calibration_cou(
                raw_angles["cou"].get("gamma", []),
                offsets["cou"]["gamma"]
            ),
        }

    # Buste (torso)
    if "buste" in raw_angles:
        calibrated["buste"] = {
            "alpha": appliquer_calibration_buste(
                raw_angles["buste"].get("alpha", []),
                offsets["buste"]["alpha"]
            ),
            "beta": appliquer_calibration_buste(
                raw_angles["buste"].get("beta", []),
                offsets["buste"]["beta"]
            ),
            "gamma": appliquer_calibration_buste(
                raw_angles["buste"].get("gamma", []),
                offsets["buste"]["gamma"]
            ),
        }

    # Épaule droite (right shoulder)
    if "epaule_droite" in raw_angles:
        calibrated["epaule_droite"] = {
            "alpha": appliquer_calibration_epaule(
                raw_angles["epaule_droite"].get("alpha", []),
                offsets["epaule_droite"]["alpha"]
            ),
            "beta": appliquer_calibration_epaule(
                raw_angles["epaule_droite"].get("beta", []),
                offsets["epaule_droite"]["beta"]
            ),
            "gamma": appliquer_calibration_epaule(
                raw_angles["epaule_droite"].get("gamma", []),
                offsets["epaule_droite"]["gamma"]
            ),
            "elevation": appliquer_calibration_epaule(
                raw_angles["epaule_droite"].get("elevation", []),
                offsets["epaule_droite"]["elevation"]
            ),
        }

    # Coude droit (right elbow)
    if "coude_droit" in raw_angles:
        calibrated["coude_droit"] = {
            "angle": appliquer_calibration(
                raw_angles["coude_droit"].get("angle", []),
                offsets["coude_droit"]["angle"]
            ),
        }

    # Genou droit (right knee)
    if "genou_droit" in raw_angles:
        calibrated["genou_droit"] = {
            "angle": appliquer_calibration(
                raw_angles["genou_droit"].get("angle", []),
                offsets["genou_droit"]["angle"]
            ),
        }

    # Épaule gauche (left shoulder)
    if "epaule_gauche" in raw_angles:
        calibrated["epaule_gauche"] = {
            "alpha": appliquer_calibration_epaule(
                raw_angles["epaule_gauche"].get("alpha", []),
                offsets["epaule_gauche"]["alpha"]
            ),
            "beta": appliquer_calibration_epaule(
                raw_angles["epaule_gauche"].get("beta", []),
                offsets["epaule_gauche"]["beta"]
            ),
            "gamma": appliquer_calibration_epaule(
                raw_angles["epaule_gauche"].get("gamma", []),
                offsets["epaule_gauche"]["gamma"]
            ),
            "elevation": appliquer_calibration_epaule(
                raw_angles["epaule_gauche"].get("elevation", []),
                offsets["epaule_gauche"]["elevation"]
            ),
        }

    # Coude gauche (left elbow)
    if "coude_gauche" in raw_angles:
        calibrated["coude_gauche"] = {
            "angle": appliquer_calibration(
                raw_angles["coude_gauche"].get("angle", []),
                offsets["coude_gauche"]["angle"]
            ),
        }

    # Genou gauche (left knee)
    if "genou_gauche" in raw_angles:
        calibrated["genou_gauche"] = {
            "angle": appliquer_calibration(
                raw_angles["genou_gauche"].get("angle", []),
                offsets["genou_gauche"]["angle"]
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
