"""
Robust calibration functions using MAD (Median Absolute Deviation).

Provides statistical methods for computing calibration offsets that are
resistant to outliers, using circular statistics for angular measurements.
"""

import numpy as np
from typing import Dict, List, Set


def normalize_angle(a: float) -> float:
    """
    Normalize angle to [-180, 180) range.

    Args:
        a: Angle in degrees

    Returns:
        Normalized angle in [-180, 180)
    """
    return (a + 180) % 360 - 180


def _filter_valid_values(seq: List[float]) -> np.ndarray:
    """
    Filter out invalid values (NaN and zeros).

    Args:
        seq: Sequence of values to filter

    Returns:
        Array of valid values
    """
    arr = np.asarray(seq, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr != 0.0]
    return arr


def circular_mean_degrees(arr_deg: np.ndarray) -> float:
    """
    Compute circular mean for angles in degrees.

    Uses vector averaging to correctly handle angle wraparound.

    Args:
        arr_deg: Array of angles in degrees

    Returns:
        Circular mean in degrees
    """
    ang = np.deg2rad(arr_deg)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))))


def robust_linear_offset(seq: List[float], k_mad: float = 3.5, use_mean: bool = True) -> float:
    """
    Compute robust offset for non-circular measurements using MAD filtering.

    Uses Median Absolute Deviation (MAD) to identify and remove outliers
    before computing the final offset value.

    Args:
        seq: Sequence of angle measurements
        k_mad: Number of MAD units for outlier threshold (default: 3.5)
        use_mean: Use mean (True) or median (False) after filtering

    Returns:
        Robust offset value
    """
    arr = _filter_valid_values(seq)
    if arr.size == 0:
        return 0.0

    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    if mad == 0:
        return float(med)

    # Convert MAD to standard deviation estimate
    sigma = 1.4826 * mad

    # Keep only values within k_mad standard deviations
    keep = np.abs(arr - med) <= k_mad * sigma
    arr2 = arr[keep]

    if arr2.size == 0:
        return float(med)

    return float(np.mean(arr2) if use_mean else np.median(arr2))


def robust_circular_offset(seq: List[float], k_mad: float = 3.5) -> float:
    """
    Compute robust offset for circular (angular) measurements using MAD filtering.

    Uses circular statistics and MAD to handle angle wraparound correctly
    while filtering outliers.

    Args:
        seq: Sequence of angle measurements in degrees
        k_mad: Number of MAD units for outlier threshold (default: 3.5)

    Returns:
        Robust circular offset in degrees
    """
    arr = _filter_valid_values(seq)
    if arr.size == 0:
        return 0.0

    # Compute circular mean
    mu = circular_mean_degrees(arr)

    # Compute angular distances normalized to [-180, 180)
    dist = np.array([normalize_angle(x - mu) for x in arr], dtype=float)

    med = np.median(dist)
    mad = np.median(np.abs(dist - med))

    if mad == 0:
        return float(mu)

    sigma = 1.4826 * mad
    keep = np.abs(dist - med) <= k_mad * sigma
    arr2 = arr[keep]

    if arr2.size == 0:
        return float(mu)

    return float(circular_mean_degrees(arr2))


def calculate_offsets_from_neutral(
    angles_dict: Dict[str, List[float]],
    N: int,
    circular_keys: Set[str] = None
) -> Dict[str, float]:
    """
    Calculate calibration offsets from neutral pose frames.

    Uses robust statistical methods (MAD filtering) to compute reliable
    offsets from the first N frames of neutral pose data.

    Args:
        angles_dict: Dictionary of angle sequences {"alpha": [...], "beta": [...]}
        N: Number of neutral frames to use for calibration
        circular_keys: Set of keys to treat as circular angles (default: empty set)

    Returns:
        Dictionary of computed offsets
    """
    if circular_keys is None:
        circular_keys = set()

    offsets = {}
    for k, seq in angles_dict.items():
        window = seq[:N]
        if k in circular_keys:
            offsets[k] = robust_circular_offset(window)
        else:
            offsets[k] = robust_linear_offset(window)

    return offsets


def apply_calibration(
    seq: List[float],
    offset: float,
    circular: bool = False,
    absval: bool = False,
    invert: bool = False
) -> List[float]:
    """
    Apply calibration offset to a sequence of angles.

    Args:
        seq: Sequence of raw angle measurements
        offset: Calibration offset to apply
        circular: Whether to normalize result to [-180, 180) (for circular angles)
        absval: Whether to take absolute value of result
        invert: Whether to invert subtraction (offset - angle instead of angle - offset)

    Returns:
        List of calibrated angles
    """
    out = []
    for a in seq:
        if a == 0.0 or np.isnan(a):
            out.append(0.0)
            continue

        val = (offset - a) if invert else (a - offset)

        if circular:
            val = normalize_angle(val)

        if absval:
            val = abs(val)

        out.append(val)

    return out


def calibrate_all_segments(
    dynamic_angles: Dict[str, Dict[str, List[float]]],
    n_neutral: int = 1,
    config_segments: Dict[str, Dict] = None
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, List[float]]]]:
    """
    Apply robust calibration to all body segments.

    This is a high-level function that:
    1. Computes robust offsets from neutral frames
    2. Applies calibration to all angle sequences

    Args:
        dynamic_angles: Dictionary of angle sequences by segment
            {
                "neck": {"alpha": [...], "beta": [...], "gamma": [...]},
                "torso": {...},
                "right_shoulder": {...},
                ...
            }
        n_neutral: Number of neutral frames to use (default: 30)
        config_segments: Optional configuration for each segment
            {
                "neck": {
                    "circulars": {"alpha", "beta", "gamma"},
                    "absval": False,
                    "invert": False
                },
                ...
            }

    Returns:
        Tuple of (offsets_dict, calibrated_angles_dict)
    """
    if config_segments is None:
        # Default configuration
        config_segments = {
            "neck": {
                "circulars": {"alpha", "beta", "gamma"},
                "absval": False,
                "invert": False
            },
            "torso": {
                "circulars": {"alpha", "beta", "gamma"},
                "absval": False,
                "invert": False
            },
            "right_shoulder": {
                "circulars": {"alpha", "beta", "gamma"},
                "absval": False,
                "invert": True
            },
            "left_shoulder": {
                "circulars": {"alpha", "beta", "gamma"},
                "absval": False,
                "invert": True
            },
            "right_elbow": {
                "circulars": set(),
                "absval": True,
                "invert": False
            },
            "left_elbow": {
                "circulars": set(),
                "absval": True,
                "invert": False
            },
            "right_knee": {
                "circulars": set(),
                "absval": True,
                "invert": False
            },
            "left_knee": {
                "circulars": set(),
                "absval": True,
                "invert": False
            },
        }

    all_offsets = {}
    calibrated_angles = {}

    for segment, angles_dict in dynamic_angles.items():
        config = config_segments.get(segment, {
            "circulars": set(),
            "absval": False,
            "invert": False
        })

        # Compute offsets for this segment
        offsets_segment = calculate_offsets_from_neutral(
            angles_dict,
            n_neutral,
            circular_keys=config["circulars"]
        )
        all_offsets[segment] = offsets_segment

        # Apply calibration to this segment
        calibrated_angles[segment] = {}
        for k, seq in angles_dict.items():
            # Special handling for elevation angles (not circular)
            is_circular = k in config["circulars"] and "elevation" not in k

            calibrated_angles[segment][k] = apply_calibration(
                seq,
                offsets_segment[k],
                circular=is_circular,
                absval=config["absval"],
                invert=config["invert"]
            )

    return all_offsets, calibrated_angles
