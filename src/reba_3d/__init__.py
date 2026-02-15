"""
reba_3d - 3D Ergonomic Posture Analysis System using REBA methodology.

This package provides tools for analyzing worker posture risk levels using
RGB-D video from Intel RealSense cameras with OpenPose skeleton detection.
"""

__version__ = "1.0.0"
__author__ = "REBA 3D Team"

from reba_3d.reba.risk_assessment import REBAAssessor
from reba_3d.core.keypoints import load_keypoints_3d
from reba_3d.io.json_io import read_keypoints_json, write_risk_times_json
from reba_3d.utils.logger import get_logger, setup_logging, LogLevel
from reba_3d.reba.calibration import (
    compute_offsets_from_neutral_robust,
    calibrate_all_angles_robust,
)
from reba_3d.core.robust_calibration import (
    normalize_angle,
    circular_mean_degrees,
    robust_linear_offset,
    robust_circular_offset,
    calculate_offsets_from_neutral,
    apply_calibration,
    calibrate_all_segments,
)

__all__ = [
    "REBAAssessor",
    "load_keypoints_3d",
    "read_keypoints_json",
    "write_risk_times_json",
    "get_logger",
    "setup_logging",
    "LogLevel",
    # Robust calibration functions
    "compute_offsets_from_neutral_robust",
    "calibrate_all_angles_robust",
    "normalize_angle",
    "circular_mean_degrees",
    "robust_linear_offset",
    "robust_circular_offset",
    "calculate_offsets_from_neutral",
    "apply_calibration",
    "calibrate_all_segments",
]
