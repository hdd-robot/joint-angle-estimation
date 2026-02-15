"""Core modules for geometry, smoothing, and keypoint processing."""

from reba_3d.core.geometry import orthonormalize_frame, extract_nautical_angles
from reba_3d.core.smoothing import local_polynomial_regression
from reba_3d.core.keypoints import load_keypoints_3d, filter_by_confidence, build_dataframe
from reba_3d.core.frame_classifier import classify_frames, check_keypoints
from reba_3d.core.angles import (
    calculate_angles_from_keypoints,
    calculate_angles_from_keypoints_2d,
    compute_calibration_offsets,
    apply_calibration,
)

__all__ = [
    "orthonormalize_frame",
    "extract_nautical_angles",
    "local_polynomial_regression",
    "load_keypoints_3d",
    "filter_by_confidence",
    "build_dataframe",
    "classify_frames",
    "check_keypoints",
    "calculate_angles_from_keypoints",
    "calculate_angles_from_keypoints_2d",
    "compute_calibration_offsets",
    "apply_calibration",
]
