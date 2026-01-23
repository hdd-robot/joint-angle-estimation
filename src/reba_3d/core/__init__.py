"""Core modules for geometry, smoothing, and keypoint processing."""

from reba_3d.core.geometry import orthonormaliser_repere, extraire_angles_nautiques
from reba_3d.core.smoothing import local_polynomial_regression
from reba_3d.core.keypoints import load_keypoints_3d, filter_by_confidence, build_dataframe
from reba_3d.core.frame_classifier import classify_frames, check_keypoints

__all__ = [
    "orthonormaliser_repere",
    "extraire_angles_nautiques",
    "local_polynomial_regression",
    "load_keypoints_3d",
    "filter_by_confidence",
    "build_dataframe",
    "classify_frames",
    "check_keypoints",
]
