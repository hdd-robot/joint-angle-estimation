"""
Calibration data storage and retrieval.

Handles saving and loading calibration offsets to/from YAML files.
"""

import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import yaml


# Default calibration offsets (neutral standing position) - Simple angles
DEFAULT_OFFSETS = {
    "neck": 154.0,           # Neck angle when standing straight
    "right_shoulder": 33.0,  # Right shoulder, arm at side
    "left_shoulder": 33.0,  # Left shoulder, arm at side
    "right_elbow": 162.0,   # Right elbow, arm extended
    "left_elbow": 162.0,  # Left elbow, arm extended
    "right_knee": 178.0,   # Right knee, leg straight
    "left_knee": 178.0,  # Left knee, leg straight
    "hip": 94.0,         # Hip angle when standing
}

# Default nautical angle offsets (neutral standing position) - Nested structure
DEFAULT_NAUTICAL_OFFSETS = {
    "neck": {"alpha": 180.0, "beta": 3.5, "gamma": 0.0},
    "torso": {"alpha": 90.0, "beta": 2.8, "gamma": 2.8},
    "right_shoulder": {"alpha": 0.0, "beta": 9.6, "gamma": 0.0, "elevation": 94.0},
    "left_shoulder": {"alpha": 21.0, "beta": -15.3, "gamma": 174.2, "elevation": 89.5},
    "right_elbow": {"angle": 170.5},
    "left_elbow": {"angle": 170.5},
    "right_knee": {"angle": 178.0},
    "left_knee": {"angle": 178.0},
}


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-180, 180).

    Args:
        angle: Angle in degrees

    Returns:
        Normalized angle in degrees
    """
    return (angle + 180) % 360 - 180


def get_calibration_path(
    config_dir: Optional[str] = None,
    mode_3d: Optional[bool] = None
) -> Path:
    """
    Get the path to the calibration file.

    Args:
        config_dir: Optional config directory. If None, uses default location.
        mode_3d: If True, returns path for 3D calibration file.
                 If False, returns path for 2D calibration file.
                 If None, returns default path (calibration_data.yaml).

    Returns:
        Path to calibration file (calibration_data_2d.yaml, calibration_data_3d.yaml,
        or calibration_data.yaml if mode_3d is None)
    """
    # Determine filename based on mode
    if mode_3d is True:
        filename = "calibration_data_3d.yaml"
    elif mode_3d is False:
        filename = "calibration_data_2d.yaml"
    else:
        # Default fallback (legacy compatibility)
        filename = "calibration_data.yaml"

    if config_dir:
        return Path(config_dir) / filename

    # Default: project root or user home
    project_root = Path(__file__).parent.parent.parent
    return project_root / filename


def load_calibration(path: Optional[Path] = None) -> Dict[str, float]:
    """
    Load calibration offsets from YAML file.

    Args:
        path: Path to calibration file. If None, uses default location.

    Returns:
        Dictionary of calibration offsets. Returns defaults if file not found.
    """
    if path is None:
        path = get_calibration_path()

    if not path.exists():
        return DEFAULT_OFFSETS.copy()

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if data and 'offsets' in data:
            # Merge with defaults (in case new keys were added)
            offsets = DEFAULT_OFFSETS.copy()
            offsets.update(data['offsets'])
            return offsets

    except Exception as e:
        print(f"Warning: Failed to load calibration from {path}: {e}")

    return DEFAULT_OFFSETS.copy()


def save_calibration(
    offsets: Dict[str, float],
    path: Optional[Path] = None,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Save calibration offsets to YAML file.

    Args:
        offsets: Dictionary of calibration offsets
        path: Path to save file. If None, uses default location.
        metadata: Optional metadata (e.g., date, frames_used)

    Returns:
        True if saved successfully, False otherwise
    """
    if path is None:
        path = get_calibration_path()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'offsets': offsets,
        'metadata': metadata or {},
        'created_at': datetime.now().isoformat(),
    }

    # Add default metadata
    if 'description' not in data['metadata']:
        data['metadata']['description'] = 'REBA calibration offsets from neutral standing position'

    try:
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"Error saving calibration to {path}: {e}")
        return False


def reset_calibration(path: Optional[Path] = None) -> bool:
    """
    Reset calibration to default values.

    Args:
        path: Path to calibration file. If None, uses default location.

    Returns:
        True if reset successfully
    """
    return save_calibration(
        DEFAULT_OFFSETS.copy(),
        path,
        metadata={'reset': True, 'reason': 'Manual reset to defaults'}
    )


class CalibrationManager:
    """
    Manages calibration data loading, saving, and application.

    Provides a singleton-like interface for accessing calibration offsets
    throughout the application.
    """

    _instance: Optional['CalibrationManager'] = None
    _offsets: Dict[str, float] = {}
    _loaded: bool = False
    _mode_3d: Optional[bool] = None  # Track current calibration mode

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self.reload()

    def reload(self, path: Optional[Path] = None, mode_3d: Optional[bool] = None) -> None:
        """
        Reload calibration from file.

        Args:
            path: Optional explicit path to calibration file.
            mode_3d: If True, load 3D calibration. If False, load 2D calibration.
                     If None, use stored mode or try to load default file.
        """
        # Update the mode if specified
        if mode_3d is not None:
            self._mode_3d = mode_3d

        # If no explicit path provided, determine based on mode
        if path is None:
            path = get_calibration_path(mode_3d=self._mode_3d)

        self._offsets = load_calibration(path)
        self._loaded = True

    def save(self, path: Optional[Path] = None, metadata: Optional[Dict] = None) -> bool:
        """Save current calibration to file."""
        return save_calibration(self._offsets, path, metadata)

    def update(self, offsets: Dict[str, float]) -> None:
        """Update calibration offsets."""
        self._offsets.update(offsets)

    def get_offset(self, name: str) -> float:
        """Get offset for a specific angle."""
        return self._offsets.get(name, 0.0)

    def get_all_offsets(self) -> Dict[str, float]:
        """Get all calibration offsets."""
        return self._offsets.copy()

    @property
    def offsets(self):
        """Get offsets (property accessor)."""
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        """Set offsets (property setter)."""
        self._offsets = value

    def apply(self, angle: float, name: str) -> float:
        """
        Apply calibration to an angle.

        Args:
            angle: Raw measured angle
            name: Name of the angle (e.g., 'neck', 'right_elbow')

        Returns:
            Calibrated angle (deviation from neutral)
        """
        if angle == 0.0:
            return 0.0
        offset = self.get_offset(name)
        return abs(angle - offset)

    def apply_all(self, angles: Dict[str, float]) -> Dict[str, float]:
        """
        Apply calibration to all angles.

        Args:
            angles: Dictionary of raw angles

        Returns:
            Dictionary of calibrated angles
        """
        return {
            name: self.apply(angle, name)
            for name, angle in angles.items()
        }

    def apply_nested(self, angles: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Apply calibration to nested angle structure (nautical angles).

        Different calibration formulas are applied based on the segment:
        - Cou (neck): normalize_angle(angle - offset)
        - Buste (torso): angle - offset
        - Épaules (shoulders): offset - angle (inverted)
        - Others: abs(angle - offset)

        Args:
            angles: Nested dictionary of raw angles
                {
                    "neck": {"alpha": float, "beta": float, "gamma": float},
                    "torso": {"alpha": float, "beta": float, "gamma": float},
                    ...
                }

        Returns:
            Nested dictionary of calibrated angles (same structure as input)
        """
        import numpy as np

        calibrated = {}

        for segment, angle_dict in angles.items():
            # If no offset exists for this segment, keep raw values
            if segment not in self._offsets:
                calibrated[segment] = angle_dict.copy()
                continue

            segment_offsets = self._offsets[segment]

            # Handle nested offsets (dict) or simple offsets (float - legacy)
            if not isinstance(segment_offsets, dict):
                # Legacy format: single float offset
                calibrated[segment] = angle_dict.copy()
                continue

            calibrated[segment] = {}

            for angle_name, value in angle_dict.items():
                # Skip NaN values
                if isinstance(value, float) and np.isnan(value):
                    calibrated[segment][angle_name] = value
                    continue

                # Get offset for this component
                offset = segment_offsets.get(angle_name, 0.0)

                # Apply calibration formula based on segment
                if segment == "neck":
                    # Neck: normalize to [-180, 180)
                    calibrated[segment][angle_name] = normalize_angle(value - offset)
                elif segment == "torso":
                    # Torso: direct subtraction
                    calibrated[segment][angle_name] = value - offset
                elif "shoulder" in segment:
                    # Shoulders: inverted (offset - angle)
                    calibrated[segment][angle_name] = offset - value
                else:
                    # Others (elbows, knees): absolute value
                    calibrated[segment][angle_name] = abs(value - offset)

        return calibrated

    @property
    def is_calibrated(self) -> bool:
        """Check if custom calibration has been loaded."""
        path = get_calibration_path()
        return path.exists()


# Global instance
_calibration_manager: Optional[CalibrationManager] = None


def get_calibration_manager() -> CalibrationManager:
    """Get the global calibration manager instance."""
    global _calibration_manager
    if _calibration_manager is None:
        _calibration_manager = CalibrationManager()
    return _calibration_manager
