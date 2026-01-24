"""
Calibration data storage and retrieval.

Handles saving and loading calibration offsets to/from YAML files.
"""

import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import yaml


# Default calibration offsets (neutral standing position)
DEFAULT_OFFSETS = {
    "cou": 154.0,           # Neck angle when standing straight
    "epaule_droite": 33.0,  # Right shoulder, arm at side
    "epaule_gauche": 33.0,  # Left shoulder, arm at side
    "coude_droit": 162.0,   # Right elbow, arm extended
    "coude_gauche": 162.0,  # Left elbow, arm extended
    "genou_droit": 178.0,   # Right knee, leg straight
    "genou_gauche": 178.0,  # Left knee, leg straight
    "hanche": 94.0,         # Hip angle when standing
}


def get_calibration_path(config_dir: Optional[str] = None) -> Path:
    """
    Get the path to the calibration file.

    Args:
        config_dir: Optional config directory. If None, uses default location.

    Returns:
        Path to calibration_data.yaml
    """
    if config_dir:
        return Path(config_dir) / "calibration_data.yaml"

    # Default: project root or user home
    project_root = Path(__file__).parent.parent.parent
    return project_root / "calibration_data.yaml"


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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self.reload()

    def reload(self, path: Optional[Path] = None) -> None:
        """Reload calibration from file."""
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

    def apply(self, angle: float, name: str) -> float:
        """
        Apply calibration to an angle.

        Args:
            angle: Raw measured angle
            name: Name of the angle (e.g., 'cou', 'coude_droit')

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
