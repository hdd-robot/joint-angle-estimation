"""
YAML Configuration loader for REBA 3D.

Loads and manages configuration from a YAML file.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml

# Logger for this module (configured later)
logger = logging.getLogger("reba_3d.config")


# Default configuration
DEFAULT_CONFIG = {
    "gui": {
        "width": 1280,
        "height": 720,
        "title": "REBA 3D - Ergonomic Postural Analysis",
        "fps": 30,
        "left_panel_width": 180,
        "right_panel_width": 250,
    },
    "realsense": {
        "color_width": 640,
        "color_height": 480,
        "color_fps": 30,
        "depth_width": 640,
        "depth_height": 480,
        "depth_fps": 30,
        "realtime_playback": True,
    },
    "paths": {
        "bag_directory": "bag",
        "default_bag_file": "recording.bag",
        "output_directory": "output",
        "openpose_path": "~/openpose_alt",
        "keypoints_directory": "../keypoints",
    },
    "reba": {
        "confidence_threshold": 0.35,
        "window_size": 30,
        "poly_order": 2,
        "feet_contact_threshold": 0.10,
        "video_fps": 15,
        "load_score": 0,
        "coupling_score": 0,
        "activity_score": 0,
    },
    "calibration": {
        "duration": 5,
        "n_neutre": 30,
        "skip_windows": 1,
        "save_file": "calibration_data.yaml",
        "neutral_frame_start": 0,  # None = use static offsets, or frame number
        "neutral_frame_end": 60,    # None = use static offsets, or frame number
    },
    "display": {
        "show_scores_on_start": True,
        "show_3d_angles": True,
        "show_2d_scores": True,
        "risk_colors": {
            "negligible": [0, 200, 0],
            "low": [200, 150, 0],
            "medium": [0, 150, 255],
            "high": [0, 0, 255],
            "very_high": [136, 47, 99],
        },
    },
    "logging": {
        "max_lines": 35,
        "font_size": 16,
        "save_to_file": False,
        "log_file": "reba_3d.log",
    },
    "export": {
        "format": "json",
        "include_raw_angles": True,
        "include_detailed_scores": True,
    },
}


@dataclass
class GUIConfig:
    """GUI configuration."""
    width: int = 1280
    height: int = 720
    title: str = "REBA 3D - Analyse Posturale Ergonomique"
    fps: int = 30
    left_panel_width: int = 180
    right_panel_width: int = 250


@dataclass
class RealSenseConfig:
    """RealSense camera configuration."""
    color_width: int = 640
    color_height: int = 480
    color_fps: int = 30
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30
    realtime_playback: bool = True


@dataclass
class PathsConfig:
    """Paths configuration."""
    bag_directory: str = "bag"
    default_bag_file: str = "recording.bag"
    output_directory: str = "output"
    openpose_path: str = "~/openpose_alt"
    keypoints_directory: str = "."

    def get_bag_path(self) -> Path:
        """Returns the full path to the default .bag file."""
        return Path(self.bag_directory) / self.default_bag_file

    def get_openpose_path(self) -> Path:
        """Returns the expanded OpenPose path."""
        return Path(os.path.expanduser(self.openpose_path))

    def get_keypoints_directory(self) -> Path:
        """Returns the expanded keypoints directory path."""
        return Path(os.path.expanduser(self.keypoints_directory))


@dataclass
class REBAConfig:
    """REBA processing configuration."""
    confidence_threshold: float = 0.35
    window_size: int = 30
    poly_order: int = 2
    feet_contact_threshold: float = 0.10
    video_fps: int = 15
    # Additional REBA scores
    # load_score: 0=<5kg, 1=5-10kg, 2=>10kg, +1 if shock/rapid force
    load_score: int = 0
    # coupling_score: 0=good grip, 1=acceptable, 2=poor, 3=unacceptable
    coupling_score: int = 0
    # activity_score: +1 for each condition (static posture >1min, repeated movements, rapid changes)
    activity_score: int = 0


@dataclass
class CalibrationConfig:
    """Calibration configuration."""
    duration: int = 5
    n_neutre: int = 30
    skip_windows: int = 1
    save_file: str = "calibration_data.yaml"
    neutral_frame_start: Optional[int] = None  # None = use static offsets
    neutral_frame_end: Optional[int] = None    # None = use static offsets

    def get_neutral_frames(self) -> Optional[tuple]:
        """
        Get neutral frames tuple for custom calibration.

        Returns:
            Tuple (start_frame, end_frame) if both are set, None otherwise
        """
        if self.neutral_frame_start is not None and self.neutral_frame_end is not None:
            return (self.neutral_frame_start, self.neutral_frame_end)
        return None


@dataclass
class DisplayConfig:
    """Display configuration."""
    show_scores_on_start: bool = True
    show_3d_angles: bool = True
    show_2d_scores: bool = True
    risk_colors: Dict[str, list] = field(default_factory=lambda: {
        "negligible": [0, 200, 0],
        "low": [200, 150, 0],
        "medium": [0, 150, 255],
        "high": [0, 0, 255],
        "very_high": [136, 47, 99],
    })

    def get_risk_color(self, level: str) -> tuple:
        """Returns the BGR color for a risk level."""
        color = self.risk_colors.get(level, [128, 128, 128])
        return tuple(color)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    max_lines: int = 35
    font_size: int = 16
    save_to_file: bool = False
    log_file: str = "reba_3d.log"


@dataclass
class ExportConfig:
    """Export configuration."""
    format: str = "json"
    include_raw_angles: bool = True
    include_detailed_scores: bool = True


class Config:
    """
    Main configuration manager.

    Loads configuration from a YAML file and provides
    typed access to the different sections.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration.

        Args:
            config_path: Path to config.yaml file (optional)
        """
        self._raw_config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None

        # Configuration sections
        self.gui = GUIConfig()
        self.realsense = RealSenseConfig()
        self.paths = PathsConfig()
        self.reba = REBAConfig()
        self.calibration = CalibrationConfig()
        self.display = DisplayConfig()
        self.logging = LoggingConfig()
        self.export = ExportConfig()

        # Load the configuration
        if config_path:
            self.load(config_path)
        else:
            # Look for config.yaml in current or parent directory
            self._auto_load()

    def _auto_load(self) -> None:
        """Automatically searches for and loads the config.yaml file."""
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd().parent / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                self.load(str(path))
                return

        # Use default configuration
        logger.debug("config.yaml file not found, using default values")
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        """Applies the default configuration."""
        self._raw_config = DEFAULT_CONFIG.copy()
        self._update_dataclasses()

    def load(self, config_path: str) -> None:
        """
        Loads configuration from a YAML file.

        Args:
            config_path: Path to the YAML file
        """
        self._config_path = Path(config_path)

        if not self._config_path.exists():
            logger.warning(f"File not found: {config_path}")
            self._apply_defaults()
            return

        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}

            # Merge with default values
            self._merge_with_defaults()
            self._update_dataclasses()

            logger.info(f"Configuration loaded: {config_path}")

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            self._apply_defaults()
        except Exception as e:
            logger.error(f"Loading error: {e}")
            self._apply_defaults()

    def _merge_with_defaults(self) -> None:
        """Merges the loaded configuration with default values."""
        def deep_merge(default: dict, loaded: dict) -> dict:
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        self._raw_config = deep_merge(DEFAULT_CONFIG, self._raw_config)

    def _update_dataclasses(self) -> None:
        """Updates dataclasses from raw configuration."""
        if "gui" in self._raw_config:
            self.gui = GUIConfig(**self._raw_config["gui"])

        if "realsense" in self._raw_config:
            self.realsense = RealSenseConfig(**self._raw_config["realsense"])

        if "paths" in self._raw_config:
            self.paths = PathsConfig(**self._raw_config["paths"])

        if "reba" in self._raw_config:
            self.reba = REBAConfig(**self._raw_config["reba"])

        if "calibration" in self._raw_config:
            self.calibration = CalibrationConfig(**self._raw_config["calibration"])

        if "display" in self._raw_config:
            self.display = DisplayConfig(**self._raw_config["display"])

        if "logging" in self._raw_config:
            self.logging = LoggingConfig(**self._raw_config["logging"])

        if "export" in self._raw_config:
            self.export = ExportConfig(**self._raw_config["export"])

    def save(self, config_path: Optional[str] = None) -> None:
        """
        Saves configuration to a YAML file.

        Args:
            config_path: Destination path (optional, uses original path)
        """
        if config_path:
            path = Path(config_path)
        elif self._config_path:
            path = self._config_path
        else:
            path = Path("config.yaml")

        # Rebuild configuration from dataclasses
        config_dict = {
            "gui": {
                "width": self.gui.width,
                "height": self.gui.height,
                "title": self.gui.title,
                "fps": self.gui.fps,
                "left_panel_width": self.gui.left_panel_width,
                "right_panel_width": self.gui.right_panel_width,
            },
            "realsense": {
                "color_width": self.realsense.color_width,
                "color_height": self.realsense.color_height,
                "color_fps": self.realsense.color_fps,
                "depth_width": self.realsense.depth_width,
                "depth_height": self.realsense.depth_height,
                "depth_fps": self.realsense.depth_fps,
                "realtime_playback": self.realsense.realtime_playback,
            },
            "paths": {
                "bag_directory": self.paths.bag_directory,
                "default_bag_file": self.paths.default_bag_file,
                "output_directory": self.paths.output_directory,
                "openpose_path": self.paths.openpose_path,
                "keypoints_directory": self.paths.keypoints_directory,
            },
            "reba": {
                "confidence_threshold": self.reba.confidence_threshold,
                "window_size": self.reba.window_size,
                "poly_order": self.reba.poly_order,
                "feet_contact_threshold": self.reba.feet_contact_threshold,
                "video_fps": self.reba.video_fps,
                "load_score": self.reba.load_score,
                "coupling_score": self.reba.coupling_score,
                "activity_score": self.reba.activity_score,
            },
            "calibration": {
                "duration": self.calibration.duration,
                "n_neutre": self.calibration.n_neutre,
                "skip_windows": self.calibration.skip_windows,
                "save_file": self.calibration.save_file,
            },
            "display": {
                "show_scores_on_start": self.display.show_scores_on_start,
                "show_3d_angles": self.display.show_3d_angles,
                "show_2d_scores": self.display.show_2d_scores,
                "risk_colors": self.display.risk_colors,
            },
            "logging": {
                "max_lines": self.logging.max_lines,
                "font_size": self.logging.font_size,
                "save_to_file": self.logging.save_to_file,
                "log_file": self.logging.log_file,
            },
            "export": {
                "format": self.export.format,
                "include_raw_angles": self.export.include_raw_angles,
                "include_detailed_scores": self.export.include_detailed_scores,
            },
        }

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f"Configuration saved: {path}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Access by key with dot notation.

        Args:
            key: Key with dot notation (e.g., "gui.width")
            default: Default value if key doesn't exist

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._raw_config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Modifies a configuration value.

        Args:
            key: Key with dot notation (e.g., "gui.width")
            value: New value
        """
        keys = key.split(".")
        config = self._raw_config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self._update_dataclasses()


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Returns the global configuration instance.

    Args:
        config_path: Path to config.yaml file (optional)

    Returns:
        Config instance
    """
    global _config

    if _config is None or config_path:
        _config = Config(config_path)

    return _config


def load_config(config_path: str) -> Config:
    """
    Loads a new configuration.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Config instance
    """
    global _config
    _config = Config(config_path)
    return _config
