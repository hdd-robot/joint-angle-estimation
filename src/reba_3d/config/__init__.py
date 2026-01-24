"""Configuration module for reba_3d package."""

from reba_3d.config.settings import (
    OPENPOSE_PATH,
    OUTPUT_DIR,
    CONFIDENCE_THRESHOLD,
    WINDOW_SIZE,
    POLY_ORDER,
    FEET_CONTACT_THRESHOLD,
    FPS,
    KEYPOINT_NAMES,
)

from reba_3d.config.yaml_config import (
    Config,
    get_config,
    load_config,
    GUIConfig,
    RealSenseConfig,
    PathsConfig,
    REBAConfig,
    CalibrationConfig,
    DisplayConfig,
    LoggingConfig,
    ExportConfig,
)

from reba_3d.config.calibration_store import (
    CalibrationManager,
    get_calibration_manager,
    load_calibration,
    save_calibration,
    reset_calibration,
    DEFAULT_OFFSETS,
)

__all__ = [
    # Legacy settings
    "OPENPOSE_PATH",
    "OUTPUT_DIR",
    "CONFIDENCE_THRESHOLD",
    "WINDOW_SIZE",
    "POLY_ORDER",
    "FEET_CONTACT_THRESHOLD",
    "FPS",
    "KEYPOINT_NAMES",
    # YAML config
    "Config",
    "get_config",
    "load_config",
    "GUIConfig",
    "RealSenseConfig",
    "PathsConfig",
    "REBAConfig",
    "CalibrationConfig",
    "DisplayConfig",
    "LoggingConfig",
    "ExportConfig",
    # Calibration store
    "CalibrationManager",
    "get_calibration_manager",
    "load_calibration",
    "save_calibration",
    "reset_calibration",
    "DEFAULT_OFFSETS",
]
