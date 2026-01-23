"""
YAML Configuration loader for REBA 3D.

Charge et gère la configuration depuis un fichier YAML.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml

# Logger pour ce module (configuré plus tard)
logger = logging.getLogger("reba_3d.config")


# Configuration par défaut
DEFAULT_CONFIG = {
    "gui": {
        "width": 1280,
        "height": 720,
        "title": "REBA 3D - Analyse Posturale Ergonomique",
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
    },
    "reba": {
        "confidence_threshold": 0.35,
        "window_size": 30,
        "poly_order": 2,
        "feet_contact_threshold": 0.10,
        "video_fps": 15,
    },
    "calibration": {
        "duration": 5,
        "skip_initial_frames": 1,
        "save_file": "calibration_data.yaml",
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
    """Configuration de l'interface graphique."""
    width: int = 1280
    height: int = 720
    title: str = "REBA 3D - Analyse Posturale Ergonomique"
    fps: int = 30
    left_panel_width: int = 180
    right_panel_width: int = 250


@dataclass
class RealSenseConfig:
    """Configuration de la caméra RealSense."""
    color_width: int = 640
    color_height: int = 480
    color_fps: int = 30
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30
    realtime_playback: bool = True


@dataclass
class PathsConfig:
    """Configuration des chemins."""
    bag_directory: str = "bag"
    default_bag_file: str = "recording.bag"
    output_directory: str = "output"
    openpose_path: str = "~/openpose_alt"

    def get_bag_path(self) -> Path:
        """Retourne le chemin complet du fichier .bag par défaut."""
        return Path(self.bag_directory) / self.default_bag_file

    def get_openpose_path(self) -> Path:
        """Retourne le chemin OpenPose expandé."""
        return Path(os.path.expanduser(self.openpose_path))


@dataclass
class REBAConfig:
    """Configuration du traitement REBA."""
    confidence_threshold: float = 0.35
    window_size: int = 30
    poly_order: int = 2
    feet_contact_threshold: float = 0.10
    video_fps: int = 15


@dataclass
class CalibrationConfig:
    """Configuration de la calibration."""
    duration: int = 5
    skip_initial_frames: int = 1
    save_file: str = "calibration_data.yaml"


@dataclass
class DisplayConfig:
    """Configuration de l'affichage."""
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
        """Retourne la couleur BGR pour un niveau de risque."""
        color = self.risk_colors.get(level, [128, 128, 128])
        return tuple(color)


@dataclass
class LoggingConfig:
    """Configuration des logs."""
    max_lines: int = 35
    font_size: int = 16
    save_to_file: bool = False
    log_file: str = "reba_3d.log"


@dataclass
class ExportConfig:
    """Configuration de l'export."""
    format: str = "json"
    include_raw_angles: bool = True
    include_detailed_scores: bool = True


class Config:
    """
    Gestionnaire de configuration principal.

    Charge la configuration depuis un fichier YAML et fournit
    un accès typé aux différentes sections.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise la configuration.

        Args:
            config_path: Chemin vers le fichier config.yaml (optionnel)
        """
        self._raw_config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None

        # Sections de configuration
        self.gui = GUIConfig()
        self.realsense = RealSenseConfig()
        self.paths = PathsConfig()
        self.reba = REBAConfig()
        self.calibration = CalibrationConfig()
        self.display = DisplayConfig()
        self.logging = LoggingConfig()
        self.export = ExportConfig()

        # Charger la configuration
        if config_path:
            self.load(config_path)
        else:
            # Chercher config.yaml dans le répertoire courant ou parent
            self._auto_load()

    def _auto_load(self) -> None:
        """Recherche et charge automatiquement le fichier config.yaml."""
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd().parent / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                self.load(str(path))
                return

        # Utiliser la configuration par défaut
        logger.debug("Fichier config.yaml non trouvé, utilisation des valeurs par défaut")
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        """Applique la configuration par défaut."""
        self._raw_config = DEFAULT_CONFIG.copy()
        self._update_dataclasses()

    def load(self, config_path: str) -> None:
        """
        Charge la configuration depuis un fichier YAML.

        Args:
            config_path: Chemin vers le fichier YAML
        """
        self._config_path = Path(config_path)

        if not self._config_path.exists():
            logger.warning(f"Fichier non trouvé: {config_path}")
            self._apply_defaults()
            return

        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}

            # Fusionner avec les valeurs par défaut
            self._merge_with_defaults()
            self._update_dataclasses()

            logger.info(f"Configuration chargée: {config_path}")

        except yaml.YAMLError as e:
            logger.error(f"Erreur de parsing YAML: {e}")
            self._apply_defaults()
        except Exception as e:
            logger.error(f"Erreur de chargement: {e}")
            self._apply_defaults()

    def _merge_with_defaults(self) -> None:
        """Fusionne la configuration chargée avec les valeurs par défaut."""
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
        """Met à jour les dataclasses depuis la configuration brute."""
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
        Sauvegarde la configuration dans un fichier YAML.

        Args:
            config_path: Chemin de destination (optionnel, utilise le chemin d'origine)
        """
        if config_path:
            path = Path(config_path)
        elif self._config_path:
            path = self._config_path
        else:
            path = Path("config.yaml")

        # Reconstruire la configuration depuis les dataclasses
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
            },
            "reba": {
                "confidence_threshold": self.reba.confidence_threshold,
                "window_size": self.reba.window_size,
                "poly_order": self.reba.poly_order,
                "feet_contact_threshold": self.reba.feet_contact_threshold,
                "video_fps": self.reba.video_fps,
            },
            "calibration": {
                "duration": self.calibration.duration,
                "skip_initial_frames": self.calibration.skip_initial_frames,
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

        logger.info(f"Configuration sauvegardée: {path}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Accès par clé avec notation pointée.

        Args:
            key: Clé avec notation pointée (ex: "gui.width")
            default: Valeur par défaut si la clé n'existe pas

        Returns:
            Valeur de la configuration
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
        Modifie une valeur de configuration.

        Args:
            key: Clé avec notation pointée (ex: "gui.width")
            value: Nouvelle valeur
        """
        keys = key.split(".")
        config = self._raw_config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self._update_dataclasses()


# Instance globale de configuration
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Retourne l'instance globale de configuration.

    Args:
        config_path: Chemin vers le fichier config.yaml (optionnel)

    Returns:
        Instance Config
    """
    global _config

    if _config is None or config_path:
        _config = Config(config_path)

    return _config


def load_config(config_path: str) -> Config:
    """
    Charge une nouvelle configuration.

    Args:
        config_path: Chemin vers le fichier config.yaml

    Returns:
        Instance Config
    """
    global _config
    _config = Config(config_path)
    return _config
