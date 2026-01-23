"""
Logging system for REBA 3D.

Provides a centralized logging configuration with console and file output.
"""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional


class LogLevel(Enum):
    """Log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Format des messages
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Logger racine pour l'application
ROOT_LOGGER_NAME = "reba_3d"

# Flag pour éviter la configuration multiple
_logging_configured = False


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Configure le système de logging.

    Args:
        level: Niveau de log minimum
        log_file: Chemin du fichier de log (None = pas de fichier)
        console: Activer la sortie console

    Returns:
        Logger racine configuré
    """
    global _logging_configured

    logger = logging.getLogger(ROOT_LOGGER_NAME)

    # Éviter la configuration multiple
    if _logging_configured:
        return logger

    logger.setLevel(level.value)

    # Supprimer les handlers existants
    logger.handlers.clear()

    # Handler console
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.value)
        console_formatter = logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Handler fichier
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level.value)
        file_formatter = logging.Formatter(FILE_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Ne pas propager aux loggers parents
    logger.propagate = False

    _logging_configured = True

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retourne un logger pour le module spécifié.

    Args:
        name: Nom du module (ex: "gui.app"). Si None, retourne le logger racine.

    Returns:
        Logger configuré
    """
    if name:
        return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    return logging.getLogger(ROOT_LOGGER_NAME)


def setup_from_config(config) -> logging.Logger:
    """
    Configure le logging depuis la configuration YAML.

    Args:
        config: Instance de Config

    Returns:
        Logger configuré
    """
    log_file = None
    if config.logging.save_to_file:
        log_file = config.logging.log_file

    return setup_logging(
        level=LogLevel.INFO,
        log_file=log_file,
        console=True
    )
