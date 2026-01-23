#!/usr/bin/env python3
"""
REBA 3D - Main Launcher

Lance l'interface graphique pour l'analyse posturale ergonomique REBA.

Usage:
    python main.py [options]

Options:
    --config FILE       Fichier de configuration YAML (défaut: config.yaml)
    --width WIDTH       Largeur de la fenêtre (surcharge config.yaml)
    --height HEIGHT     Hauteur de la fenêtre (surcharge config.yaml)
    --bag-dir DIR       Répertoire des fichiers .bag (surcharge config.yaml)
    --bag-file FILE     Nom du fichier .bag par défaut (surcharge config.yaml)
    --debug             Activer le mode debug (logs détaillés)
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="REBA 3D - Interface graphique d'analyse posturale",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Fichier de configuration YAML (défaut: config.yaml)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Largeur de la fenêtre (surcharge config.yaml)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Hauteur de la fenêtre (surcharge config.yaml)"
    )

    parser.add_argument(
        "--bag-dir",
        type=str,
        default=None,
        help="Répertoire des fichiers .bag (surcharge config.yaml)"
    )

    parser.add_argument(
        "--bag-file",
        type=str,
        default=None,
        help="Nom du fichier .bag par défaut (surcharge config.yaml)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activer le mode debug (logs détaillés)"
    )

    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()

    # Configurer le logging
    from reba_3d.utils.logger import setup_logging, get_logger, LogLevel
    log_level = LogLevel.DEBUG if args.debug else LogLevel.INFO
    setup_logging(level=log_level)
    logger = get_logger("main")

    # Vérifier que pygame est installé
    try:
        import pygame
    except ImportError:
        logger.error("pygame n'est pas installé. Installez-le avec: pip install pygame")
        sys.exit(1)

    # Charger la configuration YAML
    from reba_3d.config import get_config
    config = get_config(args.config)

    # Configurer le fichier de log si activé dans la config
    if config.logging.save_to_file:
        setup_logging(level=log_level, log_file=config.logging.log_file)

    # Récupérer les valeurs (args surcharge config)
    width = args.width if args.width is not None else config.gui.width
    height = args.height if args.height is not None else config.gui.height
    bag_dir = args.bag_dir if args.bag_dir is not None else config.paths.bag_directory
    bag_file = args.bag_file if args.bag_file is not None else config.paths.default_bag_file

    # Créer le répertoire bag s'il n'existe pas
    bag_path = Path(bag_dir)
    if not bag_path.exists():
        bag_path.mkdir(parents=True)
        logger.info(f"Répertoire créé: {bag_path}")

    # Lancer l'application
    logger.info("=" * 50)
    logger.info("  REBA 3D - Analyse Posturale Ergonomique")
    logger.info("=" * 50)
    logger.info(f"  Résolution: {width}x{height}")
    logger.info(f"  Répertoire BAG: {bag_path.absolute()}")
    logger.info(f"  Fichier par défaut: {bag_file}")
    logger.info("=" * 50)
    logger.info("Contrôles: ESPACE=Capture, ESC=Quitter")

    from reba_3d.gui.app import REBAApp

    app = REBAApp(
        width=args.width,
        height=args.height,
        bag_directory=args.bag_dir,
        default_bag_name=args.bag_file
    )

    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur")
    except Exception as e:
        logger.exception(f"Erreur fatale: {e}")
        sys.exit(1)

    logger.info("Application terminée")


if __name__ == "__main__":
    main()
