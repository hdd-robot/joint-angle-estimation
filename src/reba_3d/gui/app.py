"""
Main REBA GUI Application.

Interface graphique Pygame avec 3 colonnes:
- Gauche: Boutons de contrôle
- Centre: Affichage vidéo
- Droite: Panneau de logs
"""

import threading
import time
from pathlib import Path
from typing import Optional, List, Dict

import pygame
import numpy as np

from reba_3d.gui.components import (
    Button, ToggleButton, RadioButtonGroup, LogPanel, VideoDisplay, ScoreGraph,
    ScoreButtonGroup,
    WHITE, BLACK, GRAY, DARK_GRAY, GREEN, RED, BLUE, ORANGE
)

# Hauteur du graphique en bas de la fenêtre
GRAPH_HEIGHT = 140
from reba_3d.config import get_config
from reba_3d.config.settings import OPENPOSE_MODE
from reba_3d.config.calibration_store import (
    get_calibration_manager, save_calibration, load_calibration
)
from reba_3d.core.angles import (
    calculate_angles_from_keypoints_2d,
    calculate_angles_from_keypoints_3d,
    compute_calibration_offsets
)
from reba_3d.reba.realtime_scorer import RealtimeREBAScorer, REBAScore
from reba_3d.utils.logger import get_logger

# Logger pour ce module
logger = get_logger("gui.app")


class REBAApp:
    """
    Application principale REBA avec interface Pygame.

    Attributes:
        width: Largeur de la fenêtre
        height: Hauteur de la fenêtre
        running: État d'exécution
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        bag_directory: Optional[str] = None,
        default_bag_name: Optional[str] = None
    ):
        """
        Initialise l'application REBA.

        Args:
            width: Largeur de la fenêtre (si None, utilise config.yaml)
            height: Hauteur de la fenêtre (si None, utilise config.yaml)
            bag_directory: Répertoire des fichiers .bag (si None, utilise config.yaml)
            default_bag_name: Nom du fichier .bag par défaut (si None, utilise config.yaml)
        """
        # Charger la configuration YAML
        self.config = get_config()

        # Utiliser les valeurs passées en argument ou celles de la config
        # Ajouter GRAPH_HEIGHT pour le graphique en bas
        self.width = width if width is not None else self.config.gui.width
        self.height = (height if height is not None else self.config.gui.height) + GRAPH_HEIGHT
        self.bag_directory = Path(bag_directory if bag_directory is not None else self.config.paths.bag_directory)
        self.default_bag_name = default_bag_name if default_bag_name is not None else self.config.paths.default_bag_file

        # Dimensions des colonnes depuis la config
        self.left_panel_width = self.config.gui.left_panel_width
        self.right_panel_width = self.config.gui.right_panel_width
        self.fps = self.config.gui.fps

        # État de l'application
        self.running = False
        self.mode = "offline"  # "inline" ou "offline"
        self.calibration_active = False
        self.show_scores = True
        self.capturing = False
        self.paused = False

        # Composants vidéo
        self.current_frame = None
        self.frozen_frame = None  # Frame figée pour le mode pause inline
        self.frame_lock = threading.Lock()

        # Thread de capture
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_capture = threading.Event()
        self.pause_event = threading.Event()  # Event pour gérer la pause

        # OpenPose detector (initialisé au premier besoin)
        self.openpose_detector = None
        self._openpose_initialized = False

        # Calibration data
        self.calibration_start_time = None
        self.calibration_angles: List[Dict[str, float]] = []  # Liste des angles collectés
        self.calibration_duration = self.config.calibration.duration  # Durée en secondes
        self.calibration_manager = get_calibration_manager()

        # REBA scoring
        self.reba_scorer = RealtimeREBAScorer(buffer_size=10)
        # Initialiser avec les valeurs de config
        self.reba_scorer.set_load_score(self.config.reba.load_score)
        self.reba_scorer.set_coupling_score(self.config.reba.coupling_score)
        self.reba_scorer.set_activity_score(self.config.reba.activity_score)
        self.current_reba_score: Optional[REBAScore] = None
        self.current_angles: Dict[str, float] = {}

        # Depth/3D data
        self.depth_intrinsics = None  # Camera intrinsics for 3D projection
        self.use_3d = True  # Use 3D angles when available
        self.current_depth_frame = None  # Depth frame pour traitement temps réel
        self._last_depth_data = None  # Dernières données depth copiées (numpy array)
        self.frozen_depth_data = None  # Données depth figées pour mesure (numpy array)

        # Mode mesure de distance
        self.measure_mode = False
        self.measure_points = []  # Liste de points cliqués [(x, y), ...]
        self.measure_distance = None  # Distance calculée en mètres
        self.measure_points_3d = []  # Points 3D correspondants

        # Comparison mode (shows both 2D and 3D scores)
        self.show_comparison = False
        self.reba_scorer_2d = RealtimeREBAScorer(buffer_size=10)  # Scorer séparé pour 2D
        # Initialiser avec les mêmes valeurs de config
        self.reba_scorer_2d.set_load_score(self.config.reba.load_score)
        self.reba_scorer_2d.set_coupling_score(self.config.reba.coupling_score)
        self.reba_scorer_2d.set_activity_score(self.config.reba.activity_score)
        self.current_reba_score_2d: Optional[REBAScore] = None

        # Initialisation Pygame
        pygame.init()
        pygame.display.set_caption(self.config.gui.title)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        logger.debug(f"Fenêtre créée: {self.width}x{self.height}")

        # Initialisation des composants UI
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialise les composants de l'interface."""
        padding = 10
        button_width = self.left_panel_width - 2 * padding
        button_height = 40

        # Position de départ pour les boutons
        x = padding
        y = padding

        # Titre du panneau gauche
        self.title_font = pygame.font.Font(None, 24)

        # --- Mode Selection (Radio Buttons) ---
        y += 30
        self.mode_selector = RadioButtonGroup(
            x, y,
            options=["Offline", "Inline"],
            callback=self._on_mode_change,
            button_width=button_width,
            button_height=35,
            spacing=5
        )
        y += 80

        # --- Bouton Start/Stop Capture ---
        self.btn_capture = ToggleButton(
            x, y, button_width, button_height,
            text_on="Stop Capture",
            text_off="Start Capture",
            callback=self._on_capture_toggle,
            color_on=RED,
            color_off=GREEN
        )
        y += button_height + padding

        # --- Bouton Pause ---
        self.btn_pause = ToggleButton(
            x, y, button_width, button_height,
            text_on="Reprendre",
            text_off="Pause",
            callback=self._on_pause_toggle,
            color_on=GREEN,
            color_off=ORANGE
        )
        y += button_height + padding

        # --- Bouton Calibration ---
        self.btn_calibration = ToggleButton(
            x, y, button_width, button_height,
            text_on="Stop Calibration",
            text_off="Start Calibration",
            callback=self._on_calibration_toggle,
            color_on=RED,
            color_off=BLUE
        )
        y += button_height + padding

        # --- Bouton Show/Hide Scores ---
        self.btn_scores = ToggleButton(
            x, y, button_width, button_height,
            text_on="Cacher Scores",
            text_off="Afficher Scores",
            callback=self._on_scores_toggle,
            color_on=GRAY,
            color_off=GREEN,
            initial_state=True
        )
        y += button_height + padding

        # --- Bouton Comparaison 2D/3D ---
        self.btn_comparison = ToggleButton(
            x, y, button_width, button_height,
            text_on="Mode Normal",
            text_off="Comparer 2D/3D",
            callback=self._on_comparison_toggle,
            color_on=ORANGE,
            color_off=BLUE
        )
        y += button_height + padding

        # --- Bouton Mesurer Taille ---
        self.btn_measure = ToggleButton(
            x, y, button_width, button_height,
            text_on="Annuler Mesure",
            text_off="Mesurer Taille",
            callback=self._on_measure_toggle,
            color_on=RED,
            color_off=(100, 50, 150)  # Violet
        )
        y += button_height + padding + 10

        # --- Section paramètres REBA ---
        section_font = pygame.font.Font(None, 18)
        self.params_label_y = y
        y += 20

        # Charger les valeurs depuis la config
        reba_cfg = self.config.reba

        # --- Score Charge (Load) ---
        self.score_load = ScoreButtonGroup(
            x, y,
            label="Poids:",
            values=[0, 1, 2, 3],
            callback=self._on_load_change,
            initial_value=reba_cfg.load_score
        )
        y += 35

        # --- Score Prise (Coupling) ---
        self.score_coupling = ScoreButtonGroup(
            x, y,
            label="Prise:",
            values=[0, 1, 2, 3],
            callback=self._on_coupling_change,
            initial_value=reba_cfg.coupling_score
        )
        y += 35

        # --- Score Activité ---
        self.score_activity = ScoreButtonGroup(
            x, y,
            label="Activité:",
            values=[0, 1, 2, 3],
            callback=self._on_activity_change,
            initial_value=reba_cfg.activity_score
        )
        y += 35

        # --- Spacer ---
        # Position du bouton Exit au-dessus du graphique
        y = self.height - GRAPH_HEIGHT - button_height - padding - 10

        # --- Bouton Exit ---
        self.btn_exit = Button(
            x, y, button_width, button_height,
            "Exit",
            callback=self._on_exit,
            color=RED
        )

        # --- Hauteur de la zone principale (sans le graphique) ---
        main_height = self.height - GRAPH_HEIGHT

        # --- Zone vidéo (colonne centrale) ---
        video_x = self.left_panel_width
        video_width = self.width - self.left_panel_width - self.right_panel_width
        self.video_display = VideoDisplay(
            video_x, 0, video_width, main_height
        )

        # --- Panneau de logs (colonne droite) ---
        log_x = self.width - self.right_panel_width
        self.log_panel = LogPanel(
            log_x, 0, self.right_panel_width, main_height,
            max_lines=self.config.logging.max_lines
        )

        # --- Graphique des scores (en bas, toute la largeur) ---
        self.score_graph = ScoreGraph(
            0, main_height, self.width, GRAPH_HEIGHT,
            max_points=300  # ~10 secondes à 30 FPS
        )

        # Message de bienvenue
        self.log_panel.add_info("Application REBA 3D démarrée")
        self.log_panel.add_info(f"Mode: {self.mode}")

        # Vérifier le statut de calibration
        if self.calibration_manager.is_calibrated:
            self.log_panel.add_success("Calibration: Chargée")
        else:
            self.log_panel.add_warning("Calibration: Non effectuée")

    def _on_mode_change(self, mode: str) -> None:
        """Callback pour changement de mode."""
        self.mode = mode.lower()
        self.log_panel.add_info(f"Mode changé: {self.mode}")

        if self.mode == "inline":
            self.log_panel.add_warning("Mode inline: caméra RealSense")
        else:
            bag_path = self.bag_directory / self.default_bag_name
            self.log_panel.add_info(f"Fichier: {bag_path}")

    def _on_capture_toggle(self, state: bool) -> None:
        """Callback pour start/stop capture."""
        if state:
            self._start_capture()
        else:
            self._stop_capture()

    def _on_calibration_toggle(self, state: bool) -> None:
        """Callback pour start/stop calibration."""
        if state:
            # Vérifier que la capture est active
            if not self.capturing:
                self.log_panel.add_error("Démarrez la capture d'abord")
                self.btn_calibration.set_state(False)
                return

            # Démarrer la calibration
            self.calibration_active = True
            self.calibration_angles = []
            self.calibration_start_time = time.time()

            self.log_panel.add_info(f"Calibration ({self.calibration_duration}s)...")
            self.log_panel.add_warning("Restez en position neutre")
            logger.info(f"Calibration démarrée pour {self.calibration_duration}s")

            # Lancer un thread pour arrêter la calibration après la durée
            def calibration_timer():
                time.sleep(self.calibration_duration)
                if self.calibration_active:
                    self._finish_calibration()

            timer_thread = threading.Thread(target=calibration_timer, daemon=True)
            timer_thread.start()
        else:
            # Arrêt manuel de la calibration
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        """Termine la calibration et calcule les offsets."""
        self.calibration_active = False
        self.btn_calibration.set_state(False)

        n_frames = len(self.calibration_angles)
        logger.info(f"Calibration terminée: {n_frames} frames collectées")

        if n_frames < 30:
            self.log_panel.add_error(f"Calibration échouée: {n_frames} frames (min 30)")
            logger.warning("Pas assez de frames pour la calibration")
            self.calibration_angles = []
            self.calibration_start_time = None
            return

        # Calculer les offsets à partir des angles collectés
        try:
            offsets = compute_calibration_offsets(
                self.calibration_angles,
                window_size=30,
                skip_windows=1  # Ignorer la première fenêtre
            )

            if not offsets:
                self.log_panel.add_error("Calibration échouée: pas d'angles valides")
                logger.warning("Aucun offset calculé")
                return

            # Sauvegarder les offsets
            metadata = {
                'frames_collected': n_frames,
                'duration_seconds': self.calibration_duration,
                'mode': self.mode,
            }

            if save_calibration(offsets, metadata=metadata):
                # Recharger le gestionnaire de calibration
                self.calibration_manager.reload()

                self.log_panel.add_success(f"Calibration OK ({n_frames} frames)")
                self.log_panel.add_info("Offsets sauvegardés")

                # Afficher les offsets calculés
                for name, value in offsets.items():
                    logger.info(f"  {name}: {value:.2f}°")

                logger.info("Calibration réussie et sauvegardée")
            else:
                self.log_panel.add_error("Erreur sauvegarde calibration")
                logger.error("Échec sauvegarde calibration")

        except Exception as e:
            self.log_panel.add_error(f"Erreur calibration: {str(e)[:20]}")
            logger.exception(f"Erreur lors du calcul des offsets: {e}")

        # Réinitialiser les données
        self.calibration_angles = []
        self.calibration_start_time = None

    def _on_scores_toggle(self, state: bool) -> None:
        """Callback pour afficher/cacher les scores."""
        self.show_scores = state
        self.video_display.overlay_scores = state
        if state:
            self.log_panel.add_info("Affichage des scores activé")
        else:
            self.log_panel.add_info("Affichage des scores désactivé")

    def _on_comparison_toggle(self, state: bool) -> None:
        """Callback pour activer/désactiver le mode comparaison 2D/3D."""
        self.show_comparison = state
        if state:
            self.log_panel.add_info("Mode comparaison 2D/3D activé")
        else:
            self.log_panel.add_info("Mode comparaison désactivé")

    def _on_load_change(self, value: int) -> None:
        """Callback pour changement du score de charge."""
        self.reba_scorer.set_load_score(value)
        self.reba_scorer_2d.set_load_score(value)
        self.log_panel.add_info(f"Poids: {value}")

    def _on_coupling_change(self, value: int) -> None:
        """Callback pour changement du score de prise."""
        self.reba_scorer.set_coupling_score(value)
        self.reba_scorer_2d.set_coupling_score(value)
        self.log_panel.add_info(f"Prise: {value}")

    def _on_activity_change(self, value: int) -> None:
        """Callback pour changement du score d'activité."""
        self.reba_scorer.set_activity_score(value)
        self.reba_scorer_2d.set_activity_score(value)
        self.log_panel.add_info(f"Activité: {value}")

    def _on_pause_toggle(self, state: bool) -> None:
        """Callback pour pause/reprendre la vidéo."""
        self.paused = state
        if state:
            self.pause_event.set()
            # Figer le frame et le depth actuels
            with self.frame_lock:
                if self.current_frame is not None:
                    self.frozen_frame = self.current_frame.copy()
                # Utiliser les données depth déjà copiées dans _process_frame
                if hasattr(self, '_last_depth_data') and self._last_depth_data is not None:
                    self.frozen_depth_data = self._last_depth_data.copy()
                    logger.debug(f"Depth figé: {self.frozen_depth_data.shape}")
                else:
                    self.frozen_depth_data = None
                    logger.warning("Pas de données depth disponibles")
            self.log_panel.add_info("Pause activée")
            logger.debug("Pause activée")
        else:
            self.pause_event.clear()
            self.frozen_frame = None
            self.frozen_depth_data = None
            # Désactiver le mode mesure si on reprend la lecture
            if self.measure_mode or self.measure_points:
                self._reset_measure_mode()
                self.btn_measure.set_state(False)
            self.log_panel.add_info("Lecture reprise")
            logger.debug("Lecture reprise")

    def _on_measure_toggle(self, state: bool) -> None:
        """Callback pour activer/désactiver le mode mesure de distance."""
        if state:
            # Vérifier qu'on est en pause
            if not self.paused:
                self.log_panel.add_error("Mettez en pause d'abord")
                self.btn_measure.set_state(False)
                return
            # Vérifier qu'on a le depth figé
            if self.frozen_depth_data is None:
                self.log_panel.add_error("Pas de données depth")
                self.btn_measure.set_state(False)
                return
            # Vérifier les intrinsics
            if self.depth_intrinsics is None:
                self.log_panel.add_error("Pas d'intrinsics caméra")
                self.btn_measure.set_state(False)
                return

            self.measure_mode = True
            self.measure_points = []
            self.measure_points_3d = []
            self.measure_distance = None
            self.log_panel.add_info("Mode mesure: cliquez 2 points")
            logger.debug(f"Mode mesure activé, depth shape: {self.frozen_depth_data.shape}")
        else:
            self._reset_measure_mode()

    def _reset_measure_mode(self) -> None:
        """Réinitialise le mode mesure."""
        self.measure_mode = False
        self.measure_points = []
        self.measure_points_3d = []
        self.measure_distance = None
        self.log_panel.add_info("Mode mesure désactivé")

    def _handle_measure_click(self, pos: tuple, point_index: int = 0) -> None:
        """
        Gère un clic en mode mesure de distance.

        Args:
            pos: Position du clic (x, y) dans la fenêtre Pygame
            point_index: 0 pour point 1 (clic gauche), 1 pour point 2 (clic droit)
        """
        if not self.measure_mode and not self.measure_points:
            return

        # Vérifier si le clic est dans la zone vidéo
        video_rect = self.video_display.rect
        if not video_rect.collidepoint(pos):
            return

        # Convertir les coordonnées Pygame en coordonnées image
        # La vidéo est centrée dans video_display
        display_frame = self.frozen_frame if self.frozen_frame is not None else self.current_frame
        if display_frame is None:
            return

        frame_h, frame_w = display_frame.shape[:2]

        # Calculer le scale et l'offset de l'image dans la zone vidéo
        scale = min(video_rect.width / frame_w, video_rect.height / frame_h)
        display_w, display_h = int(frame_w * scale), int(frame_h * scale)
        offset_x = video_rect.x + (video_rect.width - display_w) // 2
        offset_y = video_rect.y + (video_rect.height - display_h) // 2

        # Vérifier si le clic est dans l'image affichée
        click_x, click_y = pos
        if not (offset_x <= click_x <= offset_x + display_w and
                offset_y <= click_y <= offset_y + display_h):
            return

        # Convertir en coordonnées image originale
        img_x = int((click_x - offset_x) / scale)
        img_y = int((click_y - offset_y) / scale)

        # Projeter en 3D
        point_3d = self._project_to_3d(img_x, img_y)

        if point_3d is None:
            self.log_panel.add_warning("Depth invalide à ce point")
            return

        # Initialiser les listes si nécessaire
        while len(self.measure_points) < 2:
            self.measure_points.append(None)
            self.measure_points_3d.append(None)

        # Mettre à jour le point spécifié
        self.measure_points[point_index] = (img_x, img_y)
        self.measure_points_3d[point_index] = point_3d

        point_num = point_index + 1
        depth_m = point_3d[2]
        self.log_panel.add_info(f"Point {point_num}: ({img_x}, {img_y}) z={depth_m:.2f}m")

        # Si on a les 2 points, calculer la distance
        if self.measure_points[0] is not None and self.measure_points[1] is not None:
            self._calculate_distance()

    def _project_to_3d(self, x: int, y: int):
        """
        Projette un point 2D en 3D en utilisant les données depth figées.

        Args:
            x, y: Coordonnées pixel dans l'image

        Returns:
            Point 3D [x, y, z] en mètres ou None si invalide
        """
        if self.frozen_depth_data is None or self.depth_intrinsics is None:
            logger.debug("Pas de données depth figées ou intrinsics")
            return None

        try:
            import pyrealsense2 as rs

            # Vérifier les bornes de l'image
            height, width = self.frozen_depth_data.shape[:2]

            if x < 0 or x >= width or y < 0 or y >= height:
                logger.debug(f"Coordonnées hors bornes: ({x}, {y}) vs ({width}x{height})")
                return None

            # Récupérer la valeur depth brute (uint16)
            depth_raw = self.frozen_depth_data[y, x]

            # Convertir en mètres (depth scale RealSense = 0.001 par défaut)
            depth_scale = 0.001  # 1mm par unité
            depth = float(depth_raw) * depth_scale

            if depth <= 0 or depth > 10.0:
                logger.debug(f"Depth invalide: {depth}m (raw={depth_raw}) à ({x}, {y})")
                return None

            # Déprojeter en 3D
            point_3d = rs.rs2_deproject_pixel_to_point(
                self.depth_intrinsics, [float(x), float(y)], depth
            )
            return np.array(point_3d)

        except Exception as e:
            logger.debug(f"Erreur projection 3D: {e}")
            return None

    def _calculate_distance(self) -> None:
        """Calcule la distance 3D entre les deux points mesurés."""
        if len(self.measure_points_3d) < 2:
            return

        p1 = self.measure_points_3d[0]
        p2 = self.measure_points_3d[1]

        if p1 is None or p2 is None:
            return

        # Distance euclidienne 3D
        distance = np.linalg.norm(p2 - p1)
        self.measure_distance = distance

        # Convertir en cm pour l'affichage
        distance_cm = distance * 100

        self.log_panel.add_success(f"Taille: {distance_cm:.1f} cm")
        logger.info(f"Distance mesurée: {distance_cm:.1f} cm (3D: {p1} -> {p2})")

    def _on_exit(self) -> None:
        """Callback pour quitter."""
        self.log_panel.add_info("Fermeture de l'application...")
        self.running = False

    def _start_capture(self) -> None:
        """Démarre la capture vidéo."""
        if self.capturing:
            return

        self.stop_capture.clear()
        self.capturing = True

        if self.mode == "inline":
            self.capture_thread = threading.Thread(
                target=self._capture_inline,
                daemon=True
            )
            self.log_panel.add_info("Démarrage capture inline...")
        else:
            self.capture_thread = threading.Thread(
                target=self._capture_offline,
                daemon=True
            )
            self.log_panel.add_info("Démarrage capture offline...")

        self.capture_thread.start()

    def _stop_capture(self) -> None:
        """Arrête la capture vidéo."""
        self.stop_capture.set()
        self.capturing = False
        # Réinitialiser l'état de pause
        self.paused = False
        self.pause_event.clear()
        self.frozen_frame = None
        self.btn_pause.set_state(False)
        # Réinitialiser le graphique
        self.score_graph.clear()
        self.current_reba_score = None
        self.current_reba_score_2d = None
        self.log_panel.add_info("Capture arrêtée")

    def _capture_inline(self) -> None:
        """Thread de capture en mode inline (caméra RealSense live)."""
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            rs_config = rs.config()

            # Utiliser les paramètres de config.yaml
            rs_cfg = self.config.realsense
            rs_config.enable_stream(
                rs.stream.color,
                rs_cfg.color_width, rs_cfg.color_height,
                rs.format.bgr8, rs_cfg.color_fps
            )
            rs_config.enable_stream(
                rs.stream.depth,
                rs_cfg.depth_width, rs_cfg.depth_height,
                rs.format.z16, rs_cfg.depth_fps
            )

            profile = pipeline.start(rs_config)
            logger.info("Caméra RealSense connectée")
            self.log_panel.add_success("Caméra RealSense connectée")

            # Aligner depth sur color
            align = rs.align(rs.stream.color)

            # Récupérer les intrinsics de la caméra COLOR (car depth aligné sur color)
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.depth_intrinsics = color_profile.get_intrinsics()
            logger.info(f"Intrinsics 3D (aligned): {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")

            while not self.stop_capture.is_set():
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()  # Frame depth alignée

                if color_frame and depth_frame:
                    frame = np.asanyarray(color_frame.get_data())
                    # En mode pause inline, on continue de capturer mais on n'actualise pas l'affichage
                    if not self.pause_event.is_set():
                        self._process_frame(frame, depth_frame)

            pipeline.stop()
            self.depth_intrinsics = None
            logger.info("Caméra RealSense déconnectée")
            self.log_panel.add_info("Caméra RealSense déconnectée")

        except ImportError:
            logger.error("pyrealsense2 non installé")
            self.log_panel.add_error("pyrealsense2 non installé")
        except Exception as e:
            logger.exception(f"Erreur caméra: {e}")
            self.log_panel.add_error(f"Erreur caméra: {str(e)[:30]}")

    def _capture_offline(self) -> None:
        """Thread de capture en mode offline (fichier .bag)."""
        bag_path = self.bag_directory / self.default_bag_name

        if not bag_path.exists():
            logger.error(f"Fichier non trouvé: {bag_path}")
            self.log_panel.add_error(f"Fichier non trouvé: {bag_path}")
            self.capturing = False
            self.btn_capture.set_state(False)
            return

        try:
            import pyrealsense2 as rs
            import cv2
            import time

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device_from_file(str(bag_path))

            profile = pipeline.start(config)
            device = profile.get_device()
            playback = device.as_playback()
            playback.set_real_time(self.config.realsense.realtime_playback)

            # Aligner depth sur color
            align = rs.align(rs.stream.color)
            logger.info(f"Fichier .bag chargé: {bag_path}")
            self.log_panel.add_success(f"Fichier .bag chargé")

            # Récupérer les intrinsics depuis le stream COLOR (car depth aligné sur color)
            try:
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                self.depth_intrinsics = color_profile.get_intrinsics()
                logger.info(f"Intrinsics 3D (aligned): {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
                self.log_panel.add_success("Mode 3D activé")
            except Exception as e:
                logger.warning(f"Impossible d'obtenir les intrinsics: {e}")
                self.depth_intrinsics = None
                self.log_panel.add_warning("Mode 2D (pas de depth)")

            was_paused = False
            frame_count = 0

            while not self.stop_capture.is_set():
                # Gérer la pause pour le mode offline
                if self.pause_event.is_set():
                    if not was_paused:
                        playback.pause()
                        was_paused = True
                    time.sleep(0.05)
                    continue
                else:
                    if was_paused:
                        playback.resume()
                        was_paused = False

                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    aligned = align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()

                    if color_frame:
                        frame = np.asanyarray(color_frame.get_data())
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # Passer le depth_frame pour calcul 3D
                        self._process_frame(frame, depth_frame)
                        frame_count += 1

                        # Log périodique
                        if frame_count % 100 == 0:
                            logger.debug(f"Frame {frame_count} traité")

                except RuntimeError:
                    # Fin du fichier
                    logger.info("Fin du fichier .bag")
                    self.log_panel.add_info("Fin du fichier .bag")
                    break

            pipeline.stop()
            self.depth_intrinsics = None

        except ImportError:
            logger.error("pyrealsense2 non installé")
            self.log_panel.add_error("pyrealsense2 non installé")
        except Exception as e:
            logger.exception(f"Erreur lecture: {e}")
            self.log_panel.add_error(f"Erreur lecture: {str(e)[:30]}")

        self.capturing = False
        self.btn_capture.set_state(False)

    def _init_openpose(self) -> bool:
        """
        Initialise OpenPose detector si mode local.

        Returns:
            True si initialisé avec succès, False sinon
        """
        if self._openpose_initialized:
            return self.openpose_detector is not None

        self._openpose_initialized = True

        if OPENPOSE_MODE != "local":
            logger.info(f"Mode OpenPose: {OPENPOSE_MODE} (pas d'init local)")
            return False

        try:
            from reba_3d.capture.realsense_capture import OpenPoseDetector
            from reba_3d.config.settings import OPENPOSE_PATH

            logger.info(f"Initialisation OpenPose: {OPENPOSE_PATH}")
            self.log_panel.add_info("Init OpenPose...")

            self.openpose_detector = OpenPoseDetector(OPENPOSE_PATH)

            logger.info("OpenPose initialisé avec succès")
            self.log_panel.add_success("OpenPose OK")
            return True

        except Exception as e:
            logger.error(f"Erreur init OpenPose: {e}")
            self.log_panel.add_error(f"OpenPose erreur: {str(e)[:25]}")
            self.openpose_detector = None
            return False

    def _process_frame(self, frame, depth_frame=None) -> None:
        """
        Traite un frame (détection OpenPose, calcul REBA 3D).

        Args:
            frame: Image BGR numpy array
            depth_frame: RealSense depth frame aligné (optionnel)
        """
        import cv2

        # Stocker le depth frame et ses données pour les mesures de distance
        self.current_depth_frame = depth_frame
        if depth_frame is not None:
            try:
                # Stocker une copie des données depth à chaque frame
                # pour que les données soient disponibles au moment de la pause
                self._last_depth_data = np.asanyarray(depth_frame.get_data()).copy()
            except Exception:
                self._last_depth_data = None

        output_frame = frame.copy()
        keypoints = None

        # Détection OpenPose si disponible
        if OPENPOSE_MODE == "local":
            # Initialiser OpenPose au premier appel
            if not self._openpose_initialized:
                self._init_openpose()

            # Exécuter la détection
            if self.openpose_detector is not None:
                try:
                    output_frame, keypoints = self.openpose_detector.detect(frame)
                except Exception as e:
                    logger.warning(f"Erreur détection: {e}")

        # Traiter les keypoints si disponibles
        if keypoints is not None:
            try:
                if len(keypoints) > 0 and len(keypoints.shape) >= 2:
                    person_keypoints = keypoints[0]

                    # Calculer les angles (3D si depth disponible, sinon 2D)
                    raw_angles = None

                    if self.use_3d and depth_frame is not None and self.depth_intrinsics is not None:
                        # Calcul 3D avec projection de profondeur
                        raw_angles = calculate_angles_from_keypoints_3d(
                            person_keypoints,
                            depth_frame,
                            self.depth_intrinsics
                        )
                        # Fallback 2D si pas assez d'angles 3D
                        if len(raw_angles) < 4:
                            raw_angles = calculate_angles_from_keypoints_2d(person_keypoints)
                    else:
                        # Calcul 2D classique
                        raw_angles = calculate_angles_from_keypoints_2d(person_keypoints)

                    if raw_angles:
                        # Mode calibration: collecter les angles
                        if self.calibration_active:
                            self.calibration_angles.append(raw_angles)

                        # Mode normal: appliquer calibration et calculer REBA
                        else:
                            # Appliquer la calibration
                            calibrated_angles = self.calibration_manager.apply_all(raw_angles)
                            self.current_angles = calibrated_angles

                            # Calculer le score REBA principal (3D si dispo)
                            self.current_reba_score = self.reba_scorer.update(calibrated_angles)

                            # Mode comparaison: calculer aussi le score 2D
                            if self.show_comparison and depth_frame is not None:
                                raw_angles_2d = calculate_angles_from_keypoints_2d(person_keypoints)
                                if raw_angles_2d:
                                    calibrated_2d = self.calibration_manager.apply_all(raw_angles_2d)
                                    self.current_reba_score_2d = self.reba_scorer_2d.update(calibrated_2d)

                            # Mettre à jour le graphique des scores
                            score_3d = self.current_reba_score.final_score if self.current_reba_score else None
                            score_2d = self.current_reba_score_2d.final_score if self.current_reba_score_2d else None
                            self.score_graph.add_scores(score_3d, score_2d)

                            # Dessiner le score sur la frame si activé
                            if self.show_scores and self.current_reba_score:
                                output_frame = self._draw_reba_overlay(output_frame)

            except Exception as e:
                logger.debug(f"Erreur traitement keypoints: {e}")

        # Afficher le frame (avec ou sans overlay)
        with self.frame_lock:
            self.current_frame = output_frame.copy()

    def _draw_reba_overlay(self, frame) -> np.ndarray:
        """
        Dessine l'overlay REBA sur le frame.

        Args:
            frame: Image BGR

        Returns:
            Frame avec overlay
        """
        import cv2

        if self.current_reba_score is None:
            return frame

        score = self.current_reba_score
        h, w = frame.shape[:2]

        # Mode comparaison: affichage côte à côte 2D/3D
        if self.show_comparison and self.current_reba_score_2d is not None:
            return self._draw_comparison_overlay(frame)

        # Couleur de fond basée sur le risque
        color = score.risk_color

        # Rectangle de fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 145), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Indicateur 2D/3D
        mode_3d = self.use_3d and self.depth_intrinsics is not None
        mode_text = "3D" if mode_3d else "2D"
        mode_color = (0, 255, 0) if mode_3d else (100, 100, 255)
        cv2.putText(
            frame, f"[{mode_text}]",
            (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1
        )

        # Texte du score REBA
        cv2.putText(
            frame, f"REBA: {score.final_score}",
            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
        )

        # Niveau de risque
        cv2.putText(
            frame, f"{score.risk_level.upper()}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        # Détail des scores (optionnel)
        detail = f"A:{score.score_a} B:{score.score_b}"
        cv2.putText(
            frame, detail,
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        # Barre de couleur indicateur de risque
        bar_width = int((score.final_score / 12) * 180)
        cv2.rectangle(frame, (10, 125), (10 + bar_width, 135), color, -1)
        cv2.rectangle(frame, (10, 125), (190, 135), (100, 100, 100), 1)

        return frame

    def _draw_comparison_overlay(self, frame) -> np.ndarray:
        """
        Dessine l'overlay comparaison 2D vs 3D sur le frame.

        Args:
            frame: Image BGR

        Returns:
            Frame avec overlay comparaison
        """
        import cv2

        score_3d = self.current_reba_score
        score_2d = self.current_reba_score_2d

        # Rectangle de fond semi-transparent plus large
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Titre
        cv2.putText(
            frame, "COMPARAISON 2D vs 3D",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        # Score 3D (colonne gauche)
        cv2.putText(
            frame, "3D",
            (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"REBA: {score_3d.final_score}",
            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_3d.risk_color, 2
        )
        cv2.putText(
            frame, f"{score_3d.risk_level[:10]}",
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_3d.risk_color, 1
        )

        # Barre 3D
        bar_3d = int((score_3d.final_score / 12) * 130)
        cv2.rectangle(frame, (20, 120), (20 + bar_3d, 130), score_3d.risk_color, -1)
        cv2.rectangle(frame, (20, 120), (150, 130), (100, 100, 100), 1)

        # Score 2D (colonne droite)
        cv2.putText(
            frame, "2D",
            (210, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2
        )
        cv2.putText(
            frame, f"REBA: {score_2d.final_score}",
            (170, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_2d.risk_color, 2
        )
        cv2.putText(
            frame, f"{score_2d.risk_level[:10]}",
            (170, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_2d.risk_color, 1
        )

        # Barre 2D
        bar_2d = int((score_2d.final_score / 12) * 130)
        cv2.rectangle(frame, (170, 120), (170 + bar_2d, 130), score_2d.risk_color, -1)
        cv2.rectangle(frame, (170, 120), (300, 130), (100, 100, 100), 1)

        # Différence
        diff = score_3d.final_score - score_2d.final_score
        diff_text = f"Diff: {diff:+d}"
        diff_color = (0, 255, 255) if diff != 0 else (200, 200, 200)
        cv2.putText(
            frame, diff_text,
            (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, diff_color, 1
        )

        return frame

    def _handle_events(self) -> None:
        """Gère les événements Pygame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Toggle capture avec espace
                    self.btn_capture.toggle()
                    self._on_capture_toggle(self.btn_capture.state)
                elif event.key == pygame.K_p:
                    # Toggle pause avec P
                    if self.capturing:
                        self.btn_pause.toggle()
                        self._on_pause_toggle(self.btn_pause.state)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Clic gauche = point 1
                    self._handle_measure_click(event.pos, point_index=0)
                elif event.button == 3:  # Clic droit = point 2
                    self._handle_measure_click(event.pos, point_index=1)

            # Propager aux composants
            self.mode_selector.handle_event(event)
            self.btn_capture.handle_event(event)
            self.btn_pause.handle_event(event)
            self.btn_calibration.handle_event(event)
            self.btn_scores.handle_event(event)
            self.btn_comparison.handle_event(event)
            self.btn_measure.handle_event(event)
            self.score_load.handle_event(event)
            self.score_coupling.handle_event(event)
            self.score_activity.handle_event(event)
            self.btn_exit.handle_event(event)

    def _update(self) -> None:
        """Met à jour l'état de l'application."""
        # Mettre à jour l'affichage vidéo
        with self.frame_lock:
            # Déterminer quel frame afficher
            display_frame = None
            if self.paused and self.mode == "inline" and self.frozen_frame is not None:
                display_frame = self.frozen_frame.copy()
            elif self.current_frame is not None:
                display_frame = self.current_frame.copy()

            # Ajouter l'overlay de mesure si nécessaire
            if display_frame is not None and (self.measure_mode or self.measure_points):
                display_frame = self._draw_measure_overlay(display_frame)

            if display_frame is not None:
                self.video_display.set_frame(display_frame)

    def _draw_measure_overlay(self, frame) -> np.ndarray:
        """
        Dessine l'overlay de mesure de distance sur le frame.

        Args:
            frame: Image BGR

        Returns:
            Frame avec overlay de mesure
        """
        import cv2

        # Dessiner les points cliqués
        for i, point in enumerate(self.measure_points):
            if point is None:
                continue
            px, py = point
            # Cercle du point
            color = (0, 255, 255) if i == 0 else (255, 0, 255)  # Jaune puis magenta
            cv2.circle(frame, (px, py), 8, color, -1)
            cv2.circle(frame, (px, py), 10, (255, 255, 255), 2)

            # Numéro du point
            label = "1 (G)" if i == 0 else "2 (D)"
            cv2.putText(
                frame, label,
                (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        # Dessiner la ligne entre les points
        if (len(self.measure_points) >= 2 and
            self.measure_points[0] is not None and
            self.measure_points[1] is not None):
            p1 = self.measure_points[0]
            p2 = self.measure_points[1]
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

            # Afficher la distance au milieu de la ligne
            if self.measure_distance is not None:
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2

                distance_cm = self.measure_distance * 100
                text = f"{distance_cm:.1f} cm"

                # Fond pour le texte
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(
                    frame,
                    (mid_x - text_w // 2 - 5, mid_y - text_h - 10),
                    (mid_x + text_w // 2 + 5, mid_y + 5),
                    (0, 0, 0), -1
                )
                cv2.putText(
                    frame, text,
                    (mid_x - text_w // 2, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

        # Instructions si en mode mesure
        if self.measure_mode:
            text = "Gauche=Pt1  Droit=Pt2"
            cv2.putText(
                frame, text,
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

        return frame

    def _draw(self) -> None:
        """Dessine l'interface."""
        # Fond
        self.screen.fill(DARK_GRAY)

        # Hauteur de la zone principale (sans graphique)
        main_height = self.height - GRAPH_HEIGHT

        # Panneau gauche (fond)
        pygame.draw.rect(
            self.screen,
            (40, 40, 50),
            pygame.Rect(0, 0, self.left_panel_width, main_height)
        )

        # Titre panneau gauche
        title = self.title_font.render("Contrôles", True, WHITE)
        self.screen.blit(title, (10, 10))

        # Mode label
        mode_label = pygame.font.Font(None, 18).render("Mode:", True, GRAY)
        self.screen.blit(mode_label, (10, 45))

        # Composants
        self.mode_selector.draw(self.screen)
        self.btn_capture.draw(self.screen)
        self.btn_pause.draw(self.screen)
        self.btn_calibration.draw(self.screen)
        self.btn_scores.draw(self.screen)
        self.btn_comparison.draw(self.screen)
        self.btn_measure.draw(self.screen)

        # Label section paramètres
        params_font = pygame.font.Font(None, 18)
        params_label = params_font.render("Paramètres REBA:", True, GRAY)
        self.screen.blit(params_label, (10, self.params_label_y))

        # Boutons de score
        self.score_load.draw(self.screen)
        self.score_coupling.draw(self.screen)
        self.score_activity.draw(self.screen)

        self.btn_exit.draw(self.screen)

        # Zone vidéo
        self.video_display.draw(self.screen)

        # Panneau logs
        self.log_panel.draw(self.screen)

        # Graphique des scores (en bas)
        self.score_graph.draw(self.screen)

        # Rafraîchir l'écran
        pygame.display.flip()

    def run(self) -> None:
        """Boucle principale de l'application."""
        logger.info("Démarrage de la boucle principale")
        self.running = True

        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(self.fps)

        # Nettoyage
        logger.debug("Arrêt de l'application...")
        self._stop_capture()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        pygame.quit()
        logger.info("Application fermée")


def main():
    """Point d'entrée pour l'interface graphique."""
    from reba_3d.utils.logger import setup_logging
    setup_logging()

    app = REBAApp()
    app.run()


if __name__ == "__main__":
    main()
