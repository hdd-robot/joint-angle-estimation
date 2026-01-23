"""
Main REBA GUI Application.

Interface graphique Pygame avec 3 colonnes:
- Gauche: Boutons de contrôle
- Centre: Affichage vidéo
- Droite: Panneau de logs
"""

import threading
from pathlib import Path
from typing import Optional

import pygame
import numpy as np

from reba_3d.gui.components import (
    Button, ToggleButton, RadioButtonGroup, LogPanel, VideoDisplay,
    WHITE, BLACK, GRAY, DARK_GRAY, GREEN, RED, BLUE, ORANGE
)
from reba_3d.config import get_config
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
        self.width = width if width is not None else self.config.gui.width
        self.height = height if height is not None else self.config.gui.height
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

        # --- Spacer ---
        y = self.height - button_height - padding - 50

        # --- Bouton Exit ---
        self.btn_exit = Button(
            x, y, button_width, button_height,
            "Exit",
            callback=self._on_exit,
            color=RED
        )

        # --- Zone vidéo (colonne centrale) ---
        video_x = self.left_panel_width
        video_width = self.width - self.left_panel_width - self.right_panel_width
        self.video_display = VideoDisplay(
            video_x, 0, video_width, self.height
        )

        # --- Panneau de logs (colonne droite) ---
        log_x = self.width - self.right_panel_width
        self.log_panel = LogPanel(
            log_x, 0, self.right_panel_width, self.height,
            max_lines=self.config.logging.max_lines
        )

        # Message de bienvenue
        self.log_panel.add_info("Application REBA 3D démarrée")
        self.log_panel.add_info(f"Mode: {self.mode}")
        self.log_panel.add_info("Calibration: Non effectuée")

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
        self.calibration_active = state
        if state:
            self.log_panel.add_info("Calibration démarrée...")
            self.log_panel.add_warning("Restez en position neutre")
        else:
            self.log_panel.add_success("Calibration terminée")

    def _on_scores_toggle(self, state: bool) -> None:
        """Callback pour afficher/cacher les scores."""
        self.show_scores = state
        self.video_display.overlay_scores = state
        if state:
            self.log_panel.add_info("Affichage des scores activé")
        else:
            self.log_panel.add_info("Affichage des scores désactivé")

    def _on_pause_toggle(self, state: bool) -> None:
        """Callback pour pause/reprendre la vidéo."""
        self.paused = state
        if state:
            self.pause_event.set()
            # En mode inline, figer le frame actuel
            if self.mode == "inline":
                with self.frame_lock:
                    if self.current_frame is not None:
                        self.frozen_frame = self.current_frame.copy()
            self.log_panel.add_info("Pause activée")
            logger.debug("Pause activée")
        else:
            self.pause_event.clear()
            self.frozen_frame = None
            self.log_panel.add_info("Lecture reprise")
            logger.debug("Lecture reprise")

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

            pipeline.start(rs_config)
            logger.info("Caméra RealSense connectée")
            self.log_panel.add_success("Caméra RealSense connectée")

            align = rs.align(rs.stream.color)

            while not self.stop_capture.is_set():
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()

                if color_frame:
                    frame = np.asanyarray(color_frame.get_data())
                    # En mode pause inline, on continue de capturer mais on n'actualise pas l'affichage
                    if not self.pause_event.is_set():
                        self._process_frame(frame)

            pipeline.stop()
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

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device_from_file(str(bag_path))

            profile = pipeline.start(config)
            device = profile.get_device()
            playback = device.as_playback()
            playback.set_real_time(self.config.realsense.realtime_playback)

            align = rs.align(rs.stream.color)
            logger.info(f"Fichier .bag chargé: {bag_path}")
            self.log_panel.add_success(f"Fichier .bag chargé")

            was_paused = False

            while not self.stop_capture.is_set():
                # Gérer la pause pour le mode offline
                if self.pause_event.is_set():
                    if not was_paused:
                        playback.pause()
                        was_paused = True
                    # Attendre un peu pour ne pas surcharger le CPU
                    import time
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

                    if color_frame:
                        frame = np.asanyarray(color_frame.get_data())
                        import cv2
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        self._process_frame(frame)

                except RuntimeError:
                    # Fin du fichier
                    logger.info("Fin du fichier .bag")
                    self.log_panel.add_info("Fin du fichier .bag")
                    break

            pipeline.stop()

        except ImportError:
            logger.error("pyrealsense2 non installé")
            self.log_panel.add_error("pyrealsense2 non installé")
        except Exception as e:
            logger.exception(f"Erreur lecture: {e}")
            self.log_panel.add_error(f"Erreur lecture: {str(e)[:30]}")

        self.capturing = False
        self.btn_capture.set_state(False)

    def _process_frame(self, frame) -> None:
        """
        Traite un frame (détection OpenPose, calcul REBA).

        Args:
            frame: Image BGR numpy array
        """
        # Pour l'instant, juste afficher le frame
        with self.frame_lock:
            self.current_frame = frame.copy()

        # TODO: Intégrer OpenPose et calcul REBA
        # Si calibration active, collecter les données
        if self.calibration_active:
            pass  # Collecter données de calibration

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

            # Propager aux composants
            self.mode_selector.handle_event(event)
            self.btn_capture.handle_event(event)
            self.btn_pause.handle_event(event)
            self.btn_calibration.handle_event(event)
            self.btn_scores.handle_event(event)
            self.btn_exit.handle_event(event)

    def _update(self) -> None:
        """Met à jour l'état de l'application."""
        # Mettre à jour l'affichage vidéo
        with self.frame_lock:
            # En mode pause inline, afficher le frame figé
            if self.paused and self.mode == "inline" and self.frozen_frame is not None:
                self.video_display.set_frame(self.frozen_frame)
            elif self.current_frame is not None:
                self.video_display.set_frame(self.current_frame)

    def _draw(self) -> None:
        """Dessine l'interface."""
        # Fond
        self.screen.fill(DARK_GRAY)

        # Panneau gauche (fond)
        pygame.draw.rect(
            self.screen,
            (40, 40, 50),
            pygame.Rect(0, 0, self.left_panel_width, self.height)
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
        self.btn_exit.draw(self.screen)

        # Zone vidéo
        self.video_display.draw(self.screen)

        # Panneau logs
        self.log_panel.draw(self.screen)

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
