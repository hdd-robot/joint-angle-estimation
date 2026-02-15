"""
Main REBA GUI Application.

Pygame graphical interface with 3 columns:
- Left: Control buttons
- Center: Video display
- Right: Log panel
"""

import threading
import time
import json
import os
from pathlib import Path
from typing import Optional, List, Dict

import pygame
import numpy as np

from reba_3d.gui.components import (
    Button, ToggleButton, RadioButtonGroup, LogPanel, VideoDisplay, ScoreGraph,
    ScoreButtonGroup, RiskTimeline, FileSelector,
    WHITE, BLACK, GRAY, DARK_GRAY, GREEN, RED, BLUE, ORANGE
)

# Height of the graph at the bottom of the window
GRAPH_HEIGHT = 140

# ---- USER-DEFINED FPS for angle logging timestamps ----
# Set this to match your camera / video source frame rate.
LOG_ANGLES_FPS =15.0
from reba_3d.config import get_config
from reba_3d.config.settings import OPENPOSE_MODE
from reba_3d.config.calibration_store import (
    get_calibration_manager, save_calibration, load_calibration
)
from reba_3d.core.angles import (
    calculate_angles_from_keypoints_2d,
    calculate_angles_from_keypoints_3d,
    compute_calibration_offsets,
    calculate_nautical_angles_3d,
    calculate_nautical_angles_2d,
    compute_calibration_offsets_nested,
    compute_calibration_offsets_robust
)
from reba_3d.reba.realtime_scorer import RealtimeREBAScorer, REBAScore
from reba_3d.utils.logger import get_logger
from reba_3d.io.angle_logger import AngleLogger
from reba_3d.io.video_io import VideoWriter

# Logger for this module
logger = get_logger("gui.app")


class REBAApp:
    """
    Main REBA application with Pygame interface.

    Attributes:
        width: Window width
        height: Window height
        running: Running state
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        bag_directory: Optional[str] = None,
        default_bag_name: Optional[str] = None
    ):
        """
        Initialize the REBA application.

        Args:
            width: Window width (if None, uses config.yaml)
            height: Window height (if None, uses config.yaml)
            bag_directory: Directory for .bag files (if None, uses config.yaml)
            default_bag_name: Default .bag filename (if None, uses config.yaml)
        """
        # Load YAML configuration
        self.config = get_config()

        # Use values passed as arguments or from config
        # Add GRAPH_HEIGHT for the graph at the bottom
        self.width = width if width is not None else self.config.gui.width
        self.height = (height if height is not None else self.config.gui.height) + GRAPH_HEIGHT
        self.bag_directory = Path(bag_directory if bag_directory is not None else self.config.paths.bag_directory)
        self.default_bag_name = default_bag_name if default_bag_name is not None else self.config.paths.default_bag_file

        # Column dimensions from config
        self.left_panel_width = self.config.gui.left_panel_width
        self.right_panel_width = self.config.gui.right_panel_width
        self.fps = self.config.gui.fps

        # Application state
        self.running = False
        self.mode = "offline"  # "inline" or "offline"
        self.calibration_active = False
        self.show_scores = True
        self.capturing = False
        self.paused = False

        # Video components
        self.current_frame = None
        self.frozen_frame = None  # Frozen frame for inline pause mode
        self.frame_lock = threading.Lock()

        # Capture thread
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_capture = threading.Event()
        self.pause_event = threading.Event()  # Event to handle pause

        # OpenPose detector (initialized on first use)
        self.openpose_detector = None
        self._openpose_initialized = False

        # Calibration data
        self.calibration_start_time = None
        self.calibration_angles: List[Dict[str, float]] = []  # List of collected angles (3D or main)
        self.calibration_angles_2d: List[Dict[str, float]] = []  # List of 2D angles (for comparison mode)
        self.calibration_duration = self.config.calibration.duration  # Duration in seconds
        self.calibration_mode_3d = None  # Stores if calibration is in 3D or 2D mode
        self.calibration_mode_dual = False  # True if simultaneous 2D+3D calibration in comparison mode
        self.current_mode_3d = None  # Track current mode to detect changes
        self.calibration_manager = get_calibration_manager()

        # Separate calibration offsets for 2D comparison mode
        # Stored directly in a dict because CalibrationManager is a singleton
        from reba_3d.config.calibration_store import get_calibration_path, DEFAULT_NAUTICAL_OFFSETS
        calib_2d_path = get_calibration_path(mode_3d=False)
        if calib_2d_path.exists():
            self.calibration_offsets_2d = load_calibration(calib_2d_path)
        else:
            self.calibration_offsets_2d = DEFAULT_NAUTICAL_OFFSETS.copy()

        # REBA scoring
        self.reba_scorer = RealtimeREBAScorer(buffer_size=10)
        # Initialize with config values
        self.reba_scorer.set_load_score(self.config.reba.load_score)
        self.reba_scorer.set_coupling_score(self.config.reba.coupling_score)
        self.reba_scorer.set_activity_score(self.config.reba.activity_score)
        self.current_reba_score: Optional[REBAScore] = None
        self.current_angles: Dict[str, float] = {}

        # Depth/3D data
        self.depth_intrinsics = None  # Camera intrinsics for 3D projection
        self.use_3d = True  # Use 3D angles when available

        # Comparison mode (shows both 2D and 3D scores)
        self.show_comparison = False
        self.reba_scorer_2d = RealtimeREBAScorer(buffer_size=10)  # Separate scorer for 2D
        # Initialize with the same config values
        self.reba_scorer_2d.set_load_score(self.config.reba.load_score)
        self.reba_scorer_2d.set_coupling_score(self.config.reba.coupling_score)
        self.reba_scorer_2d.set_activity_score(self.config.reba.activity_score)
        self.current_reba_score_2d: Optional[REBAScore] = None

        # Risk timeline data (loaded from JSON)
        self.risk_data: Optional[Dict] = None
        self.current_frame_number: int = 0

        # Keypoints 3D accumulator (for offline JSON export)
        self.keypoints_3d_data: List[Dict] = []

        # Angle logger for offline mode
        self.angle_logger = AngleLogger(mode="3d")
        self.angle_logging_enabled = False
        self.video_writer: Optional[VideoWriter] = None

        # Pygame initialization
        pygame.init()
        pygame.display.set_caption(self.config.gui.title)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        logger.debug(f"Window created: {self.width}x{self.height}")

        # UI components initialization
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize the interface components."""
        padding = 10
        button_width = self.left_panel_width - 2 * padding
        button_height = 40

        # Starting position for buttons
        x = padding
        y = padding

        # Left panel title
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

        # --- Start/Stop Capture Button ---
        self.btn_capture = ToggleButton(
            x, y, button_width, button_height,
            text_on="Stop Capture",
            text_off="Start Capture",
            callback=self._on_capture_toggle,
            color_on=RED,
            color_off=GREEN
        )
        y += button_height + padding

        # --- Pause Button ---
        self.btn_pause = ToggleButton(
            x, y, button_width, button_height,
            text_on="Resume",
            text_off="Pause",
            callback=self._on_pause_toggle,
            color_on=GREEN,
            color_off=ORANGE
        )
        y += button_height + padding

        # --- Calibration Button ---
        self.btn_calibration = ToggleButton(
            x, y, button_width, button_height,
            text_on="Stop Calibration",
            text_off="Start Calibration",
            callback=self._on_calibration_toggle,
            color_on=RED,
            color_off=BLUE
        )
        y += button_height + padding

        # --- Show/Hide Scores Button ---
        self.btn_scores = ToggleButton(
            x, y, button_width, button_height,
            text_on="Hide Scores",
            text_off="Show Scores",
            callback=self._on_scores_toggle,
            color_on=GRAY,
            color_off=GREEN,
            initial_state=True
        )
        y += button_height + padding

        # --- 2D/3D Comparison Button ---
        self.btn_comparison = ToggleButton(
            x, y, button_width, button_height,
            text_on="Normal Mode",
            text_off="Compare 2D/3D",
            callback=self._on_comparison_toggle,
            color_on=ORANGE,
            color_off=BLUE
        )
        y += button_height + padding

        # --- Analyze Keypoints Button ---
        self.btn_load_keypoints = Button(
            x, y, button_width, button_height,
            text="Analyze Keypoints",
            callback=self._analyze_keypoints_json,
            color=(100, 0, 200),
            hover_color=(150, 50, 255)
        )
        y += button_height + padding

        # --- Log Angles Button (disabled) ---
        self.btn_log_angles = ToggleButton(
            x, y, button_width, button_height,
            text_on="Stop Log Angles",
            text_off="Log Angles",
            callback=self._on_log_angles_toggle,
            color_on=RED,
            color_off=(0, 128, 128)  # Teal
        )
        y += button_height + padding + 10
        y += 10  # Reduced spacing without the button

        # --- REBA Parameters Section ---
        section_font = pygame.font.Font(None, 18)
        self.params_label_y = y
        y += 20

        # Load values from config
        reba_cfg = self.config.reba

        # --- Load Score ---
        self.score_load = ScoreButtonGroup(
            x, y,
            label="Load:",
            values=[0, 1, 2, 3],
            callback=self._on_load_change,
            initial_value=reba_cfg.load_score
        )
        y += 35

        # --- Coupling Score ---
        self.score_coupling = ScoreButtonGroup(
            x, y,
            label="Coupling:",
            values=[0, 1, 2, 3],
            callback=self._on_coupling_change,
            initial_value=reba_cfg.coupling_score
        )
        y += 35

        # --- Activity Score ---
        self.score_activity = ScoreButtonGroup(
            x, y,
            label="Activity:",
            values=[0, 1, 2, 3],
            callback=self._on_activity_change,
            initial_value=reba_cfg.activity_score
        )
        y += 35

        # --- Spacer ---
        # Position of Exit button above the graph
        y = self.height - GRAPH_HEIGHT - button_height - padding - 50

        # --- Exit Button ---
        self.btn_exit = Button(
            x, y, button_width, button_height,
            "Exit",
            callback=self._on_exit,
            color=RED
        )

        # --- Main area height (without the graph) ---
        main_height = self.height - GRAPH_HEIGHT

        # --- Video area (center column) ---
        video_x = self.left_panel_width
        video_width = self.width - self.left_panel_width - self.right_panel_width
        self.video_display = VideoDisplay(
            video_x, 0, video_width, main_height
        )

        # --- Log panel (right column) ---
        log_x = self.width - self.right_panel_width
        self.log_panel = LogPanel(
            log_x, 0, self.right_panel_width, main_height,
            max_lines=self.config.logging.max_lines
        )

        # --- Risk timeline (above the graph) ---
        timeline_height = 35
        self.risk_timeline = RiskTimeline(
            0, main_height - timeline_height - 5, self.width, timeline_height
        )

        # --- Score graph (at bottom, full width) ---
        self.score_graph = ScoreGraph(
            0, main_height, self.width, GRAPH_HEIGHT,
            max_points=300  # ~10 secondes à 30 FPS
        )

        # Welcome message
        self.log_panel.add_info("REBA 3D Application started")
        self.log_panel.add_info(f"Mode: {self.mode}")

        # Check calibration status
        if self.calibration_manager.is_calibrated:
            self.log_panel.add_success("Calibration: Loaded")
        else:
            self.log_panel.add_warning("Calibration: Not performed")

    def _apply_calibration_2d(self, angles: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Applique la calibration 2D aux angles (méthode indépendante).

        Args:
            angles: Nested dictionary of raw angles

        Returns:
            Nested dictionary of calibrated angles
        """
        import numpy as np
        from reba_3d.config.calibration_store import normalize_angle

        calibrated = {}

        for segment, angle_dict in angles.items():
            # If no offset for this segment, keep raw values
            if segment not in self.calibration_offsets_2d:
                calibrated[segment] = angle_dict.copy()
                continue

            segment_offsets = self.calibration_offsets_2d[segment]

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
                elif segment == "buste":
                    # Torso: direct subtraction
                    calibrated[segment][angle_name] = value - offset
                elif "shoulder" in segment:
                    # Shoulders: inverted (offset - angle)
                    calibrated[segment][angle_name] = offset - value
                else:
                    # Others (elbows, knees): absolute value
                    calibrated[segment][angle_name] = abs(value - offset)

        return calibrated

    def _on_mode_change(self, mode: str) -> None:
        """Callback for mode change."""
        self.mode = mode.lower()
        self.log_panel.add_info(f"Mode changed: {self.mode}")

        if self.mode == "inline":
            self.log_panel.add_warning("Inline mode: RealSense camera")
        else:
            bag_path = self.bag_directory / self.default_bag_name
            self.log_panel.add_info(f"File: {bag_path}")

    def _on_capture_toggle(self, state: bool) -> None:
        """Callback for start/stop capture."""
        if state:
            self._start_capture()
        else:
            self._stop_capture()

    def _on_calibration_toggle(self, state: bool) -> None:
        """Callback for start/stop calibration."""
        if state:
            # Check that capture is active
            if not self.capturing:
                self.log_panel.add_error("Start capture first")
                self.btn_calibration.set_state(False)
                return

            # Start calibration
            self.calibration_active = True
            self.calibration_angles = []
            self.calibration_angles_2d = []
            self.calibration_start_time = time.time()

            # Detect calibration mode
            # ALWAYS calibrate 2D+3D simultaneously if depth available (independent of comparison mode)
            if self.depth_intrinsics is not None:
                self.calibration_mode_dual = True
                self.calibration_mode_3d = True  # Used for main mode (3D)
                mode_str = "2D+3D"
                self.log_panel.add_info(f"Calibration {mode_str} ({self.calibration_duration}s)...")
                self.log_panel.add_warning("Stay in neutral position")
                logger.info(f"Dual 2D+3D calibration started for {self.calibration_duration}s")
            else:
                # No depth: 2D calibration only
                self.calibration_mode_dual = False
                self.calibration_mode_3d = False
                mode_str = "2D"
                self.log_panel.add_info(f"Calibration {mode_str} ({self.calibration_duration}s)...")
                self.log_panel.add_warning("Stay in neutral position")
                logger.info(f"Calibration {mode_str} started for {self.calibration_duration}s")

            logger.info(f"N_Neutre={self.config.calibration.n_neutre} frames, skip_windows={self.config.calibration.skip_windows}")

            # Start a thread to stop calibration after the duration
            def calibration_timer():
                time.sleep(self.calibration_duration)
                if self.calibration_active:
                    self._finish_calibration()

            timer_thread = threading.Thread(target=calibration_timer, daemon=True)
            timer_thread.start()
        else:
            # Manual calibration stop
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        """Finish calibration and compute offsets."""
        self.calibration_active = False
        self.btn_calibration.set_state(False)

        n_frames = len(self.calibration_angles)
        n_frames_2d = len(self.calibration_angles_2d)

        # Dual mode: check both lists
        if self.calibration_mode_dual:
            logger.info(f"Dual calibration finished: {n_frames} 3D frames, {n_frames_2d} 2D frames collected")
            if n_frames < 30 or n_frames_2d < 30:
                self.log_panel.add_error(f"Calibration failed: {n_frames} 3D frames, {n_frames_2d} 2D frames (min 30 each)")
                logger.warning("Not enough frames for dual calibration")
                self.calibration_angles = []
                self.calibration_angles_2d = []
                self.calibration_start_time = None
                return
        else:
            logger.info(f"Calibration finished: {n_frames} frames collected")
            if n_frames < 30:
                self.log_panel.add_error(f"Calibration failed: {n_frames} frames (min 30)")
                logger.warning("Not enough frames for calibration")
                self.calibration_angles = []
                self.calibration_angles_2d = []
                self.calibration_start_time = None
                return

        # Compute offsets from collected angles (nested structure)
        try:
            # Choice of calibration method via configuration
            use_robust = getattr(self.config.calibration, 'use_robust_calibration', True)
            k_mad = getattr(self.config.calibration, 'k_mad', 3.5)

            from pathlib import Path
            from reba_3d.config.calibration_store import get_calibration_path

            # Dual mode: calibrate 2D and 3D simultaneously
            if self.calibration_mode_dual:
                logger.info("Dual 2D+3D calibration in progress...")
                success_count = 0

                # --- 3D Calibration ---
                logger.info(f"Computing 3D offsets: {'robust MAD' if use_robust else 'legacy'}")
                if use_robust:
                    offsets_3d = compute_calibration_offsets_robust(
                        self.calibration_angles,
                        n_neutre=self.config.calibration.n_neutre,
                        k_mad=k_mad
                    )
                else:
                    offsets_3d = compute_calibration_offsets_nested(
                        self.calibration_angles,
                        window_size=self.config.calibration.n_neutre,
                        skip_windows=self.config.calibration.skip_windows
                    )

                if offsets_3d:
                    metadata_3d = {
                        'frames_collected': n_frames,
                        'duration_seconds': self.calibration_duration,
                        'n_neutre': self.config.calibration.n_neutre,
                        'mode': self.mode,
                        'mode_3d': True,
                        'dual_calibration': True,
                        'method': 'robust_mad' if use_robust else 'legacy_averaging',
                        'k_mad': k_mad if use_robust else None,
                    }
                    calibration_path_3d = get_calibration_path(mode_3d=True)
                    if save_calibration(offsets_3d, path=calibration_path_3d, metadata=metadata_3d):
                        self.calibration_manager.reload(path=calibration_path_3d, mode_3d=True)
                        logger.info(f"✓ 3D calibration saved: {calibration_path_3d.name}")
                        # Log 3D offsets for debug
                        for segment, angle_dict in offsets_3d.items():
                            if isinstance(angle_dict, dict):
                                angle_strs = [f"{k}={v:.1f}°" for k, v in angle_dict.items()]
                                logger.info(f"  3D {segment}: {', '.join(angle_strs)}")
                        success_count += 1
                    else:
                        logger.error("✗ Failed to save 3D calibration")
                else:
                    logger.warning("✗ No 3D offset computed")

                # --- 2D Calibration ---
                logger.info(f"Computing 2D offsets: {'robust MAD' if use_robust else 'legacy'}")
                if use_robust:
                    offsets_2d = compute_calibration_offsets_robust(
                        self.calibration_angles_2d,
                        n_neutre=self.config.calibration.n_neutre,
                        k_mad=k_mad
                    )
                else:
                    offsets_2d = compute_calibration_offsets_nested(
                        self.calibration_angles_2d,
                        window_size=self.config.calibration.n_neutre,
                        skip_windows=self.config.calibration.skip_windows
                    )

                if offsets_2d:
                    metadata_2d = {
                        'frames_collected': n_frames_2d,
                        'duration_seconds': self.calibration_duration,
                        'n_neutre': self.config.calibration.n_neutre,
                        'mode': self.mode,
                        'mode_3d': False,
                        'dual_calibration': True,
                        'method': 'robust_mad' if use_robust else 'legacy_averaging',
                        'k_mad': k_mad if use_robust else None,
                    }
                    calibration_path_2d = get_calibration_path(mode_3d=False)
                    if save_calibration(offsets_2d, path=calibration_path_2d, metadata=metadata_2d):
                        # Reload 2D offsets in our separate dict
                        self.calibration_offsets_2d = offsets_2d.copy()
                        logger.info(f"✓ 2D calibration saved: {calibration_path_2d.name}")
                        # Log 2D offsets for debug
                        for segment, angle_dict in offsets_2d.items():
                            if isinstance(angle_dict, dict):
                                angle_strs = [f"{k}={v:.1f}°" for k, v in angle_dict.items()]
                                logger.info(f"  2D {segment}: {', '.join(angle_strs)}")
                        # Reset log flag to display new offsets
                        if hasattr(self, '_logged_2d_offsets'):
                            delattr(self, '_logged_2d_offsets')
                        success_count += 1
                    else:
                        logger.error("✗ Failed to save 2D calibration")
                else:
                    logger.warning("✗ No 2D offset computed")

                # Final messages
                if success_count == 2:
                    method_str = "robust MAD" if use_robust else "simple average"
                    self.log_panel.add_success(f"Calibration 2D+3D OK")
                    self.log_panel.add_info(f"3D: {n_frames} frames, 2D: {n_frames_2d} frames")
                    logger.info("Dual calibration successful and saved")
                elif success_count == 1:
                    self.log_panel.add_warning("Partial calibration (1/2 successful)")
                else:
                    self.log_panel.add_error("Dual calibration failed")

            # Simple mode: 2D or 3D calibration
            else:
                if use_robust:
                    logger.info(
                        f"Calibration robuste MAD (n_neutre={self.config.calibration.n_neutre}, "
                        f"k_mad={k_mad})"
                    )
                    offsets = compute_calibration_offsets_robust(
                        self.calibration_angles,
                        n_neutre=self.config.calibration.n_neutre,
                        k_mad=k_mad
                    )
                else:
                    logger.info(
                        f"Calibration legacy (window_size={self.config.calibration.n_neutre}, "
                        f"skip_windows={self.config.calibration.skip_windows})"
                    )
                    offsets = compute_calibration_offsets_nested(
                        self.calibration_angles,
                        window_size=self.config.calibration.n_neutre,
                        skip_windows=self.config.calibration.skip_windows
                    )

                if not offsets:
                    self.log_panel.add_error("Calibration failed: no valid angles")
                    logger.warning("No offset computed")
                    return

                # Save offsets with enriched metadata
                metadata = {
                    'frames_collected': n_frames,
                    'duration_seconds': self.calibration_duration,
                    'n_neutre': self.config.calibration.n_neutre,
                    'skip_windows': self.config.calibration.skip_windows,
                    'mode': self.mode,
                    'mode_3d': self.calibration_mode_3d,
                    'dual_calibration': False,
                    'method': 'robust_mad' if use_robust else 'legacy_averaging',
                    'k_mad': k_mad if use_robust else None,
                }

                calibration_path = get_calibration_path(mode_3d=self.calibration_mode_3d)

                if save_calibration(offsets, path=calibration_path, metadata=metadata):
                    # Reload calibration manager with the correct mode
                    self.calibration_manager.reload(path=calibration_path, mode_3d=self.calibration_mode_3d)

                    # Success message with method and mode indication
                    method_str = "robust MAD" if use_robust else "simple average"
                    mode_str = "3D" if self.calibration_mode_3d else "2D"
                    self.log_panel.add_success(f"Calibration {mode_str} OK ({method_str})")
                    self.log_panel.add_info(f"{n_frames} frames → {self.config.calibration.n_neutre} used")
                    self.log_panel.add_info(f"File: {calibration_path.name}")

                    # Display computed offsets (nested format)
                    for segment, angle_dict in offsets.items():
                        if isinstance(angle_dict, dict):
                            # Nested format (nautical)
                            angle_strs = [f"{k}={v:.1f}°" for k, v in angle_dict.items()]
                            logger.info(f"  {segment}: {', '.join(angle_strs)}")
                        else:
                            # Simple format (legacy, should not happen)
                            logger.info(f"  {segment}: {angle_dict:.2f}°")

                    logger.info("Calibration successful and saved")
                else:
                    self.log_panel.add_error("Calibration save error")
                    logger.error("Calibration save failed")

        except ValueError as e:
            # Specific calibration errors (insufficient frames, etc.)
            self.log_panel.add_error(f"Calibration impossible: {str(e)[:40]}")
            logger.error(f"Calibration error: {e}")
        except Exception as e:
            self.log_panel.add_error(f"Calibration error: {str(e)[:20]}")
            logger.exception(f"Error during offset calculation: {e}")

        # Reset data
        self.calibration_angles = []
        self.calibration_angles_2d = []
        self.calibration_start_time = None
        self.calibration_mode_dual = False

    def _on_scores_toggle(self, state: bool) -> None:
        """Callback for show/hide scores."""
        self.show_scores = state
        self.video_display.overlay_scores = state
        if state:
            self.log_panel.add_info("Score display enabled")
        else:
            self.log_panel.add_info("Score display disabled")

    def _on_comparison_toggle(self, state: bool) -> None:
        """Callback for enabling/disabling 2D/3D comparison mode."""
        self.show_comparison = state
        if state:
            # Reload 2D offsets in case they were updated
            from reba_3d.config.calibration_store import get_calibration_path, DEFAULT_NAUTICAL_OFFSETS
            calib_2d_path = get_calibration_path(mode_3d=False)
            if calib_2d_path.exists():
                self.calibration_offsets_2d = load_calibration(calib_2d_path)
                logger.info(f"Calibration 2D rechargée: {calib_2d_path.name}")
            else:
                logger.warning("Aucune calibration 2D trouvée, utilisation des valeurs par défaut")
                self.calibration_offsets_2d = DEFAULT_NAUTICAL_OFFSETS.copy()
            self.log_panel.add_info("2D/3D comparison mode enabled")
        else:
            self.log_panel.add_info("Comparison mode disabled")

    def _on_log_angles_toggle(self, state: bool) -> None:
        """Callback for enabling/disabling angle logging."""
        self.angle_logging_enabled = state
        if state:
            # Start recording — pick mode based on current state
            if self.show_comparison:
                mode = "2d3d"
            elif self.use_3d and self.depth_intrinsics:
                mode = "3d"
            else:
                mode = "2d"
            self.angle_logger = AngleLogger(mode=mode, fps=LOG_ANGLES_FPS)
            self.angle_logger.start_recording()
            # VideoWriter will be lazily created on the first frame
            # (we need the frame dimensions which are only known at that point)
            self.video_writer = None
            self.log_panel.add_info(f"Angle log started (mode {mode.upper()}, {LOG_ANGLES_FPS:g} fps)")
            logger.info(f"Starting angle log in {mode.upper()} mode, fps={LOG_ANGLES_FPS}")
        else:
            # Stop and save
            if self.angle_logger.is_recording():
                self.angle_logger.stop_recording()
                # Save files
                txt_path = self.angle_logger.save_to_file()
                csv_path = self.angle_logger.save_to_csv()
                self.log_panel.add_success(f"Angles saved: {txt_path.name}")
                logger.info(f"Angles saved: {txt_path}")
                logger.info(f"CSV saved: {csv_path}")

                # Finalize skeleton video
                if self.video_writer is not None:
                    video_path = self.video_writer.output_path
                    self.video_writer.release()
                    self.video_writer = None
                    self.log_panel.add_success(f"Video saved: {video_path.name}")
                    logger.info(f"Skeleton video saved: {video_path}")

    def _on_load_change(self, value: int) -> None:
        """Callback for load score change."""
        self.reba_scorer.set_load_score(value)
        self.reba_scorer_2d.set_load_score(value)
        self.log_panel.add_info(f"Load: {value}")

    def _on_coupling_change(self, value: int) -> None:
        """Callback for coupling score change."""
        self.reba_scorer.set_coupling_score(value)
        self.reba_scorer_2d.set_coupling_score(value)
        self.log_panel.add_info(f"Coupling: {value}")

    def _on_activity_change(self, value: int) -> None:
        """Callback for activity score change."""
        self.reba_scorer.set_activity_score(value)
        self.reba_scorer_2d.set_activity_score(value)
        self.log_panel.add_info(f"Activity: {value}")

    def _on_pause_toggle(self, state: bool) -> None:
        """Callback for pause/resume video."""
        self.paused = state
        if state:
            self.pause_event.set()
            # In inline mode, freeze current frame
            if self.mode == "inline":
                with self.frame_lock:
                    if self.current_frame is not None:
                        self.frozen_frame = self.current_frame.copy()
            self.log_panel.add_info("Pause enabled")
            logger.debug("Pause enabled")
        else:
            self.pause_event.clear()
            self.frozen_frame = None
            self.log_panel.add_info("Playback resumed")
            logger.debug("Playback resumed")

    def _on_exit(self) -> None:
        """Callback for exit."""
        self.log_panel.add_info("Closing application...")
        self.running = False

    def _analyze_keypoints_json(self) -> None:
        """Open a file selector and analyze the keypoints_3d.json file."""
        # Clear old risk data
        self.risk_data = None
        self.risk_timeline.clear()

        try:
            # Determine default directory (try config, then data_output, then cwd)
            default_dir = os.getcwd()
            if hasattr(self.config, 'paths') and hasattr(self.config.paths, 'keypoints_directory'):
                candidate = Path(self.config.paths.keypoints_directory).resolve()
                if candidate.is_dir():
                    default_dir = str(candidate)
            # Fallback: if configured dir missing, try src/data_output
            if not Path(default_dir).is_dir() or default_dir == os.getcwd():
                data_output = Path("src/data_output").resolve()
                if data_output.is_dir():
                    default_dir = str(data_output)

            # Create file selector
            file_selector = FileSelector(
                directory=default_dir,
                extension=".json",
                title="Select keypoints_3d.json"
            )

            # Display selector and get file
            filepath = file_selector.run()

            # Restore main screen after selector
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.config.gui.title)

            if filepath:
                self.log_panel.add_info("Analysis in progress...")
                logger.info(f"Analyse de: {filepath}")

                try:
                    # Utiliser REBAAssessor pour l'analyse complète
                    from reba_3d.reba.risk_assessment import REBAAssessor

                    keypoints_file = Path(filepath)
                    risk_times_dir = keypoints_file.parent / "risk_times"
                    risk_times_dir.mkdir(exist_ok=True)

                    # Get neutral frames for custom calibration (if configured)
                    neutral_frames = self.config.calibration.get_neutral_frames()
                    if neutral_frames:
                        self.log_panel.add_info(f"Calibration: frames {neutral_frames[0]}-{neutral_frames[1]}")
                    else:
                        self.log_panel.add_info("Calibration: offsets statiques")

                    # === Analyse REBA 3D ===
                    self.log_panel.add_info("Analyse REBA 3D...")
                    assessor_3d = REBAAssessor(
                        window_size=self.config.reba.window_size,
                        fps=self.config.reba.video_fps,
                        feet_threshold=self.config.reba.feet_contact_threshold,
                        mode="3d",
                        neutral_frames=neutral_frames
                    )

                    results_3d = assessor_3d.analyze(
                        filepath,
                        load_malus=self.config.reba.load_score,
                        coupling_malus=self.config.reba.coupling_score,
                        activity_malus=self.config.reba.activity_score
                    )

                    # Sauvegarder risk_times_3d.json
                    risk_times_3d = assessor_3d.get_risk_times_for_video()
                    risk_times_path_3d = risk_times_dir / "risk_times_3d.json"
                    with open(risk_times_path_3d, 'w') as f:
                        json.dump(risk_times_3d, f, indent=2)

                    # Sauvegarder le rapport détaillé 3D
                    detailed_log_path_3d = keypoints_file.parent / "reba_analysis_3d.log"
                    assessor_3d.save_detailed_analysis_log(str(detailed_log_path_3d))

                    self.log_panel.add_success(f"✓ risk_times_3d.json créé")
                    self.log_panel.add_success(f"✓ reba_analysis_3d.log créé")
                    logger.info(f"Fichier exporté: {risk_times_path_3d}")
                    logger.info(f"Fichier exporté: {detailed_log_path_3d}")

                    # === Analyse REBA 2D ===
                    self.log_panel.add_info("Analyse REBA 2D...")
                    assessor_2d = REBAAssessor(
                        window_size=self.config.reba.window_size,
                        fps=self.config.reba.video_fps,
                        feet_threshold=self.config.reba.feet_contact_threshold,
                        mode="2d",
                        neutral_frames=neutral_frames
                    )

                    results_2d = assessor_2d.analyze(
                        filepath,
                        load_malus=self.config.reba.load_score,
                        coupling_malus=self.config.reba.coupling_score,
                        activity_malus=self.config.reba.activity_score
                    )

                    # Sauvegarder risk_times_2d.json
                    risk_times_2d = assessor_2d.get_risk_times_for_video()
                    risk_times_path_2d = risk_times_dir / "risk_times_2d.json"
                    with open(risk_times_path_2d, 'w') as f:
                        json.dump(risk_times_2d, f, indent=2)

                    # Sauvegarder le rapport détaillé 2D
                    detailed_log_path_2d = keypoints_file.parent / "reba_analysis_2d.log"
                    assessor_2d.save_detailed_analysis_log(str(detailed_log_path_2d))

                    self.log_panel.add_success(f"✓ risk_times_2d.json créé")
                    self.log_panel.add_success(f"✓ reba_analysis_2d.log créé")
                    logger.info(f"Fichier exporté: {risk_times_path_2d}")
                    logger.info(f"Fichier exporté: {detailed_log_path_2d}")

                    # Summary
                    self.log_panel.add_success(f"Analysis complete: {results_3d['num_windows']} windows")
                    logger.info(f"Analyse réussie: {results_3d['num_windows']} fenêtres")

                    # Display formatted results (use 3D as reference)
                    self._display_reba_results(assessor_3d)

                    # Convert risk_times for timeline (frames instead of seconds)
                    # Use 3D results for timeline
                    self._update_timeline_from_results(assessor_3d, results_3d)

                except json.JSONDecodeError as e:
                    self.log_panel.add_error(f"JSON invalide: {str(e)[:20]}")
                    logger.error(f"Erreur parsing JSON: {e}")
                except Exception as e:
                    self.log_panel.add_error(f"Erreur analyse: {str(e)[:30]}")
                    logger.exception(f"Erreur lors de l'analyse: {e}")
            else:
                self.log_panel.add_info("Selection cancelled")
                logger.info("File selection cancelled by user")

        except Exception as e:
            self.log_panel.add_error(f"Selection error: {str(e)[:20]}")
            logger.exception(f"Error during file selection: {e}")

    def _display_reba_results(self, assessor) -> None:
        """Display formatted REBA results in the LogPanel."""
        self.log_panel.add_info("=" * 30)
        self.log_panel.add_info("REBA RESULTS")
        self.log_panel.add_info("=" * 30)

        # Build requested format: [ F1: no risk, F2: no risk, ... ]
        risk_labels = assessor.scores["risk_labels"]
        recalculated = assessor.scores["recalculated"]

        formatted_results = []
        for i, (label, recalc) in enumerate(zip(risk_labels, recalculated), start=1):
            tag = "*" if recalc else ""
            formatted_results.append(f"F{i}{tag}: {label}")

        # Display in groups of 5 to avoid overloading the log
        result_text = "[ " + ", ".join(formatted_results) + " ]"

        # Split into lines of max 60 characters
        lines = []
        current_line = "["
        for i, item in enumerate(formatted_results):
            separator = ", " if i > 0 else " "
            if len(current_line + separator + item) > 60 and current_line != "[":
                lines.append(current_line + ",")
                current_line = "  " + item
            else:
                current_line += separator + item

        if current_line:
            lines.append(current_line + " ]")

        for line in lines:
            self.log_panel.add_info(line)

        # Statistics by risk level
        self.log_panel.add_info("")
        self.log_panel.add_info("STATISTICS:")

        from collections import Counter
        risk_counts = Counter(risk_labels)

        for risk_level, count in risk_counts.items():
            if risk_level == "negligible risk":
                self.log_panel.add_success(f"  {risk_level}: {count}")
            elif risk_level == "low risk":
                self.log_panel.add_info(f"  {risk_level}: {count}")
            elif risk_level == "medium risk":
                self.log_panel.add_warning(f"  {risk_level}: {count}")
            elif risk_level in ["high risk", "very high risk"]:
                self.log_panel.add_error(f"  {risk_level}: {count}")
            else:
                self.log_panel.add_info(f"  {risk_level}: {count}")

        self.log_panel.add_info("=" * 30)

    def _update_timeline_from_results(self, assessor, results) -> None:
        """Update timeline with REBA results (in frames)."""
        # Build risk intervals in frames from windows_info
        fps = self.config.reba.video_fps
        window_size = self.config.reba.window_size

        risk_data_frames = {
            "negligible risk": [],
            "low risk": [],
            "medium risk": [],
            "high risk": [],
            "very high risk": [],
            "invalid": []
        }

        label_map_fr_to_en = {
            "negligible risk": "negligible risk",
            "low risk": "low risk",
            "medium risk": "medium risk",
            "high risk": "high risk",
            "very high risk": "very high risk",
            "invalide": "invalid"
        }

        # Iterate through windows_info and risk_labels to build intervals in frames
        for i, window_info in enumerate(results['windows_info']):
            if i < len(assessor.scores["risk_labels"]):
                label_fr = assessor.scores["risk_labels"][i]
                label_en = label_map_fr_to_en.get(label_fr, "invalid")

                # Get start and end frames from windows_info
                frames = window_info.get("frames", [])
                if frames:
                    start_frame = frames[0]
                    end_frame = frames[-1]
                    risk_data_frames[label_en].append([start_frame, end_frame])

        # Calculate total frames
        total_frames = 0
        for intervals in risk_data_frames.values():
            for start, end in intervals:
                total_frames = max(total_frames, end)

        # Update timeline
        if total_frames > 0:
            self.risk_data = risk_data_frames
            self.risk_timeline.set_data(risk_data_frames, total_frames)
            logger.info(f"Timeline mise à jour: {total_frames} frames")

    def _start_capture(self) -> None:
        """Start video capture."""
        if self.capturing:
            return

        # Clear old risk data from timeline
        self.risk_data = None
        self.risk_timeline.clear()

        # Reset keypoints 3D accumulator
        self.keypoints_3d_data = []

        self.stop_capture.clear()
        self.capturing = True

        if self.mode == "inline":
            self.capture_thread = threading.Thread(
                target=self._capture_inline,
                daemon=True
            )
            self.log_panel.add_info("Starting inline capture...")
        else:
            self.capture_thread = threading.Thread(
                target=self._capture_offline,
                daemon=True
            )
            self.log_panel.add_info("Starting offline capture...")

        self.capture_thread.start()

    def _stop_capture(self) -> None:
        """Stop video capture."""
        self.stop_capture.set()
        self.capturing = False
        # Reset pause state
        self.paused = False
        self.pause_event.clear()
        self.frozen_frame = None
        self.btn_pause.set_state(False)
        # Reset graph
        self.score_graph.clear()
        self.current_reba_score = None
        self.current_reba_score_2d = None

        # Save accumulated 3D keypoints to JSON
        if self.keypoints_3d_data:
            output_dir = Path("src/data_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            kp_path = output_dir / "keypoints_3d.json"
            try:
                with open(kp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.keypoints_3d_data, f, indent=4)
                self.log_panel.add_success(f"Keypoints 3D: {kp_path.name} ({len(self.keypoints_3d_data)} frames)")
                logger.info(f"Keypoints 3D saved: {kp_path} ({len(self.keypoints_3d_data)} frames)")
            except Exception as e:
                self.log_panel.add_error(f"Keypoints save error: {str(e)[:30]}")
                logger.exception(f"Error saving keypoints 3D: {e}")
            self.keypoints_3d_data = []

        # Reset frame counter
        self.current_frame_number = 0
        self.log_panel.add_info("Capture stopped")

    def _capture_inline(self) -> None:
        """Capture thread in inline mode (live RealSense camera)."""
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            rs_config = rs.config()

            # Use config.yaml parameters
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
            logger.info("RealSense camera connected")
            self.log_panel.add_success("RealSense camera connected")

            # Align depth to color
            align = rs.align(rs.stream.color)

            # Get COLOR camera intrinsics (since depth is aligned to color)
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
                    # In inline pause mode, we continue capturing but don't update display
                    if not self.pause_event.is_set():
                        self._process_frame(frame, depth_frame)

            pipeline.stop()
            self.depth_intrinsics = None
            logger.info("RealSense camera disconnected")
            self.log_panel.add_info("RealSense camera disconnected")

        except ImportError:
            logger.error("pyrealsense2 not installed")
            self.log_panel.add_error("pyrealsense2 not installed")
        except Exception as e:
            logger.exception(f"Camera error: {e}")
            self.log_panel.add_error(f"Camera error: {str(e)[:30]}")

    def _capture_offline(self) -> None:
        """Capture thread in offline mode (.bag file)."""
        bag_path = self.bag_directory / self.default_bag_name

        if not bag_path.exists():
            logger.error(f"File not found: {bag_path}")
            self.log_panel.add_error(f"File not found: {bag_path}")
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

            # Align depth to color
            align = rs.align(rs.stream.color)
            logger.info(f".bag file loaded: {bag_path}")
            self.log_panel.add_success(f".bag file loaded")

            # Get intrinsics from COLOR stream (since depth is aligned to color)
            try:
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                self.depth_intrinsics = color_profile.get_intrinsics()
                logger.info(f"Intrinsics 3D (aligned): {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
                self.log_panel.add_success("3D mode enabled")
            except Exception as e:
                logger.warning(f"Unable to get intrinsics: {e}")
                self.depth_intrinsics = None
                self.log_panel.add_warning("2D mode (no depth)")

            was_paused = False
            frame_count = 0

            while not self.stop_capture.is_set():
                # Handle pause for offline mode
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

                        # Pass depth_frame for 3D calculation
                        self._process_frame(frame, depth_frame)
                        frame_count += 1

                        # Periodic log
                        if frame_count % 100 == 0:
                            logger.debug(f"Frame {frame_count} traité")

                except RuntimeError:
                    # End of file
                    logger.info("End of .bag file")
                    self.log_panel.add_info("End of .bag file")
                    break

            pipeline.stop()
            self.depth_intrinsics = None

        except ImportError:
            logger.error("pyrealsense2 not installed")
            self.log_panel.add_error("pyrealsense2 not installed")
        except Exception as e:
            logger.exception(f"Read error: {e}")
            self.log_panel.add_error(f"Read error: {str(e)[:30]}")

        self.capturing = False
        self.btn_capture.set_state(False)

    def _init_openpose(self) -> bool:
        """
        Initialize OpenPose detector if local mode.

        Returns:
            True if initialized successfully, False otherwise
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
        Process a frame (OpenPose detection, REBA 3D calculation).

        Args:
            frame: BGR numpy array image
            depth_frame: Aligned RealSense depth frame (optional)
        """
        import cv2

        # Increment frame counter
        self.current_frame_number += 1

        # Update timeline avec la frame actuelle
        if self.risk_data:
            self.risk_timeline.set_current_frame(self.current_frame_number)

        output_frame = frame.copy()
        keypoints = None

        # OpenPose detection if available
        if OPENPOSE_MODE == "local":
            # Initialize OpenPose on first call
            if not self._openpose_initialized:
                self._init_openpose()

            # Run detection
            if self.openpose_detector is not None:
                try:
                    output_frame, keypoints = self.openpose_detector.detect(frame)
                except Exception as e:
                    logger.warning(f"Erreur détection: {e}")

        # Process keypoints if available
        if keypoints is not None:
            try:
                if len(keypoints) > 0 and len(keypoints.shape) >= 2:
                    person_keypoints = keypoints[0]

                    # Accumulate 3D keypoints for JSON export
                    if depth_frame is not None and self.depth_intrinsics is not None:
                        import pyrealsense2 as rs
                        person_kps = []
                        for person in keypoints:
                            joints = []
                            for kp in person:
                                x_px, y_px, conf = kp
                                x_px, y_px = int(x_px), int(y_px)
                                x_px = max(0, min(x_px, depth_frame.get_width() - 1))
                                y_px = max(0, min(y_px, depth_frame.get_height() - 1))
                                depth = depth_frame.get_distance(x_px, y_px)
                                if depth > 0:
                                    p3d = rs.rs2_deproject_pixel_to_point(
                                        self.depth_intrinsics, [x_px, y_px], depth
                                    )
                                else:
                                    p3d = [0, 0, 0]
                                joints.append({
                                    "x": float(p3d[0]),
                                    "y": float(p3d[1]),
                                    "z": float(p3d[2]),
                                    "confidence": float(conf)
                                })
                            person_kps.append(joints)
                        self.keypoints_3d_data.append({
                            "frame": self.current_frame_number,
                            "keypoints_3d": person_kps
                        })

                    # Calibration mode: collect angles
                    if self.calibration_active:
                        raw_angles = None
                        raw_angles_2d = None

                        # Dual calibration mode: always calculate 2D and 3D if depth available
                        if self.calibration_mode_dual:
                            if depth_frame is not None and self.depth_intrinsics is not None:
                                # 3D calculation
                                raw_angles = calculate_nautical_angles_3d(
                                    person_keypoints,
                                    depth_frame,
                                    self.depth_intrinsics
                                )
                                # 2D calculation in parallel
                                raw_angles_2d = calculate_nautical_angles_2d(person_keypoints)

                                if raw_angles:
                                    self.calibration_angles.append(raw_angles)
                                if raw_angles_2d:
                                    self.calibration_angles_2d.append(raw_angles_2d)
                        else:
                            # No depth: 2D calibration only
                            raw_angles = calculate_nautical_angles_2d(person_keypoints)

                            if raw_angles:
                                self.calibration_angles.append(raw_angles)

                    # Normal mode: calculate REBA scores
                    else:
                        # Comparison mode: ALWAYS calculate both scores (2D and 3D)
                        if self.show_comparison:
                            if depth_frame is not None and self.depth_intrinsics is not None:
                                # Calculate 3D angles
                                raw_angles_3d = calculate_nautical_angles_3d(
                                    person_keypoints,
                                    depth_frame,
                                    self.depth_intrinsics
                                )
                                # Calculate 2D angles
                                raw_angles_2d = calculate_nautical_angles_2d(person_keypoints)

                                # Assurer que le calibration_manager a la calibration 3D chargée
                                if self.current_mode_3d != True:
                                    self.current_mode_3d = True
                                    from pathlib import Path
                                    from reba_3d.config.calibration_store import get_calibration_path
                                    calibration_path_3d = get_calibration_path(mode_3d=True)
                                    if calibration_path_3d.exists():
                                        self.calibration_manager.reload(path=calibration_path_3d, mode_3d=True)
                                        logger.info(f"Calibration 3D chargée: {calibration_path_3d.name}")

                                # Score 3D
                                if raw_angles_3d:
                                    calibrated_3d = self.calibration_manager.apply_nested(raw_angles_3d)
                                    self.current_angles = calibrated_3d
                                    self.current_reba_score = self.reba_scorer.update(calibrated_3d)
                                    # Log 3D angles if enabled
                                    if self.angle_logging_enabled and self.angle_logger.is_recording():
                                        self.angle_logger.log_frame(calibrated_3d, mode="3d")
                                else:
                                    self.current_reba_score = None

                                # Score 2D
                                if raw_angles_2d:
                                    calibrated_2d = self._apply_calibration_2d(raw_angles_2d)
                                    self.current_reba_score_2d = self.reba_scorer_2d.update(calibrated_2d)
                                    # Log 2D angles if enabled
                                    if self.angle_logging_enabled and self.angle_logger.is_recording():
                                        self.angle_logger.log_frame(calibrated_2d, mode="2d")
                                    # Debug: log les offsets utilisés (première fois uniquement)
                                    if not hasattr(self, '_logged_2d_offsets'):
                                        logger.debug(f"Offsets 2D actifs: {self.calibration_offsets_2d.get('neck', {})}")
                                        logger.debug(f"Offsets 3D actifs: {self.calibration_manager._offsets.get('neck', {})}")
                                        self._logged_2d_offsets = True
                                else:
                                    self.current_reba_score_2d = None

                            else:
                                # No depth: 2D only
                                raw_angles_2d = calculate_nautical_angles_2d(person_keypoints)
                                if raw_angles_2d:
                                    calibrated_2d = self._apply_calibration_2d(raw_angles_2d)
                                    self.current_reba_score_2d = self.reba_scorer_2d.update(calibrated_2d)
                                    self.current_angles = calibrated_2d
                                    # Log 2D angles if enabled
                                    if self.angle_logging_enabled and self.angle_logger.is_recording():
                                        self.angle_logger.log_frame(calibrated_2d, mode="2d")
                                else:
                                    self.current_reba_score_2d = None
                                # No 3D score
                                self.current_reba_score = None

                        # Normal mode (no comparison): calculate according to use_3d
                        else:
                            raw_angles = None
                            mode_3d = self.use_3d and depth_frame is not None and self.depth_intrinsics is not None

                            if mode_3d:
                                # 3D calculation
                                raw_angles = calculate_nautical_angles_3d(
                                    person_keypoints,
                                    depth_frame,
                                    self.depth_intrinsics
                                )
                            else:
                                # 2D calculation
                                raw_angles = calculate_nautical_angles_2d(person_keypoints)

                            if raw_angles:
                                # Reload appropriate calibration if mode changed
                                if mode_3d != self.current_mode_3d:
                                    self.current_mode_3d = mode_3d
                                    mode_str = "3D" if mode_3d else "2D"
                                    logger.info(f"Mode changé: passage en {mode_str}, rechargement calibration...")

                                    from pathlib import Path
                                    from reba_3d.config.calibration_store import get_calibration_path

                                    calibration_path = get_calibration_path(mode_3d=mode_3d)
                                    if calibration_path.exists():
                                        self.calibration_manager.reload(path=calibration_path, mode_3d=mode_3d)
                                        logger.info(f"Calibration {mode_str} chargée: {calibration_path.name}")
                                    else:
                                        logger.warning(f"Fichier calibration {mode_str} introuvable: {calibration_path.name}")
                                        logger.warning("Utilisation des valeurs par défaut")

                                # Apply calibration
                                calibrated_angles = self.calibration_manager.apply_nested(raw_angles)
                                self.current_angles = calibrated_angles
                                self.current_reba_score = self.reba_scorer.update(calibrated_angles)

                                # Log angles if enabled
                                if self.angle_logging_enabled and self.angle_logger.is_recording():
                                    log_mode = "3d" if mode_3d else "2d"
                                    self.angle_logger.log_frame(calibrated_angles, mode=log_mode)
                            else:
                                self.current_reba_score = None

                        # Update score graph
                        score_3d = self.current_reba_score.final_score if self.current_reba_score else None
                        score_2d = self.current_reba_score_2d.final_score if self.current_reba_score_2d else None
                        self.score_graph.add_scores(score_3d, score_2d)

                        # Save skeleton frame (before REBA overlay) for video export
                        if self.angle_logging_enabled and self.angle_logger.is_recording():
                            video_frame = output_frame.copy()
                            csv_frame_num = self.angle_logger._frame_count
                            cv2.putText(
                                video_frame,
                                f"Frame: {csv_frame_num}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 255), 2, cv2.LINE_AA,
                            )
                            h, w = video_frame.shape[:2]
                            if self.video_writer is None:
                                from datetime import datetime as _dt
                                ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                                video_path = self.angle_logger.output_dir / f"skeleton_{self.angle_logger.mode}_{ts}.avi"
                                self.video_writer = VideoWriter(
                                    str(video_path), fps=LOG_ANGLES_FPS, resolution=(w, h)
                                )
                                logger.info(f"Skeleton video started: {video_path}")
                            self.video_writer.write(video_frame)

                        # Draw score on frame if enabled (GUI display only)
                        if self.show_scores and (self.current_reba_score or self.current_reba_score_2d):
                            output_frame = self._draw_reba_overlay(output_frame)

            except Exception as e:
                logger.debug(f"Erreur traitement keypoints: {e}")

        # Display frame (with or without overlay)
        with self.frame_lock:
            self.current_frame = output_frame.copy()

    def _draw_reba_overlay(self, frame) -> np.ndarray:
        """
        Draw REBA overlay on frame.

        Args:
            frame: BGR image

        Returns:
            Frame with overlay
        """
        import cv2

        # Comparison mode: side by side 2D/3D display
        if self.show_comparison and (self.current_reba_score_2d is not None or self.current_reba_score is not None):
            return self._draw_comparison_overlay(frame)

        # Normal mode: display only main score
        if self.current_reba_score is None:
            return frame

        score = self.current_reba_score

        # Background color based on risk
        color = score.risk_color

        # Semi-transparent background rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 145), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 2D/3D indicator
        mode_3d = self.use_3d and self.depth_intrinsics is not None
        mode_text = "3D" if mode_3d else "2D"
        mode_color = (0, 255, 0) if mode_3d else (100, 100, 255)
        cv2.putText(
            frame, f"[{mode_text}]",
            (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1
        )

        # REBA score text
        cv2.putText(
            frame, f"REBA: {score.final_score}",
            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
        )

        # Risk level
        cv2.putText(
            frame, f"{score.risk_level.upper()}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        # Score details (optional)
        detail = f"A:{score.score_a} B:{score.score_b}"
        cv2.putText(
            frame, detail,
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        # Risk indicator color bar
        bar_width = int((score.final_score / 12) * 180)
        cv2.rectangle(frame, (10, 125), (10 + bar_width, 135), color, -1)
        cv2.rectangle(frame, (10, 125), (190, 135), (100, 100, 100), 1)

        return frame

    def _draw_comparison_overlay(self, frame) -> np.ndarray:
        """
        Draw 2D vs 3D comparison overlay on frame.

        Args:
            frame: BGR image

        Returns:
            Frame with comparison overlay
        """
        import cv2

        score_3d = self.current_reba_score
        score_2d = self.current_reba_score_2d

        # Larger semi-transparent background rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(
            frame, "2D vs 3D COMPARISON",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        # 3D Score (left column)
        cv2.putText(
            frame, "3D",
            (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        if score_3d is not None:
            cv2.putText(
                frame, f"REBA: {score_3d.final_score}",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_3d.risk_color, 2
            )
            cv2.putText(
                frame, f"{score_3d.risk_level[:10]}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_3d.risk_color, 1
            )
            # 3D bar
            bar_3d = int((score_3d.final_score / 12) * 130)
            cv2.rectangle(frame, (20, 120), (20 + bar_3d, 130), score_3d.risk_color, -1)
            cv2.rectangle(frame, (20, 120), (150, 130), (100, 100, 100), 1)
        else:
            cv2.putText(
                frame, "N/A",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2
            )
            cv2.putText(
                frame, "(pas de profondeur)",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1
            )

        # 2D Score (right column)
        cv2.putText(
            frame, "2D",
            (210, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2
        )
        if score_2d is not None:
            cv2.putText(
                frame, f"REBA: {score_2d.final_score}",
                (170, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_2d.risk_color, 2
            )
            cv2.putText(
                frame, f"{score_2d.risk_level[:10]}",
                (170, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_2d.risk_color, 1
            )
            # 2D bar
            bar_2d = int((score_2d.final_score / 12) * 130)
            cv2.rectangle(frame, (170, 120), (170 + bar_2d, 130), score_2d.risk_color, -1)
            cv2.rectangle(frame, (170, 120), (300, 130), (100, 100, 100), 1)
        else:
            cv2.putText(
                frame, "N/A",
                (170, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2
            )

        # Difference (only if both scores exist)
        if score_3d is not None and score_2d is not None:
            diff =  score_2d.final_score - score_3d.final_score
            diff_text = f"Diff: {diff:+d}"
            diff_color = (0, 255, 255) if diff != 0 else (200, 200, 200)
        else:
            diff_text = "Diff: N/A"
            diff_color = (100, 100, 100)
        cv2.putText(
            frame, diff_text,
            (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, diff_color, 1
        )

        return frame

    def _handle_events(self) -> None:
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Toggle capture with space
                    self.btn_capture.toggle()
                    self._on_capture_toggle(self.btn_capture.state)
                elif event.key == pygame.K_p:
                    # Toggle pause with P
                    if self.capturing:
                        self.btn_pause.toggle()
                        self._on_pause_toggle(self.btn_pause.state)

            # Propagate to components
            self.mode_selector.handle_event(event)
            self.btn_capture.handle_event(event)
            self.btn_pause.handle_event(event)
            self.btn_calibration.handle_event(event)
            self.btn_scores.handle_event(event)
            self.btn_comparison.handle_event(event)
            self.btn_load_keypoints.handle_event(event)
            self.btn_log_angles.handle_event(event)  # Désactivé
            self.score_load.handle_event(event)
            self.score_coupling.handle_event(event)
            self.score_activity.handle_event(event)
            self.btn_exit.handle_event(event)

    def _update(self) -> None:
        """Update application state."""
        # Update video display
        with self.frame_lock:
            # In inline pause mode, display frozen frame
            if self.paused and self.mode == "inline" and self.frozen_frame is not None:
                self.video_display.set_frame(self.frozen_frame)
            elif self.current_frame is not None:
                self.video_display.set_frame(self.current_frame)

    def _draw(self) -> None:
        """Draw the interface."""
        # Background
        self.screen.fill(DARK_GRAY)

        # Main area height (without graph)
        main_height = self.height - GRAPH_HEIGHT

        # Left panel (background)
        pygame.draw.rect(
            self.screen,
            (40, 40, 50),
            pygame.Rect(0, 0, self.left_panel_width, main_height)
        )

        # Left panel title
        title = self.title_font.render("Controls", True, WHITE)
        self.screen.blit(title, (10, 10))

        # Mode label
        mode_label = pygame.font.Font(None, 18).render("Mode:", True, GRAY)
        self.screen.blit(mode_label, (10, 45))

        # Components
        self.mode_selector.draw(self.screen)
        self.btn_capture.draw(self.screen)
        self.btn_pause.draw(self.screen)
        self.btn_calibration.draw(self.screen)
        self.btn_scores.draw(self.screen)
        self.btn_comparison.draw(self.screen)
        self.btn_load_keypoints.draw(self.screen)
        self.btn_log_angles.draw(self.screen)  # Désactivé

        # Parameters section label
        params_font = pygame.font.Font(None, 18)
        params_label = params_font.render("REBA Parameters:", True, GRAY)
        self.screen.blit(params_label, (10, self.params_label_y))

        # Score buttons
        self.score_load.draw(self.screen)
        self.score_coupling.draw(self.screen)
        self.score_activity.draw(self.screen)

        self.btn_exit.draw(self.screen)

        # Video area
        self.video_display.draw(self.screen)

        # Log panel
        self.log_panel.draw(self.screen)

        # Risk timeline
        self.risk_timeline.draw(self.screen)

        # Score graph (at bottom)
        self.score_graph.draw(self.screen)

        # Refresh screen
        pygame.display.flip()

    def run(self) -> None:
        """Main application loop."""
        logger.info("Starting main loop")
        self.running = True

        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(self.fps)

        # Cleanup
        logger.debug("Stopping application...")
        self._stop_capture()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        pygame.quit()
        logger.info("Application closed")


def main():
    """Entry point for the graphical interface."""
    from reba_3d.utils.logger import setup_logging
    setup_logging()

    app = REBAApp()
    app.run()


if __name__ == "__main__":
    main()
