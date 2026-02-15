"""
REBA risk assessment pipeline.

Orchestrates the complete REBA assessment process from keypoint data to risk scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from reba_3d.config.settings import (
    WINDOW_SIZE,
    POLY_ORDER,
    NEEDED_KEYPOINTS,
    FPS,
    FEET_CONTACT_THRESHOLD,
)
from reba_3d.core.keypoints import load_keypoints_3d, build_dataframe
from reba_3d.core.frame_classifier import classify_frames, filter_pertinent_frames
from reba_3d.core.smoothing import smooth_keypoints_window
from reba_3d.reba.angles import (
    compute_neck_angles,
    compute_torso_angles,
    compute_shoulder_angles_right,
    compute_shoulder_angles_left,
    compute_shoulder_elevation_right,
    compute_shoulder_elevation_left,
    compute_elbow_angle_right,
    compute_elbow_angle_left,
    compute_knee_angle_right,
    compute_knee_angle_left,
    compute_feet_contact,
    # 2D angle functions
    compute_neck_angles_2d,
    compute_torso_angles_2d,
    compute_shoulder_angles_right_2d,
    compute_shoulder_angles_left_2d,
)
from reba_3d.reba.calibration import (
    calibrate_all_angles,
    calibrate_all_angles_robust,
    compute_offsets_from_neutral_robust,
)
from reba_3d.reba.scoring import (
    score_neck,
    score_torso,
    score_shoulder,
    score_elbow,
    score_knee,
    score_feet_contact,
    score_wrist,
    calculate_score_a,
    calculate_score_b,
    calculate_score_c,
    apply_activity_malus,
    score_to_risk_label,
    score_to_risk_label_en,
)


class REBAAssessor:
    """
    REBA (Rapid Entire Body Assessment) analyzer.

    Processes 3D keypoint data to calculate REBA posture risk scores.

    Attributes:
        window_size: Number of frames per analysis window
        poly_order: Polynomial order for smoothing
        fps: Video frames per second
        feet_threshold: Threshold for feet contact detection
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        poly_order: int = POLY_ORDER,
        fps: int = FPS,
        feet_threshold: float = FEET_CONTACT_THRESHOLD,
        mode: str = "3d",
        neutral_frames: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize REBA assessor.

        Args:
            window_size: Number of frames per analysis window (default: 30)
            poly_order: Polynomial order for smoothing (default: 2)
            fps: Video frames per second (default: 15)
            feet_threshold: Threshold for feet contact detection (default: 0.10)
            mode: Angle calculation mode - "3d" for full 3D nautical angles,
                  "2d" for 2D planar projection (default: "3d")
            neutral_frames: Optional tuple (start_frame, end_frame) specifying frames
                           where the person is in a neutral standing position for calibration.
                           If None, uses pre-defined static calibration offsets.
        """
        self.window_size = window_size
        self.poly_order = poly_order
        self.fps = fps
        self.feet_threshold = feet_threshold
        self.mode = mode.lower()
        self.neutral_frames = neutral_frames

        if self.mode not in ("3d", "2d"):
            raise ValueError(f"Invalid mode '{mode}'. Must be '3d' or '2d'.")

        # Analysis results
        self.windows_info: List[Dict] = []
        self.raw_angles: Dict[str, Dict[str, List[float]]] = {}
        self.calibrated_angles: Dict[str, Dict[str, List[float]]] = {}
        self.scores: Dict[str, List] = {}
        self.risk_times: Dict[str, List] = {}
        self.custom_offsets: Optional[Dict[str, Dict[str, float]]] = None
        self.neutral_shoulder_width: Optional[float] = None

    def analyze(
        self,
        keypoints_path: str,
        load_malus: int = 1,
        coupling_malus: int = 1,
        activity_malus: int = 1,
    ) -> Dict[str, Any]:
        """
        Perform complete REBA analysis on keypoint data.

        Args:
            keypoints_path: Path to keypoints_3d.json file
            load_malus: Load/force malus score (default: 1)
            coupling_malus: Coupling quality malus score (default: 1)
            activity_malus: Activity malus score (default: 1)

        Returns:
            Dictionary containing analysis results
        """
        # Load and preprocess data
        data_json = load_keypoints_3d(keypoints_path)
        df = build_dataframe(data_json)

        # Classify frames
        pertinent_frames = classify_frames(df)
        df = filter_pertinent_frames(df, pertinent_frames)

        print(f"[OK] {len(pertinent_frames['face'])} front-view frames detected.")
        print(f"[OK] {len(pertinent_frames['profil_droit'])} right-profile frames detected.")
        print(f"[OK] {len(pertinent_frames['profil_gauche'])} left-profile frames detected.")

        # First pass: compute smoothed positions and raw angles
        self._first_pass(df)

        # Calculate ankle means for feet contact
        r_ankle_mean, l_ankle_mean = self._compute_ankle_means()

        # Second pass: compute contact states and finalize angles
        self._second_pass(r_ankle_mean, l_ankle_mean)

        # Calibrate angles
        if self.neutral_frames is not None:
            # Use custom calibration from neutral frames
            self._compute_custom_offsets_from_neutral()
            self.calibrated_angles = self._apply_custom_calibration()
            print(
                f"[OK] Custom calibration applied (frames {self.neutral_frames[0]}-{self.neutral_frames[1]})"
            )
        else:
            # Use static pre-defined calibration offsets
            self.calibrated_angles = calibrate_all_angles(self.raw_angles)
            print("[OK] Static calibration applied (pre-defined offsets)")

        # Calculate REBA scores
        self._calculate_scores(load_malus, coupling_malus, activity_malus)

        return self._build_results()

    def _first_pass(self, df: pd.DataFrame) -> None:
        """First pass: compute smoothed positions for each window."""
        self.windows_info = []
        self._positions_by_window = []
        self._r_ankle_list = []
        self._l_ankle_list = []
        self._contact_raw_state = []

        for start in range(0, len(df) - self.window_size + 1, self.window_size):
            df_window = df.iloc[start:start + self.window_size]
            frames_in_window = df_window["frame"].tolist()

            # Smoothed positions
            block_positions = smooth_keypoints_window(
                df_window, NEEDED_KEYPOINTS, self.poly_order
            )
            self._positions_by_window.append(block_positions)

            # Ankle tracking for feet contact
            if (np.isnan(block_positions.get("RAnkle", [np.nan])).any() or
                np.isnan(block_positions.get("LAnkle", [np.nan])).any()):
                self._contact_raw_state.append("404")
                self._r_ankle_list.append(np.nan)
                self._l_ankle_list.append(np.nan)
            else:
                self._contact_raw_state.append("...")
                self._r_ankle_list.append(float(block_positions["RAnkle"][1]))
                self._l_ankle_list.append(float(block_positions["LAnkle"][1]))

            self.windows_info.append({
                "start_index": start,
                "end_index": start + self.window_size - 1,
                "frames": frames_in_window,
            })

        # Compute neutral shoulder width for 2D enhanced mode (Strategy 3)
        if self.mode == "2d":
            self._compute_neutral_shoulder_width()

    def _compute_ankle_means(self) -> Tuple[float, float]:
        """Compute mean ankle positions for feet contact detection."""
        r_valid = [v for v in self._r_ankle_list if not np.isnan(v)]
        l_valid = [v for v in self._l_ankle_list if not np.isnan(v)]

        r_mean = np.mean(r_valid) if r_valid else 0.0
        l_mean = np.mean(l_valid) if l_valid else 0.0

        return r_mean, l_mean

    def _compute_neutral_shoulder_width(self) -> None:
        """
        Compute neutral shoulder width for 2D rotation estimation (Strategy 3).

        Uses the neutral frames if specified, otherwise takes the maximum
        projected shoulder width from all windows (assumes the widest
        observation is closest to a true frontal neutral pose).
        """
        widths = []

        if self.neutral_frames is not None:
            start_frame, end_frame = self.neutral_frames
            for i, window_info in enumerate(self.windows_info):
                window_frames = window_info["frames"]
                if window_frames[0] <= end_frame and window_frames[-1] >= start_frame:
                    pos = self._positions_by_window[i]
                    if ("RShoulder" in pos and "LShoulder" in pos and
                            not np.isnan(pos["RShoulder"]).any() and
                            not np.isnan(pos["LShoulder"]).any()):
                        w = abs(float(pos["RShoulder"][0] - pos["LShoulder"][0]))
                        widths.append(w)

        if not widths:
            # Fallback: use max width across all windows
            for pos in self._positions_by_window:
                if ("RShoulder" in pos and "LShoulder" in pos and
                        not np.isnan(pos["RShoulder"]).any() and
                        not np.isnan(pos["LShoulder"]).any()):
                    w = abs(float(pos["RShoulder"][0] - pos["LShoulder"][0]))
                    widths.append(w)

        if widths:
            self.neutral_shoulder_width = float(max(widths))
        else:
            self.neutral_shoulder_width = None

    def _second_pass(self, r_ankle_mean: float, l_ankle_mean: float) -> None:
        """Second pass: compute angles and contact states."""
        # Initialize angle storage
        self.raw_angles = {
            "neck": {"alpha": [], "beta": [], "gamma": []},
            "torso": {"alpha": [], "beta": [], "gamma": []},
            "right_shoulder": {"alpha": [], "beta": [], "gamma": [], "elevation": []},
            "left_shoulder": {"alpha": [], "beta": [], "gamma": [], "elevation": []},
            "right_elbow": {"angle": []},
            "left_elbow": {"angle": []},
            "right_knee": {"angle": []},
            "left_knee": {"angle": []},
        }
        self._contact_results = []

        for i, block_positions in enumerate(self._positions_by_window):
            # Feet contact
            if self._contact_raw_state[i] == "404":
                self._contact_results.append("404")
            else:
                contact = compute_feet_contact(
                    block_positions, r_ankle_mean, l_ankle_mean, self.feet_threshold
                )
                self._contact_results.append(contact)

            # Neck angles (3D or 2D based on mode)
            if self.mode == "2d":
                neck_angles = compute_neck_angles_2d(block_positions, neutral_shoulder_width=self.neutral_shoulder_width)
            else:
                neck_angles = compute_neck_angles(block_positions)
            if neck_angles:
                self.raw_angles["neck"]["alpha"].append(neck_angles[0])
                self.raw_angles["neck"]["beta"].append(neck_angles[1])
                self.raw_angles["neck"]["gamma"].append(neck_angles[2])
            else:
                self.raw_angles["neck"]["alpha"].append(np.nan)
                self.raw_angles["neck"]["beta"].append(np.nan)
                self.raw_angles["neck"]["gamma"].append(np.nan)

            # Torso angles (3D or 2D based on mode)
            if self.mode == "2d":
                torso_angles = compute_torso_angles_2d(block_positions, neutral_shoulder_width=self.neutral_shoulder_width)
            else:
                torso_angles = compute_torso_angles(block_positions)
            if torso_angles:
                self.raw_angles["torso"]["alpha"].append(torso_angles[0])
                self.raw_angles["torso"]["beta"].append(torso_angles[1])
                self.raw_angles["torso"]["gamma"].append(torso_angles[2])
            else:
                self.raw_angles["torso"]["alpha"].append(np.nan)
                self.raw_angles["torso"]["beta"].append(np.nan)
                self.raw_angles["torso"]["gamma"].append(np.nan)

            # Right shoulder (3D or 2D based on mode)
            if self.mode == "2d":
                shoulder_r = compute_shoulder_angles_right_2d(block_positions, neutral_shoulder_width=self.neutral_shoulder_width)
            else:
                shoulder_r = compute_shoulder_angles_right(block_positions)
            elev_r = compute_shoulder_elevation_right(block_positions)
            if shoulder_r:
                self.raw_angles["right_shoulder"]["alpha"].append(shoulder_r[0])
                self.raw_angles["right_shoulder"]["beta"].append(shoulder_r[1])
                self.raw_angles["right_shoulder"]["gamma"].append(shoulder_r[2])
            else:
                self.raw_angles["right_shoulder"]["alpha"].append(np.nan)
                self.raw_angles["right_shoulder"]["beta"].append(np.nan)
                self.raw_angles["right_shoulder"]["gamma"].append(np.nan)
            self.raw_angles["right_shoulder"]["elevation"].append(
                elev_r if elev_r else np.nan
            )

            # Left shoulder (3D or 2D based on mode)
            if self.mode == "2d":
                shoulder_l = compute_shoulder_angles_left_2d(block_positions, neutral_shoulder_width=self.neutral_shoulder_width)
            else:
                shoulder_l = compute_shoulder_angles_left(block_positions)
            elev_l = compute_shoulder_elevation_left(block_positions)
            if shoulder_l:
                self.raw_angles["left_shoulder"]["alpha"].append(shoulder_l[0])
                self.raw_angles["left_shoulder"]["beta"].append(shoulder_l[1])
                self.raw_angles["left_shoulder"]["gamma"].append(shoulder_l[2])
            else:
                self.raw_angles["left_shoulder"]["alpha"].append(np.nan)
                self.raw_angles["left_shoulder"]["beta"].append(np.nan)
                self.raw_angles["left_shoulder"]["gamma"].append(np.nan)
            self.raw_angles["left_shoulder"]["elevation"].append(
                elev_l if elev_l else np.nan
            )

            # Elbows
            elbow_r = compute_elbow_angle_right(block_positions)
            elbow_l = compute_elbow_angle_left(block_positions)
            self.raw_angles["right_elbow"]["angle"].append(elbow_r if elbow_r else np.nan)
            self.raw_angles["left_elbow"]["angle"].append(elbow_l if elbow_l else np.nan)

            # Knees
            knee_r = compute_knee_angle_right(block_positions)
            knee_l = compute_knee_angle_left(block_positions)
            self.raw_angles["right_knee"]["angle"].append(knee_r if knee_r else np.nan)
            self.raw_angles["left_knee"]["angle"].append(knee_l if knee_l else np.nan)

    def _calculate_scores(
        self,
        load_malus: int,
        coupling_malus: int,
        activity_malus: int
    ) -> None:
        """Calculate REBA scores for all windows."""
        num_windows = len(self.windows_info)
        cal = self.calibrated_angles

        # Initialize score lists
        scores_neck = []
        scores_torso = []
        scores_shoulder = []
        scores_elbow = []
        scores_knee = []
        scores_wrist = []

        neck_malus_list = []
        torso_malus_list = []
        shoulder_malus_list = []
        contact_malus_list = []

        for i in range(num_windows):
            # Neck
            a_neck = cal["neck"]["alpha"][i] if i < len(cal["neck"]["alpha"]) else np.nan
            b_neck = cal["neck"]["beta"][i] if i < len(cal["neck"]["beta"]) else np.nan
            g_neck = cal["neck"]["gamma"][i] if i < len(cal["neck"]["gamma"]) else np.nan
            s_neck, m_neck = score_neck(a_neck, b_neck, g_neck)
            scores_neck.append(s_neck)
            neck_malus_list.append(m_neck)

            # Torso
            a_torso = cal["torso"]["alpha"][i] if i < len(cal["torso"]["alpha"]) else np.nan
            b_torso = cal["torso"]["beta"][i] if i < len(cal["torso"]["beta"]) else np.nan
            g_torso = cal["torso"]["gamma"][i] if i < len(cal["torso"]["gamma"]) else np.nan
            s_torso, m_torso = score_torso(a_torso, b_torso, g_torso)
            scores_torso.append(s_torso)
            torso_malus_list.append(m_torso)

            # Shoulder
            a_sh_r = cal["right_shoulder"]["alpha"][i] if i < len(cal["right_shoulder"]["alpha"]) else np.nan
            a_sh_l = cal["left_shoulder"]["alpha"][i] if i < len(cal["left_shoulder"]["alpha"]) else np.nan
            elev_r = cal["right_shoulder"]["elevation"][i] if i < len(cal["right_shoulder"]["elevation"]) else np.nan
            elev_l = cal["left_shoulder"]["elevation"][i] if i < len(cal["left_shoulder"]["elevation"]) else np.nan
            s_sh, m_sh_r, m_sh_l = score_shoulder(a_sh_r, a_sh_l, elev_r, elev_l)
            scores_shoulder.append(s_sh)
            shoulder_malus_list.append(max(m_sh_r, m_sh_l))

            # Elbow
            elbow_r = cal["right_elbow"]["angle"][i] if i < len(cal["right_elbow"]["angle"]) else np.nan
            elbow_l = cal["left_elbow"]["angle"][i] if i < len(cal["left_elbow"]["angle"]) else np.nan
            scores_elbow.append(score_elbow(elbow_r, elbow_l))

            # Knee
            knee_r = cal["right_knee"]["angle"][i] if i < len(cal["right_knee"]["angle"]) else np.nan
            knee_l = cal["left_knee"]["angle"][i] if i < len(cal["left_knee"]["angle"]) else np.nan
            scores_knee.append(score_knee(knee_r, knee_l))

            # Wrist (default)
            scores_wrist.append(score_wrist())

            # Feet contact
            contact = self._contact_results[i] if i < len(self._contact_results) else "404"
            contact_malus_list.append(score_feet_contact(contact))

        # Total scores
        total_neck = [s + m for s, m in zip(scores_neck, neck_malus_list)]
        total_torso = [s + m for s, m in zip(scores_torso, torso_malus_list)]
        total_knee = [s + m for s, m in zip(scores_knee, contact_malus_list)]
        total_shoulder = [s + m for s, m in zip(scores_shoulder, shoulder_malus_list)]

        # Table A and B scores
        scores_a = []
        scores_b = []
        for i in range(num_windows):
            s_a = calculate_score_a(total_torso[i], total_neck[i], total_knee[i])
            s_b = calculate_score_b(total_shoulder[i], scores_elbow[i], scores_wrist[i])
            scores_a.append(s_a)
            scores_b.append(s_b)

        # Final scores with malus
        scores_c = []
        risk_labels = []
        recalculated_flags = []

        self.risk_times = {
            "no risk": [],
            "low risk": [],
            "medium risk": [],
            "high risk": [],
            "very high": [],
        }

        for i in range(num_windows):
            if isinstance(scores_a[i], int) and isinstance(scores_b[i], int):
                score_c = calculate_score_c(scores_a[i], scores_b[i])
                label = score_to_risk_label(score_c)

                # Apply malus for risk levels
                if label in ["low risk", "medium risk", "high risk", "very high"]:
                    final_score, recalculated = apply_activity_malus(
                        score_c, scores_a[i], scores_b[i],
                        load_malus, coupling_malus, activity_malus
                    )
                    score_c = final_score
                    label = score_to_risk_label(score_c)
                else:
                    recalculated = False
            else:
                score_c = "Invalid"
                label = "invalid"
                recalculated = False

            scores_c.append(score_c)
            risk_labels.append(label)
            recalculated_flags.append(recalculated)

            # Record risk times (in frames, not seconds - anti-drift)
            if label in self.risk_times and i < len(self.windows_info):
                window_frames = self.windows_info[i]["frames"]
                start_frame = int(window_frames[0])
                end_frame = int(window_frames[-1])
                self.risk_times[label].append((i + 1, start_frame, end_frame))

        # Store results
        self.scores = {
            "neck": scores_neck,
            "torso": scores_torso,
            "shoulder": scores_shoulder,
            "elbow": scores_elbow,
            "knee": scores_knee,
            "wrist": scores_wrist,
            "total_neck": total_neck,
            "total_torso": total_torso,
            "total_knee": total_knee,
            "total_shoulder": total_shoulder,
            "score_a": scores_a,
            "score_b": scores_b,
            "score_c": scores_c,
            "risk_labels": risk_labels,
            "recalculated": recalculated_flags,
        }

    def _build_results(self) -> Dict[str, Any]:
        """Build final results dictionary."""
        results = {
            "windows_info": self.windows_info,
            "scores": self.scores,
            "risk_times": self.risk_times,
            "calibrated_angles": self.calibrated_angles,
            "num_windows": len(self.windows_info),
        }
        if self.custom_offsets is not None:
            results["custom_offsets"] = self.custom_offsets
            results["neutral_frames"] = self.neutral_frames
        return results

    def _compute_custom_offsets_from_neutral(self) -> None:
        """
        Compute calibration offsets from the specified neutral frames.

        The neutral frames should contain a person standing in a neutral
        upright position (arms at sides, looking forward).

        Sets self.custom_offsets with computed offset values.
        """
        if self.neutral_frames is None:
            return

        start_frame, end_frame = self.neutral_frames

        # Find which windows correspond to neutral frames
        neutral_window_indices = []
        for i, window_info in enumerate(self.windows_info):
            window_frames = window_info["frames"]
            window_start = window_frames[0]
            window_end = window_frames[-1]

            # Check if window overlaps with neutral frames
            if window_start <= end_frame and window_end >= start_frame:
                neutral_window_indices.append(i)

        if not neutral_window_indices:
            print(f"[WARNING] No windows found in neutral frame range [{start_frame}, {end_frame}]")
            print("[WARNING] Using first window for calibration")
            neutral_window_indices = [0]

        # Extract raw angles from neutral windows
        neutral_angles: Dict[str, Dict[str, List[float]]] = {}

        for segment, angles_dict in self.raw_angles.items():
            neutral_angles[segment] = {}
            for angle_name, values in angles_dict.items():
                # Extract values only from neutral windows
                neutral_values = [
                    values[i] for i in neutral_window_indices
                    if i < len(values) and not np.isnan(values[i])
                ]
                neutral_angles[segment][angle_name] = neutral_values

        # Compute offsets as mean of neutral values
        self.custom_offsets = {}
        for segment, angles_dict in neutral_angles.items():
            self.custom_offsets[segment] = {}
            for angle_name, values in angles_dict.items():
                if values:
                    self.custom_offsets[segment][angle_name] = float(np.mean(values))
                else:
                    # Fallback to 0 if no valid values
                    self.custom_offsets[segment][angle_name] = 0.0

        print(f"[OK] Offsets computed from {len(neutral_window_indices)} neutral window(s)")

    def _apply_custom_calibration(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Apply custom calibration offsets to raw angles.

        Uses the same calibration logic as calibrate_all_angles but with
        custom offsets computed from neutral frames.

        Returns:
            Dictionary of calibrated angles
        """
        from reba_3d.core.geometry import normalize_angle

        if self.custom_offsets is None:
            return calibrate_all_angles(self.raw_angles)

        offsets = self.custom_offsets
        calibrated = {}

        # Helper functions (same as in calibration.py)
        def apply_neck(angles: List[float], offset: float) -> List[float]:
            return [
                normalize_angle(angle - offset) if not np.isnan(angle) and angle != 0.0 else 0.0
                for angle in angles
            ]

        def apply_torso(angles: List[float], offset: float) -> List[float]:
            return [
                angle - offset if not np.isnan(angle) and angle != 0.0 else 0.0
                for angle in angles
            ]

        def apply_shoulder(angles: List[float], offset: float) -> List[float]:
            return [
                offset - angle if not np.isnan(angle) and angle != 0.0 else 0.0
                for angle in angles
            ]

        def apply_standard(angles: List[float], offset: float) -> List[float]:
            return [
                abs(angle - offset) if not np.isnan(angle) and angle != 0.0 else 0.0
                for angle in angles
            ]

        # Apply calibration to each segment
        # Neck
        if "neck" in self.raw_angles and "neck" in offsets:
            calibrated["neck"] = {
                "alpha": apply_neck(self.raw_angles["neck"].get("alpha", []), offsets["neck"].get("alpha", 180.0)),
                "beta": apply_neck(self.raw_angles["neck"].get("beta", []), offsets["neck"].get("beta", 0.0)),
                "gamma": apply_neck(self.raw_angles["neck"].get("gamma", []), offsets["neck"].get("gamma", 0.0)),
            }

        # Torso
        if "torso" in self.raw_angles and "torso" in offsets:
            calibrated["torso"] = {
                "alpha": apply_torso(self.raw_angles["torso"].get("alpha", []), offsets["torso"].get("alpha", 90.0)),
                "beta": apply_torso(self.raw_angles["torso"].get("beta", []), offsets["torso"].get("beta", 0.0)),
                "gamma": apply_torso(self.raw_angles["torso"].get("gamma", []), offsets["torso"].get("gamma", 0.0)),
            }

        # Right shoulder
        if "right_shoulder" in self.raw_angles and "right_shoulder" in offsets:
            calibrated["right_shoulder"] = {
                "alpha": apply_shoulder(self.raw_angles["right_shoulder"].get("alpha", []), offsets["right_shoulder"].get("alpha", 0.0)),
                "beta": apply_shoulder(self.raw_angles["right_shoulder"].get("beta", []), offsets["right_shoulder"].get("beta", 0.0)),
                "gamma": apply_shoulder(self.raw_angles["right_shoulder"].get("gamma", []), offsets["right_shoulder"].get("gamma", 0.0)),
                "elevation": apply_shoulder(self.raw_angles["right_shoulder"].get("elevation", []), offsets["right_shoulder"].get("elevation", 90.0)),
            }

        # Right elbow
        if "right_elbow" in self.raw_angles and "right_elbow" in offsets:
            calibrated["right_elbow"] = {
                "angle": apply_standard(self.raw_angles["right_elbow"].get("angle", []), offsets["right_elbow"].get("angle", 170.0)),
            }

        # Right knee
        if "right_knee" in self.raw_angles and "right_knee" in offsets:
            calibrated["right_knee"] = {
                "angle": apply_standard(self.raw_angles["right_knee"].get("angle", []), offsets["right_knee"].get("angle", 178.0)),
            }

        # Left shoulder
        if "left_shoulder" in self.raw_angles and "left_shoulder" in offsets:
            calibrated["left_shoulder"] = {
                "alpha": apply_shoulder(self.raw_angles["left_shoulder"].get("alpha", []), offsets["left_shoulder"].get("alpha", 0.0)),
                "beta": apply_shoulder(self.raw_angles["left_shoulder"].get("beta", []), offsets["left_shoulder"].get("beta", 0.0)),
                "gamma": apply_shoulder(self.raw_angles["left_shoulder"].get("gamma", []), offsets["left_shoulder"].get("gamma", 0.0)),
                "elevation": apply_shoulder(self.raw_angles["left_shoulder"].get("elevation", []), offsets["left_shoulder"].get("elevation", 90.0)),
            }

        # Left elbow
        if "left_elbow" in self.raw_angles and "left_elbow" in offsets:
            calibrated["left_elbow"] = {
                "angle": apply_standard(self.raw_angles["left_elbow"].get("angle", []), offsets["left_elbow"].get("angle", 170.0)),
            }

        # Left knee
        if "left_knee" in self.raw_angles and "left_knee" in offsets:
            calibrated["left_knee"] = {
                "angle": apply_standard(self.raw_angles["left_knee"].get("angle", []), offsets["left_knee"].get("angle", 178.0)),
            }

        return calibrated

    def get_risk_times_for_video(self) -> Dict[str, List[List[int]]]:
        """
        Get risk times formatted for video annotation.

        Returns:
            Dictionary mapping English risk labels to frame intervals.
            Format: {"medium risk": [[start_frame, end_frame], ...], ...}
            Frame intervals are used instead of seconds to prevent FPS drift issues.
        """
        label_map = {
            "no risk": "negligible risk",
            "low risk": "low risk",
            "medium risk": "medium risk",
            "high risk": "high risk",
            "very high": "very high risk",
        }

        result = {}
        for en_label_key, intervals in self.risk_times.items():
            out_label = label_map.get(en_label_key)
            if out_label:
                result[out_label] = [[start_frame, end_frame] for _, start_frame, end_frame in intervals]

        return result

    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        from reba_3d.config.settings import ANSI_COLORS

        print("\n[STATS] Risks according to REBA:")
        for i, (label, recalc) in enumerate(
            zip(self.scores["risk_labels"], self.scores["recalculated"]),
            start=1
        ):
            tag = "*" if recalc else ""
            color = ANSI_COLORS.get(label, ANSI_COLORS["invalid"])
            reset = ANSI_COLORS["reset"]
            print(f"  F{i}{tag}: {color}{label}{reset}")

        print("\n🕒 Time intervals to monitor:")
        for risk_level, intervals in self.risk_times.items():
            if intervals:
                color = ANSI_COLORS.get(risk_level, ANSI_COLORS["invalid"])
                reset = ANSI_COLORS["reset"]
                print(f"\n{color}{risk_level}{reset} detected at:")
                for window_num, start_frame, end_frame in intervals:
                    # Convert frames to seconds for human-readable display
                    start_s = start_frame / self.fps
                    end_s = (end_frame + 1) / self.fps  # +1 to include end_frame
                    print(
                        f"  - W{window_num}: frames [{start_frame}..{end_frame}] => {start_s:.2f}s to {end_s:.2f}s"
                    )

    def save_calibrated_angles_log(self, output_path: str) -> None:
        """
        Save calibrated angles to a text log file.

        Args:
            output_path: Path to output log file
        """
        if not self.calibrated_angles:
            print("[WARNING] No calibrated angles to save")
            return

        segments_order = [
            ("neck", "NECK"),
            ("torso", "TORSO"),
            ("right_shoulder", "RIGHT SHOULDER"),
            ("left_shoulder", "LEFT SHOULDER"),
            ("right_elbow", "RIGHT ELBOW"),
            ("left_elbow", "LEFT ELBOW"),
            ("right_knee", "RIGHT KNEE"),
            ("left_knee", "LEFT KNEE"),
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("CALIBRATED ANGLES - REBA ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            for segment_key, segment_name in segments_order:
                if segment_key not in self.calibrated_angles:
                    continue

                angles_dict = self.calibrated_angles[segment_key]
                f.write(f"{segment_name} angles (Windows):\n")

                # Process each angle type (alpha, beta, gamma, angle, elevation)
                for angle_name, values in angles_dict.items():
                    formatted_name = f"{angle_name.capitalize()}_{segment_key}"

                    values_str = ", ".join([
                        f"F{i+1}: {val:.2f}°" for i, val in enumerate(values)
                    ])

                    f.write(f"{formatted_name} : [{values_str}]\n")

                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write(f"Total windows analyzed: {len(self.windows_info)}\n")
            f.write("=" * 80 + "\n")

    def save_detailed_analysis_log(self, output_path: str) -> None:
        """
        Save detailed REBA analysis to a text log file.

        Includes:
        - Calibrated angles per window (alpha, beta, gamma for each segment)
        - Table A scores
        - Table B scores
        - Risk labels per window

        Args:
            output_path: Path to output log file
        """
        if not self.calibrated_angles or not self.scores:
            print("[WARNING] No analysis data to save")
            return

        mode_label = "3D" if self.mode == "3d" else "2D"

        segments_order = [
            ("neck", "NECK"),
            ("torso", "TORSO"),
            ("right_shoulder", "RIGHT SHOULDER"),
            ("left_shoulder", "LEFT SHOULDER"),
            ("right_elbow", "RIGHT ELBOW"),
            ("left_elbow", "LEFT ELBOW"),
            ("right_knee", "RIGHT KNEE"),
            ("left_knee", "LEFT KNEE"),
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"REBA {mode_label} ANALYSIS - DETAILED REPORT\n")
            f.write("=" * 80 + "\n\n")

            # === Angles Section ===
            f.write(f"{mode_label} angles (Windows of {self.window_size} frames):\n")
            f.write("-" * 40 + "\n\n")

            for segment_key, segment_name in segments_order:
                if segment_key not in self.calibrated_angles:
                    continue

                angles_dict = self.calibrated_angles[segment_key]
                f.write(f"{mode_label} {segment_name} angles (Windows):\n")

                for angle_name, values in angles_dict.items():
                    formatted_name = f"{angle_name.capitalize()}_{segment_key}"
                    values_str = ", ".join([
                        f"F{i+1}: {val:.2f}°" for i, val in enumerate(values)
                    ])
                    f.write(f"{formatted_name} : [{values_str}]\n")

                f.write("\n")

            # === Table A Section ===
            f.write("-" * 40 + "\n")
            f.write(f"{mode_label} Table A\n")
            scores_a = self.scores.get("score_a", [])
            scores_a_str = ", ".join([
                str(s) if isinstance(s, int) else "N/A" for s in scores_a
            ])
            f.write(f"[{scores_a_str}]\n\n")

            # === Table B Section ===
            f.write(f"{mode_label} Table B\n")
            scores_b = self.scores.get("score_b", [])
            scores_b_str = ", ".join([
                str(s) if isinstance(s, int) else "N/A" for s in scores_b
            ])
            f.write(f"[{scores_b_str}]\n\n")

            # === Table C (Final Score) Section ===
            f.write(f"{mode_label} Table C (Final Score)\n")
            scores_c = self.scores.get("score_c", [])
            scores_c_str = ", ".join([
                str(s) if isinstance(s, (int, float)) else "N/A" for s in scores_c
            ])
            f.write(f"[{scores_c_str}]\n\n")

            # === Risk Labels Section ===
            f.write("-" * 40 + "\n")
            f.write(f"{mode_label} REBA risks:\n")
            risk_labels = self.scores.get("risk_labels", [])
            recalculated = self.scores.get("recalculated", [])

            formatted_labels = []
            for i, label in enumerate(risk_labels):
                tag = "*" if i < len(recalculated) and recalculated[i] else ""
                formatted_labels.append(f"F{i+1}{tag}: {label}")

            f.write("[ " + ", ".join(formatted_labels) + " ]\n\n")

            # === Summary ===
            f.write("=" * 80 + "\n")
            f.write(f"Total windows analyzed: {len(self.windows_info)}\n")
            f.write(f"Computation mode: {mode_label}\n")
            f.write("=" * 80 + "\n")


def assess_video(
    keypoints_path: str,
    output_path: Optional[str] = None,
    neutral_frames: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to perform REBA assessment on a video.

    Runs both 3D and 2D REBA assessments and generates two output files:
    - risk_times_3d.json: Risk intervals based on full 3D nautical angles
    - risk_times_2d.json: Risk intervals based on 2D planar projection

    Args:
        keypoints_path: Path to keypoints_3d.json
        output_path: Path to save risk_times files (optional)
                    If None, saves to a 'risk_times' folder next to keypoints file
                    Note: This will be used as base path, with _3d.json and _2d.json suffixes
        neutral_frames: Optional tuple (start_frame, end_frame) specifying frames
                       where the person is in a neutral standing position for calibration.
                       If None, uses pre-defined static calibration offsets.
        **kwargs: Additional arguments passed to REBAAssessor.analyze()

    Returns:
        Analysis results dictionary (from 3D assessment)
    """
    import json
    from pathlib import Path

    keypoints_file = Path(keypoints_path)

    # Show calibration mode
    if neutral_frames is not None:
        print(
            f"\n[INFO] Custom calibration enabled (frames {neutral_frames[0]}-{neutral_frames[1]})"
        )
    else:
        print("\n[INFO] Static calibration (pre-defined offsets)")

    # Determine output directory
    if output_path is None:
        risk_times_dir = keypoints_file.parent / "risk_times"
        risk_times_dir.mkdir(exist_ok=True)
        output_path_3d = risk_times_dir / "risk_times_3d.json"
        output_path_2d = risk_times_dir / "risk_times_2d.json"
    else:
        # If output_path is provided, derive 3d/2d paths from it
        output_path = Path(output_path)
        if output_path.suffix == ".json":
            base_path = output_path.with_suffix("")
            output_path_3d = Path(f"{base_path}_3d.json")
            output_path_2d = Path(f"{base_path}_2d.json")
        else:
            output_path_3d = output_path / "risk_times_3d.json"
            output_path_2d = output_path / "risk_times_2d.json"

    # === 3D Assessment ===
    print("\n" + "=" * 50)
    print("REBA 3D ANALYSIS")
    print("=" * 50)
    assessor_3d = REBAAssessor(mode="3d", neutral_frames=neutral_frames)
    results_3d = assessor_3d.analyze(keypoints_path, **kwargs)
    assessor_3d.print_summary()

    # Save 3D risk times
    risk_times_3d = assessor_3d.get_risk_times_for_video()
    with open(output_path_3d, "w") as f:
        json.dump(risk_times_3d, f, indent=2)
    print(f"\n[OK] REBA 3D intervals exported to {output_path_3d}")

    # Save detailed analysis log (3D)
    detailed_log_path_3d = keypoints_file.parent / "reba_analysis_3d.log"
    assessor_3d.save_detailed_analysis_log(str(detailed_log_path_3d))
    print(f"[OK] 3D detailed report exported to {detailed_log_path_3d}")

    # === 2D Assessment ===
    print("\n" + "=" * 50)
    print("REBA 2D ANALYSIS")
    print("=" * 50)
    assessor_2d = REBAAssessor(mode="2d", neutral_frames=neutral_frames)
    results_2d = assessor_2d.analyze(keypoints_path, **kwargs)
    assessor_2d.print_summary()

    # Save 2D risk times
    risk_times_2d = assessor_2d.get_risk_times_for_video()
    with open(output_path_2d, "w") as f:
        json.dump(risk_times_2d, f, indent=2)
    print(f"\n[OK] REBA 2D intervals exported to {output_path_2d}")

    # Save detailed analysis log (2D)
    detailed_log_path_2d = keypoints_file.parent / "reba_analysis_2d.log"
    assessor_2d.save_detailed_analysis_log(str(detailed_log_path_2d))
    print(f"[OK] 2D detailed report exported to {detailed_log_path_2d}")

    # Return 3D results as primary (for backward compatibility)
    return results_3d
