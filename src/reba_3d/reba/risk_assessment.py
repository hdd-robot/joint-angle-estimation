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
)
from reba_3d.reba.calibration import calibrate_all_angles
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
    ):
        """
        Initialize REBA assessor.

        Args:
            window_size: Number of frames per analysis window (default: 30)
            poly_order: Polynomial order for smoothing (default: 2)
            fps: Video frames per second (default: 15)
            feet_threshold: Threshold for feet contact detection (default: 0.10)
        """
        self.window_size = window_size
        self.poly_order = poly_order
        self.fps = fps
        self.feet_threshold = feet_threshold

        # Analysis results
        self.windows_info: List[Dict] = []
        self.raw_angles: Dict[str, Dict[str, List[float]]] = {}
        self.calibrated_angles: Dict[str, Dict[str, List[float]]] = {}
        self.scores: Dict[str, List] = {}
        self.risk_times: Dict[str, List] = {}

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

        print(f"[OK] {len(pertinent_frames['face'])} frames de vue de face détectées.")
        print(f"[OK] {len(pertinent_frames['profil_droit'])} frames de profil droit détectées.")
        print(f"[OK] {len(pertinent_frames['profil_gauche'])} frames de profil gauche détectées.")

        # First pass: compute smoothed positions and raw angles
        self._first_pass(df)

        # Calculate ankle means for feet contact
        r_ankle_mean, l_ankle_mean = self._compute_ankle_means()

        # Second pass: compute contact states and finalize angles
        self._second_pass(r_ankle_mean, l_ankle_mean)

        # Calibrate angles
        self.calibrated_angles = calibrate_all_angles(self.raw_angles)

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
            frames_in_window = df_window['frame'].tolist()

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

    def _compute_ankle_means(self) -> Tuple[float, float]:
        """Compute mean ankle positions for feet contact detection."""
        r_valid = [v for v in self._r_ankle_list if not np.isnan(v)]
        l_valid = [v for v in self._l_ankle_list if not np.isnan(v)]

        r_mean = np.mean(r_valid) if r_valid else 0.0
        l_mean = np.mean(l_valid) if l_valid else 0.0

        return r_mean, l_mean

    def _second_pass(self, r_ankle_mean: float, l_ankle_mean: float) -> None:
        """Second pass: compute angles and contact states."""
        # Initialize angle storage
        self.raw_angles = {
            "cou": {"alpha": [], "beta": [], "gamma": []},
            "buste": {"alpha": [], "beta": [], "gamma": []},
            "epaule_droite": {"alpha": [], "beta": [], "gamma": [], "elevation": []},
            "epaule_gauche": {"alpha": [], "beta": [], "gamma": [], "elevation": []},
            "coude_droit": {"angle": []},
            "coude_gauche": {"angle": []},
            "genou_droit": {"angle": []},
            "genou_gauche": {"angle": []},
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

            # Neck angles
            neck_angles = compute_neck_angles(block_positions)
            if neck_angles:
                self.raw_angles["cou"]["alpha"].append(neck_angles[0])
                self.raw_angles["cou"]["beta"].append(neck_angles[1])
                self.raw_angles["cou"]["gamma"].append(neck_angles[2])
            else:
                self.raw_angles["cou"]["alpha"].append(np.nan)
                self.raw_angles["cou"]["beta"].append(np.nan)
                self.raw_angles["cou"]["gamma"].append(np.nan)

            # Torso angles
            torso_angles = compute_torso_angles(block_positions)
            if torso_angles:
                self.raw_angles["buste"]["alpha"].append(torso_angles[0])
                self.raw_angles["buste"]["beta"].append(torso_angles[1])
                self.raw_angles["buste"]["gamma"].append(torso_angles[2])
            else:
                self.raw_angles["buste"]["alpha"].append(np.nan)
                self.raw_angles["buste"]["beta"].append(np.nan)
                self.raw_angles["buste"]["gamma"].append(np.nan)

            # Right shoulder
            shoulder_r = compute_shoulder_angles_right(block_positions)
            elev_r = compute_shoulder_elevation_right(block_positions)
            if shoulder_r:
                self.raw_angles["epaule_droite"]["alpha"].append(shoulder_r[0])
                self.raw_angles["epaule_droite"]["beta"].append(shoulder_r[1])
                self.raw_angles["epaule_droite"]["gamma"].append(shoulder_r[2])
            else:
                self.raw_angles["epaule_droite"]["alpha"].append(np.nan)
                self.raw_angles["epaule_droite"]["beta"].append(np.nan)
                self.raw_angles["epaule_droite"]["gamma"].append(np.nan)
            self.raw_angles["epaule_droite"]["elevation"].append(
                elev_r if elev_r else np.nan
            )

            # Left shoulder
            shoulder_l = compute_shoulder_angles_left(block_positions)
            elev_l = compute_shoulder_elevation_left(block_positions)
            if shoulder_l:
                self.raw_angles["epaule_gauche"]["alpha"].append(shoulder_l[0])
                self.raw_angles["epaule_gauche"]["beta"].append(shoulder_l[1])
                self.raw_angles["epaule_gauche"]["gamma"].append(shoulder_l[2])
            else:
                self.raw_angles["epaule_gauche"]["alpha"].append(np.nan)
                self.raw_angles["epaule_gauche"]["beta"].append(np.nan)
                self.raw_angles["epaule_gauche"]["gamma"].append(np.nan)
            self.raw_angles["epaule_gauche"]["elevation"].append(
                elev_l if elev_l else np.nan
            )

            # Elbows
            elbow_r = compute_elbow_angle_right(block_positions)
            elbow_l = compute_elbow_angle_left(block_positions)
            self.raw_angles["coude_droit"]["angle"].append(elbow_r if elbow_r else np.nan)
            self.raw_angles["coude_gauche"]["angle"].append(elbow_l if elbow_l else np.nan)

            # Knees
            knee_r = compute_knee_angle_right(block_positions)
            knee_l = compute_knee_angle_left(block_positions)
            self.raw_angles["genou_droit"]["angle"].append(knee_r if knee_r else np.nan)
            self.raw_angles["genou_gauche"]["angle"].append(knee_l if knee_l else np.nan)

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
        scores_cou = []
        scores_buste = []
        scores_epaule = []
        scores_coude = []
        scores_genou = []
        scores_poignet = []

        malus_cou_list = []
        malus_buste_list = []
        malus_epaule_list = []
        malus_contact_list = []

        for i in range(num_windows):
            # Neck
            a_cou = cal["cou"]["alpha"][i] if i < len(cal["cou"]["alpha"]) else np.nan
            b_cou = cal["cou"]["beta"][i] if i < len(cal["cou"]["beta"]) else np.nan
            g_cou = cal["cou"]["gamma"][i] if i < len(cal["cou"]["gamma"]) else np.nan
            s_cou, m_cou = score_neck(a_cou, b_cou, g_cou)
            scores_cou.append(s_cou)
            malus_cou_list.append(m_cou)

            # Torso
            a_buste = cal["buste"]["alpha"][i] if i < len(cal["buste"]["alpha"]) else np.nan
            b_buste = cal["buste"]["beta"][i] if i < len(cal["buste"]["beta"]) else np.nan
            g_buste = cal["buste"]["gamma"][i] if i < len(cal["buste"]["gamma"]) else np.nan
            s_buste, m_buste = score_torso(a_buste, b_buste, g_buste)
            scores_buste.append(s_buste)
            malus_buste_list.append(m_buste)

            # Shoulder
            a_ep_d = cal["epaule_droite"]["alpha"][i] if i < len(cal["epaule_droite"]["alpha"]) else np.nan
            a_ep_g = cal["epaule_gauche"]["alpha"][i] if i < len(cal["epaule_gauche"]["alpha"]) else np.nan
            elev_d = cal["epaule_droite"]["elevation"][i] if i < len(cal["epaule_droite"]["elevation"]) else np.nan
            elev_g = cal["epaule_gauche"]["elevation"][i] if i < len(cal["epaule_gauche"]["elevation"]) else np.nan
            s_ep, m_ep_d, m_ep_g = score_shoulder(a_ep_d, a_ep_g, elev_d, elev_g)
            scores_epaule.append(s_ep)
            malus_epaule_list.append(max(m_ep_d, m_ep_g))

            # Elbow
            coude_d = cal["coude_droit"]["angle"][i] if i < len(cal["coude_droit"]["angle"]) else np.nan
            coude_g = cal["coude_gauche"]["angle"][i] if i < len(cal["coude_gauche"]["angle"]) else np.nan
            scores_coude.append(score_elbow(coude_d, coude_g))

            # Knee
            genou_d = cal["genou_droit"]["angle"][i] if i < len(cal["genou_droit"]["angle"]) else np.nan
            genou_g = cal["genou_gauche"]["angle"][i] if i < len(cal["genou_gauche"]["angle"]) else np.nan
            scores_genou.append(score_knee(genou_d, genou_g))

            # Wrist (default)
            scores_poignet.append(score_wrist())

            # Feet contact
            contact = self._contact_results[i] if i < len(self._contact_results) else "404"
            malus_contact_list.append(score_feet_contact(contact))

        # Total scores
        total_cou = [s + m for s, m in zip(scores_cou, malus_cou_list)]
        total_buste = [s + m for s, m in zip(scores_buste, malus_buste_list)]
        total_genou = [s + m for s, m in zip(scores_genou, malus_contact_list)]
        total_epaule = [s + m for s, m in zip(scores_epaule, malus_epaule_list)]

        # Table A and B scores
        scores_a = []
        scores_b = []
        for i in range(num_windows):
            s_a = calculate_score_a(total_buste[i], total_cou[i], total_genou[i])
            s_b = calculate_score_b(total_epaule[i], scores_coude[i], scores_poignet[i])
            scores_a.append(s_a)
            scores_b.append(s_b)

        # Final scores with malus
        scores_c = []
        risk_labels = []
        recalculated_flags = []

        seconds_per_window = self.window_size / self.fps
        self.risk_times = {
            "sans risque": [],
            "risque moyen": [],
            "risque élevé": [],
            "très élevé": [],
        }

        for i in range(num_windows):
            if isinstance(scores_a[i], int) and isinstance(scores_b[i], int):
                score_c = calculate_score_c(scores_a[i], scores_b[i])
                label = score_to_risk_label(score_c)

                # Apply malus for risk levels
                if label in ["risque faible", "risque moyen", "risque élevé", "très élevé"]:
                    final_score, recalculated = apply_activity_malus(
                        score_c, scores_a[i], scores_b[i],
                        load_malus, coupling_malus, activity_malus
                    )
                    score_c = final_score
                    label = score_to_risk_label(score_c)
                else:
                    recalculated = False
            else:
                score_c = "Invalide"
                label = "invalide"
                recalculated = False

            scores_c.append(score_c)
            risk_labels.append(label)
            recalculated_flags.append(recalculated)

            # Record risk times
            if label in self.risk_times:
                start_time = round(i * seconds_per_window, 2)
                end_time = round((i + 1) * seconds_per_window, 2)
                self.risk_times[label].append((i + 1, start_time, end_time))

        # Store results
        self.scores = {
            "cou": scores_cou,
            "buste": scores_buste,
            "epaule": scores_epaule,
            "coude": scores_coude,
            "genou": scores_genou,
            "poignet": scores_poignet,
            "total_cou": total_cou,
            "total_buste": total_buste,
            "total_genou": total_genou,
            "total_epaule": total_epaule,
            "score_a": scores_a,
            "score_b": scores_b,
            "score_c": scores_c,
            "risk_labels": risk_labels,
            "recalculated": recalculated_flags,
        }

    def _build_results(self) -> Dict[str, Any]:
        """Build final results dictionary."""
        return {
            "windows_info": self.windows_info,
            "scores": self.scores,
            "risk_times": self.risk_times,
            "calibrated_angles": self.calibrated_angles,
            "num_windows": len(self.windows_info),
        }

    def get_risk_times_for_video(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get risk times formatted for video annotation.

        Returns:
            Dictionary mapping English risk labels to time intervals
        """
        label_map = {
            "sans risque": "negligible risk",
            "risque moyen": "medium risk",
            "risque élevé": "high risk",
            "très élevé": "very high risk",
        }

        result = {}
        for fr_label, intervals in self.risk_times.items():
            en_label = label_map.get(fr_label)
            if en_label:
                result[en_label] = [(start, end) for _, start, end in intervals]

        return result

    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        from reba_3d.config.settings import ANSI_COLORS

        print("\n[STATS] Risques d'après REBA:")
        for i, (label, recalc) in enumerate(
            zip(self.scores["risk_labels"], self.scores["recalculated"]),
            start=1
        ):
            tag = "*" if recalc else ""
            color = ANSI_COLORS.get(label, ANSI_COLORS["invalide"])
            reset = ANSI_COLORS["reset"]
            print(f"  F{i}{tag}: {color}{label}{reset}")

        print("\n🕒 Intervalles temporels à surveiller:")
        for risk_level, intervals in self.risk_times.items():
            if intervals:
                color = ANSI_COLORS.get(risk_level, ANSI_COLORS["invalide"])
                reset = ANSI_COLORS["reset"]
                print(f"\n{color}{risk_level}{reset} détecté aux instants:")
                for fenetre_num, start, end in intervals:
                    print(f"  - F{fenetre_num}: de {start:.2f}s à {end:.2f}s")


def assess_video(
    keypoints_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to perform REBA assessment on a video.

    Args:
        keypoints_path: Path to keypoints_3d.json
        output_path: Path to save risk_times.json (optional)
        **kwargs: Additional arguments passed to REBAAssessor.analyze()

    Returns:
        Analysis results dictionary
    """
    assessor = REBAAssessor()
    results = assessor.analyze(keypoints_path, **kwargs)
    assessor.print_summary()

    if output_path:
        import json
        risk_times_video = assessor.get_risk_times_for_video()
        with open(output_path, 'w') as f:
            json.dump(risk_times_video, f, indent=2)
        print(f"\n[OK] Intervalles REBA exportés vers {output_path}")

    return results
