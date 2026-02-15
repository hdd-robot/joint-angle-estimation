"""
Real-time REBA scoring from 2D angles.

Simplified REBA scoring for live video analysis.
Uses calibrated angles to determine risk levels.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Import standard REBA tables from tables.py
from reba_3d.reba.tables import TABLE_A, TABLE_B, TABLE_C


@dataclass
class REBAScore:
    """REBA score with component breakdown."""
    # Individual scores
    neck_score: int = 1
    trunk_score: int = 1
    legs_score: int = 1
    upper_arm_score: int = 1
    lower_arm_score: int = 1

    # Table scores
    score_a: int = 1  # Trunk + Neck + Legs
    score_b: int = 1  # Upper arm + Lower arm + Wrist
    score_c: int = 1  # Combined

    # Final score
    final_score: int = 1
    risk_level: str = "negligible"
    risk_color: Tuple[int, int, int] = (0, 200, 0)  # BGR Green

    def __str__(self) -> str:
        return f"REBA {self.final_score} ({self.risk_level})"

# Risk levels and colors (BGR for OpenCV)
RISK_LEVELS = {
    1: ("negligible", (0, 200, 0)),      # Green
    2: ("low", (200, 150, 0)),           # Cyan
    3: ("low", (200, 150, 0)),
    4: ("medium", (0, 150, 255)),        # Orange
    5: ("medium", (0, 150, 255)),
    6: ("medium", (0, 150, 255)),
    7: ("high", (0, 0, 255)),            # Red
    8: ("high", (0, 0, 255)),
    9: ("high", (0, 0, 255)),
    10: ("high", (0, 0, 255)),
    11: ("very high", (136, 47, 99)),    # Purple
    12: ("very high", (136, 47, 99)),
}


def score_neck_angle(angle: float) -> int:
    """
    Score neck flexion/extension.

    REBA standard:
    - 0-20° flexion: Score 1
    - >20° flexion or any extension: Score 2

    Note: Score 3 comes from +1 adjustment for twist/side bend (not handled here)

    Args:
        angle: Calibrated neck angle (0° = neutral)

    Returns:
        REBA neck score (1-2)
    """
    angle = abs(angle)
    if angle <= 20:
        return 1  # 0-20° flexion
    else:
        return 2  # >20° flexion or extension


def score_trunk_angle(angle: float) -> int:
    """
    Score trunk flexion/extension.

    Args:
        angle: Calibrated trunk angle (0° = neutral/upright)

    Returns:
        REBA trunk score (1-5)
    """
    angle = abs(angle)
    if angle <= 5:
        return 1  # Upright
    elif angle <= 20:
        return 2  # 0-20° flexion
    elif angle <= 60:
        return 3  # 20-60° flexion
    else:
        return 4  # >60° flexion


def score_legs_angle(angle: float) -> int:
    """
    Score legs/knee flexion.

    REBA standard (simplified - assumes bilateral weight bearing):
    - <30° flexion: Score 1 (base 1 + 0 adjustment)
    - 30-60° flexion: Score 2 (base 1 + 1 adjustment)
    - >60° flexion: Score 3 (base 1 + 2 adjustment)

    Note: For unilateral weight bearing, add +1 to these scores.

    Args:
        angle: Calibrated knee angle (0° = straight)

    Returns:
        REBA legs score (1-3, or 4 with unilateral adjustment)
    """
    angle = abs(angle)
    if angle <= 30:
        return 1  # Bilateral, no flexion adjustment
    elif angle <= 60:
        return 2  # Bilateral + 30-60° flexion
    else:
        return 3  # Bilateral + >60° flexion


def score_upper_arm_angle(angle: float) -> int:
    """
    Score upper arm (shoulder) position.

    Args:
        angle: Calibrated shoulder angle (0° = arm at side)

    Returns:
        REBA upper arm score (1-6)
    """
    angle = abs(angle)
    if angle <= 20:
        return 1  # 0-20° flexion/extension
    elif angle <= 45:
        return 2  # 20-45° flexion, >20° extension
    elif angle <= 90:
        return 3  # 45-90° flexion
    else:
        return 4  # >90° flexion


def score_lower_arm_angle(angle: float) -> int:
    """
    Score lower arm (elbow) position.

    Args:
        angle: Calibrated elbow angle (0° = straight arm, positive = flexion)

    Returns:
        REBA lower arm score (1-2)
    """
    # Elbow angle: 0° = fully extended, increases with flexion
    # Neutral is around 60-100° flexion
    if 60 <= angle <= 100:
        return 1  # 60-100° flexion
    else:
        return 2  # <60° or >100°


def get_table_a_score(trunk: int, neck: int, legs: int) -> int:
    """Look up score in Table A (trunk, neck, legs order)."""
    # Clamp values
    trunk = max(1, min(5, trunk))
    neck = max(1, min(3, neck))
    legs = max(1, min(4, legs))
    return TABLE_A.get((trunk, neck, legs), 5)


def get_table_b_score(upper_arm: int, lower_arm: int, wrist: int = 1) -> int:
    """Look up score in Table B."""
    # Clamp values
    upper_arm = max(1, min(6, upper_arm))
    lower_arm = max(1, min(2, lower_arm))
    wrist = max(1, min(3, wrist))
    return TABLE_B.get((upper_arm, lower_arm, wrist), 5)


def get_table_c_score(score_a: int, score_b: int) -> int:
    """Look up score in Table C."""
    # Clamp values
    score_a = max(1, min(12, score_a))
    score_b = max(1, min(12, score_b))
    return TABLE_C.get((score_a, score_b), 8)


def calculate_reba_score_simple(calibrated_angles: Dict[str, float]) -> REBAScore:
    """
    Calculate REBA score from calibrated angles (simple format - legacy).

    Args:
        calibrated_angles: Dictionary of calibrated angles:
            - neck: neck angle
            - right_shoulder/gauche: shoulder angle
            - right_elbow/gauche: elbow angle
            - right_knee/gauche: knee angle
            - hanche: hip/trunk angle

    Returns:
        REBAScore with all component scores
    """
    score = REBAScore()

    # Get angles with defaults
    neck_angle = calibrated_angles.get('neck', 0)
    trunk_angle = calibrated_angles.get('hanche', 0)  # Using hip as proxy for trunk

    # Use average of left/right for bilateral joints
    shoulder_r = calibrated_angles.get('right_shoulder', 0)
    shoulder_l = calibrated_angles.get('left_shoulder', 0)
    shoulder_angle = max(shoulder_r, shoulder_l)  # Use worse side

    elbow_r = calibrated_angles.get('right_elbow', 0)
    elbow_l = calibrated_angles.get('left_elbow', 0)
    elbow_angle = max(elbow_r, elbow_l)

    knee_r = calibrated_angles.get('right_knee', 0)
    knee_l = calibrated_angles.get('left_knee', 0)
    knee_angle = max(knee_r, knee_l)

    # Calculate individual scores
    score.neck_score = score_neck_angle(neck_angle)
    score.trunk_score = score_trunk_angle(trunk_angle)
    score.legs_score = score_legs_angle(knee_angle)
    score.upper_arm_score = score_upper_arm_angle(shoulder_angle)
    score.lower_arm_score = score_lower_arm_angle(elbow_angle)

    # Calculate table scores (trunk, neck, legs order for Table A)
    score.score_a = get_table_a_score(score.trunk_score, score.neck_score, score.legs_score)
    score.score_b = get_table_b_score(score.upper_arm_score, score.lower_arm_score, 1)
    score.score_c = get_table_c_score(score.score_a, score.score_b)

    # Final score (neckld add activity score here)
    score.final_score = score.score_c

    # Get risk level
    risk_info = RISK_LEVELS.get(score.final_score, ("high", (0, 0, 255)))
    score.risk_level = risk_info[0]
    score.risk_color = risk_info[1]

    return score


def calculate_reba_score_nautical(
    calibrated_angles: Dict[str, Dict[str, float]]
) -> REBAScore:
    """
    Calculate REBA score from nautical angles (nested format) with malus.

    Uses full 3D nautical angles (alpha, beta, gamma) to detect rotations
    and lateral bending, applying REBA malus scores accordingly.

    Args:
        calibrated_angles: Nested dictionary of calibrated angles:
            {
                "neck": {"alpha": float, "beta": float, "gamma": float},
                "torso": {"alpha": float, "beta": float, "gamma": float},
                "right_shoulder": {"alpha": float, "beta": float, "gamma": float, "elevation": float},
                ...
            }

    Returns:
        REBAScore with all component scores including malus adjustments
    """
    import numpy as np

    score = REBAScore()

    # Neck scoring with malus
    if "neck" in calibrated_angles:
        alpha = calibrated_angles["neck"].get("alpha", 0)
        beta = calibrated_angles["neck"].get("beta", 0)
        gamma = calibrated_angles["neck"].get("gamma", 0)

        # Skip NaN values
        if not (np.isnan(alpha) or np.isnan(beta) or np.isnan(gamma)):
            # Detect 2D profile mode: if alpha and beta are 0, use gamma for flexion
            is_2d_mode = (abs(alpha) < 0.1 and abs(beta) < 0.1)
            flexion_angle = gamma if is_2d_mode else alpha

            # Base score from flexion angle
            if abs(flexion_angle) > 20:
                score.neck_score = 2
            else:
                score.neck_score = 1

            # Malus for lateral bend (beta) - only in 3D mode
            if not is_2d_mode and 15 <= abs(beta) <= 30:
                score.neck_score += 1

            # Malus for rotation (gamma) - only in 3D mode
            if not is_2d_mode and abs(gamma) > 25:
                score.neck_score += 1

    # Torso scoring with malus
    if "torso" in calibrated_angles:
        alpha = calibrated_angles["torso"].get("alpha", 0)
        beta = calibrated_angles["torso"].get("beta", 0)
        gamma = calibrated_angles["torso"].get("gamma", 0)

        if not (np.isnan(alpha) or np.isnan(beta) or np.isnan(gamma)):
            # Detect 2D mode: if alpha and beta are 0, use gamma for flexion
            is_2d_mode = (abs(alpha) < 0.1 and abs(beta) < 0.1)
            flexion_angle = gamma if is_2d_mode else alpha

            # Base score from flexion angle
            if abs(flexion_angle) > 60:
                score.trunk_score = 4
            elif abs(flexion_angle) > 20:
                score.trunk_score = 3
            elif abs(flexion_angle) > 0:
                score.trunk_score = 2
            else:
                score.trunk_score = 1

            # Malus for lateral bend - only in 3D mode
            if not is_2d_mode and beta < -18:
                score.trunk_score += 1

            # Malus for rotation - only in 3D mode
            if not is_2d_mode and -15 <= gamma < -9:
                score.trunk_score += 1

    # Shoulder scoring (use worse side)
    shoulder_scores = []

    if "right_shoulder" in calibrated_angles:
        alpha_r = calibrated_angles["right_shoulder"].get("alpha", 0)
        beta_r = calibrated_angles["right_shoulder"].get("beta", 0)
        gamma_r = calibrated_angles["right_shoulder"].get("gamma", 0)
        elevation_r = calibrated_angles["right_shoulder"].get("elevation", 0)

        if not np.isnan(alpha_r):
            # Detect 2D mode: if alpha and beta are 0, use gamma
            is_2d_mode = (abs(alpha_r) < 0.1 and abs(beta_r) < 0.1)
            shoulder_angle = gamma_r if is_2d_mode else alpha_r

            # Base score from shoulder angle
            if abs(shoulder_angle) > 90:
                s = 4
            elif abs(shoulder_angle) > 45:
                s = 3
            elif abs(shoulder_angle) > 20:
                s = 2
            else:
                s = 1

            # Malus for shoulder elevation
            if not np.isnan(elevation_r) and elevation_r < -5:
                s += 1

            shoulder_scores.append(s)

    if "left_shoulder" in calibrated_angles:
        alpha_l = calibrated_angles["left_shoulder"].get("alpha", 0)
        beta_l = calibrated_angles["left_shoulder"].get("beta", 0)
        gamma_l = calibrated_angles["left_shoulder"].get("gamma", 0)
        elevation_l = calibrated_angles["left_shoulder"].get("elevation", 0)

        if not np.isnan(alpha_l):
            # Detect 2D mode: if alpha and beta are 0, use gamma
            is_2d_mode = (abs(alpha_l) < 0.1 and abs(beta_l) < 0.1)
            shoulder_angle = gamma_l if is_2d_mode else alpha_l

            # Base score from shoulder angle
            if abs(shoulder_angle) > 90:
                s = 4
            elif abs(shoulder_angle) > 45:
                s = 3
            elif abs(shoulder_angle) > 20:
                s = 2
            else:
                s = 1

            # Malus for shoulder elevation
            if not np.isnan(elevation_l) and elevation_l < -5:
                s += 1

            shoulder_scores.append(s)

    if shoulder_scores:
        score.upper_arm_score = max(shoulder_scores)  # Use worse side

    # Elbow scoring (use worse side)
    elbow_angles = []

    if "right_elbow" in calibrated_angles:
        angle_r = calibrated_angles["right_elbow"].get("angle", 0)
        if not np.isnan(angle_r):
            elbow_angles.append(angle_r)

    if "left_elbow" in calibrated_angles:
        angle_l = calibrated_angles["left_elbow"].get("angle", 0)
        if not np.isnan(angle_l):
            elbow_angles.append(angle_l)

    if elbow_angles:
        elbow_angle = max(elbow_angles)
        # Score based on angle range
        if 60 <= elbow_angle <= 100:
            score.lower_arm_score = 1
        else:
            score.lower_arm_score = 2

    # Knee/leg scoring (use worse side)
    knee_angles = []

    if "right_knee" in calibrated_angles:
        angle_r = calibrated_angles["right_knee"].get("angle", 0)
        if not np.isnan(angle_r):
            knee_angles.append(angle_r)

    if "left_knee" in calibrated_angles:
        angle_l = calibrated_angles["left_knee"].get("angle", 0)
        if not np.isnan(angle_l):
            knee_angles.append(angle_l)

    if knee_angles:
        knee_angle = max(knee_angles)
        # Assume bilateral weight bearing
        if abs(knee_angle) <= 30:
            score.legs_score = 1
        elif abs(knee_angle) <= 60:
            score.legs_score = 2
        else:
            score.legs_score = 3

    # Calculate table scores
    score.score_a = get_table_a_score(score.trunk_score, score.neck_score, score.legs_score)
    score.score_b = get_table_b_score(score.upper_arm_score, score.lower_arm_score, 1)
    score.score_c = get_table_c_score(score.score_a, score.score_b)

    # Final score
    score.final_score = score.score_c

    # Get risk level
    risk_info = RISK_LEVELS.get(score.final_score, ("high", (0, 0, 255)))
    score.risk_level = risk_info[0]
    score.risk_color = risk_info[1]

    return score


def calculate_reba_score(calibrated_angles) -> REBAScore:
    """
    Calculate REBA score from calibrated angles (auto-detect format).

    Automatically detects whether the input is in simple format (Dict[str, float])
    or nested nautical format (Dict[str, Dict[str, float]]) and calls the
    appropriate scoring function.

    Args:
        calibrated_angles: Dictionary of calibrated angles in either format

    Returns:
        REBAScore with all component scores
    """
    # Detect format by checking the first value
    if not calibrated_angles:
        return REBAScore()

    first_key = next(iter(calibrated_angles))
    first_value = calibrated_angles[first_key]

    # If first value is a dict, it's nested nautical format
    if isinstance(first_value, dict):
        return calculate_reba_score_nautical(calibrated_angles)
    else:
        # Simple flat format (legacy)
        return calculate_reba_score_simple(calibrated_angles)


class RealtimeREBAScorer:
    """
    Real-time REBA scorer with smoothing.

    Maintains a buffer of recent scores for smoothing.
    """

    def __init__(self, buffer_size: int = 10):
        """
        Initialize scorer.

        Args:
            buffer_size: Number of recent scores to average
        """
        self.buffer_size = buffer_size
        self.score_buffer: list = []
        self.current_score: Optional[REBAScore] = None

        # Additional REBA scores (malus)
        self.load_score: int = 0      # 0=<5kg, 1=5-10kg, 2=>10kg, 3=>10kg+shock
        self.coupling_score: int = 0  # 0=good, 1=fair, 2=poor, 3=unacceptable
        self.activity_score: int = 0  # 0-3 based on activity conditions

    def set_load_score(self, value: int) -> None:
        """Set load/force score (0-3)."""
        self.load_score = max(0, min(3, value))

    def set_coupling_score(self, value: int) -> None:
        """Set coupling/grip score (0-3)."""
        self.coupling_score = max(0, min(3, value))

    def set_activity_score(self, value: int) -> None:
        """Set activity score (0-3)."""
        self.activity_score = max(0, min(3, value))

    def update(self, calibrated_angles: Dict[str, float]) -> REBAScore:
        """
        Update scorer with new angles.

        Args:
            calibrated_angles: Dictionary of calibrated angles

        Returns:
            Current (smoothed) REBA score
        """
        # Calculate raw score
        raw_score = calculate_reba_score(calibrated_angles)

        # Apply load/coupling/activity malus
        # Load is added to Score A, Coupling to Score B
        new_score_a = min(raw_score.score_a + self.load_score, 12)
        new_score_b = min(raw_score.score_b + self.coupling_score, 12)

        # Recalculate Score C with adjusted A and B
        new_score_c = get_table_c_score(new_score_a, new_score_b)

        # Add activity score to final
        final_with_malus = min(new_score_c + self.activity_score, 15)

        # Add to buffer (use the score with malus applied)
        self.score_buffer.append(final_with_malus)
        if len(self.score_buffer) > self.buffer_size:
            self.score_buffer.pop(0)

        # Smooth the final score
        avg_score = round(sum(self.score_buffer) / len(self.score_buffer))
        avg_score = max(1, min(15, avg_score))

        # Update the score with smoothed value
        raw_score.score_a = new_score_a
        raw_score.score_b = new_score_b
        raw_score.score_c = new_score_c
        raw_score.final_score = avg_score
        risk_info = RISK_LEVELS.get(min(avg_score, 12), ("very high", (136, 47, 99)))
        raw_score.risk_level = risk_info[0]
        raw_score.risk_color = risk_info[1]

        self.current_score = raw_score
        return raw_score

    def reset(self):
        """Reset the score buffer."""
        self.score_buffer.clear()
        self.current_score = None
