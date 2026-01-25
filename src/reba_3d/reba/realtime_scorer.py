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


def calculate_reba_score(calibrated_angles: Dict[str, float]) -> REBAScore:
    """
    Calculate REBA score from calibrated angles.

    Args:
        calibrated_angles: Dictionary of calibrated angles:
            - cou: neck angle
            - epaule_droite/gauche: shoulder angle
            - coude_droit/gauche: elbow angle
            - genou_droit/gauche: knee angle
            - hanche: hip/trunk angle

    Returns:
        REBAScore with all component scores
    """
    score = REBAScore()

    # Get angles with defaults
    neck_angle = calibrated_angles.get('cou', 0)
    trunk_angle = calibrated_angles.get('hanche', 0)  # Using hip as proxy for trunk

    # Use average of left/right for bilateral joints
    shoulder_r = calibrated_angles.get('epaule_droite', 0)
    shoulder_l = calibrated_angles.get('epaule_gauche', 0)
    shoulder_angle = max(shoulder_r, shoulder_l)  # Use worse side

    elbow_r = calibrated_angles.get('coude_droit', 0)
    elbow_l = calibrated_angles.get('coude_gauche', 0)
    elbow_angle = max(elbow_r, elbow_l)

    knee_r = calibrated_angles.get('genou_droit', 0)
    knee_l = calibrated_angles.get('genou_gauche', 0)
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

    # Final score (could add activity score here)
    score.final_score = score.score_c

    # Get risk level
    risk_info = RISK_LEVELS.get(score.final_score, ("high", (0, 0, 255)))
    score.risk_level = risk_info[0]
    score.risk_color = risk_info[1]

    return score


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
