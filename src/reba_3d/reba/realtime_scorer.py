"""
Real-time REBA scoring from 2D angles.

Simplified REBA scoring for live video analysis.
Uses calibrated angles to determine risk levels.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass


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
    score_a: int = 1  # Neck + Trunk + Legs
    score_b: int = 1  # Upper arm + Lower arm + Wrist
    score_c: int = 1  # Combined

    # Final score
    final_score: int = 1
    risk_level: str = "negligible"
    risk_color: Tuple[int, int, int] = (0, 200, 0)  # BGR Green

    def __str__(self) -> str:
        return f"REBA {self.final_score} ({self.risk_level})"


# REBA Table A: Neck x Trunk x Legs
# Simplified version - returns approximate score
TABLE_A = {
    # (neck, trunk, legs) -> score
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 1, 3): 3, (1, 1, 4): 4,
    (1, 2, 1): 2, (1, 2, 2): 3, (1, 2, 3): 4, (1, 2, 4): 5,
    (1, 3, 1): 3, (1, 3, 2): 4, (1, 3, 3): 5, (1, 3, 4): 6,
    (1, 4, 1): 4, (1, 4, 2): 5, (1, 4, 3): 6, (1, 4, 4): 7,
    (1, 5, 1): 5, (1, 5, 2): 6, (1, 5, 3): 7, (1, 5, 4): 8,
    (2, 1, 1): 2, (2, 1, 2): 3, (2, 1, 3): 4, (2, 1, 4): 5,
    (2, 2, 1): 3, (2, 2, 2): 4, (2, 2, 3): 5, (2, 2, 4): 6,
    (2, 3, 1): 4, (2, 3, 2): 5, (2, 3, 3): 6, (2, 3, 4): 7,
    (2, 4, 1): 5, (2, 4, 2): 6, (2, 4, 3): 7, (2, 4, 4): 8,
    (2, 5, 1): 6, (2, 5, 2): 7, (2, 5, 3): 8, (2, 5, 4): 9,
    (3, 1, 1): 3, (3, 1, 2): 4, (3, 1, 3): 5, (3, 1, 4): 6,
    (3, 2, 1): 4, (3, 2, 2): 5, (3, 2, 3): 6, (3, 2, 4): 7,
    (3, 3, 1): 5, (3, 3, 2): 6, (3, 3, 3): 7, (3, 3, 4): 8,
    (3, 4, 1): 6, (3, 4, 2): 7, (3, 4, 3): 8, (3, 4, 4): 9,
    (3, 5, 1): 7, (3, 5, 2): 8, (3, 5, 3): 9, (3, 5, 4): 9,
}

# REBA Table B: Upper Arm x Lower Arm x Wrist
TABLE_B = {
    # (upper_arm, lower_arm, wrist) -> score
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 1, 3): 2,
    (1, 2, 1): 1, (1, 2, 2): 2, (1, 2, 3): 3,
    (2, 1, 1): 1, (2, 1, 2): 2, (2, 1, 3): 3,
    (2, 2, 1): 2, (2, 2, 2): 3, (2, 2, 3): 4,
    (3, 1, 1): 3, (3, 1, 2): 4, (3, 1, 3): 5,
    (3, 2, 1): 4, (3, 2, 2): 5, (3, 2, 3): 5,
    (4, 1, 1): 4, (4, 1, 2): 5, (4, 1, 3): 5,
    (4, 2, 1): 5, (4, 2, 2): 6, (4, 2, 3): 7,
    (5, 1, 1): 6, (5, 1, 2): 7, (5, 1, 3): 8,
    (5, 2, 1): 7, (5, 2, 2): 8, (5, 2, 3): 8,
    (6, 1, 1): 7, (6, 1, 2): 8, (6, 1, 3): 8,
    (6, 2, 1): 8, (6, 2, 2): 9, (6, 2, 3): 9,
}

# REBA Table C: Score A x Score B
TABLE_C = {
    (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 2, (1, 5): 3, (1, 6): 3, (1, 7): 4, (1, 8): 5, (1, 9): 6, (1, 10): 7, (1, 11): 7, (1, 12): 7,
    (2, 1): 1, (2, 2): 2, (2, 3): 2, (2, 4): 3, (2, 5): 4, (2, 6): 4, (2, 7): 5, (2, 8): 6, (2, 9): 6, (2, 10): 7, (2, 11): 7, (2, 12): 8,
    (3, 1): 2, (3, 2): 3, (3, 3): 3, (3, 4): 3, (3, 5): 4, (3, 6): 5, (3, 7): 6, (3, 8): 7, (3, 9): 7, (3, 10): 8, (3, 11): 8, (3, 12): 8,
    (4, 1): 3, (4, 2): 4, (4, 3): 4, (4, 4): 4, (4, 5): 5, (4, 6): 6, (4, 7): 7, (4, 8): 8, (4, 9): 8, (4, 10): 9, (4, 11): 9, (4, 12): 9,
    (5, 1): 4, (5, 2): 4, (5, 3): 4, (5, 4): 5, (5, 5): 6, (5, 6): 7, (5, 7): 8, (5, 8): 8, (5, 9): 9, (5, 10): 9, (5, 11): 9, (5, 12): 9,
    (6, 1): 6, (6, 2): 6, (6, 3): 6, (6, 4): 7, (6, 5): 8, (6, 6): 8, (6, 7): 9, (6, 8): 9, (6, 9): 10, (6, 10): 10, (6, 11): 10, (6, 12): 10,
    (7, 1): 7, (7, 2): 7, (7, 3): 7, (7, 4): 8, (7, 5): 9, (7, 6): 9, (7, 7): 9, (7, 8): 10, (7, 9): 10, (7, 10): 11, (7, 11): 11, (7, 12): 11,
    (8, 1): 8, (8, 2): 8, (8, 3): 8, (8, 4): 9, (8, 5): 10, (8, 6): 10, (8, 7): 10, (8, 8): 10, (8, 9): 10, (8, 10): 11, (8, 11): 11, (8, 12): 11,
    (9, 1): 9, (9, 2): 9, (9, 3): 9, (9, 4): 10, (9, 5): 10, (9, 6): 10, (9, 7): 11, (9, 8): 11, (9, 9): 11, (9, 10): 12, (9, 11): 12, (9, 12): 12,
    (10, 1): 10, (10, 2): 10, (10, 3): 10, (10, 4): 11, (10, 5): 11, (10, 6): 11, (10, 7): 11, (10, 8): 12, (10, 9): 12, (10, 10): 12, (10, 11): 12, (10, 12): 12,
    (11, 1): 11, (11, 2): 11, (11, 3): 11, (11, 4): 11, (11, 5): 12, (11, 6): 12, (11, 7): 12, (11, 8): 12, (11, 9): 12, (11, 10): 12, (11, 11): 12, (11, 12): 12,
    (12, 1): 12, (12, 2): 12, (12, 3): 12, (12, 4): 12, (12, 5): 12, (12, 6): 12, (12, 7): 12, (12, 8): 12, (12, 9): 12, (12, 10): 12, (12, 11): 12, (12, 12): 12,
}

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

    Args:
        angle: Calibrated neck angle (0° = neutral)

    Returns:
        REBA neck score (1-3)
    """
    angle = abs(angle)
    if angle <= 10:
        return 1  # 0-10° flexion
    elif angle <= 20:
        return 2  # 10-20° flexion
    else:
        return 3  # >20° flexion or extension


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

    Args:
        angle: Calibrated knee angle (0° = straight)

    Returns:
        REBA legs score (1-4)
    """
    angle = abs(angle)
    if angle <= 15:
        return 1  # Bilateral weight bearing, walking
    elif angle <= 30:
        return 2  # 30-60° flexion
    elif angle <= 60:
        return 3  # >60° flexion
    else:
        return 4  # Significant flexion


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


def get_table_a_score(neck: int, trunk: int, legs: int) -> int:
    """Look up score in Table A."""
    # Clamp values
    neck = max(1, min(3, neck))
    trunk = max(1, min(5, trunk))
    legs = max(1, min(4, legs))
    return TABLE_A.get((neck, trunk, legs), 5)


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

    # Calculate table scores
    score.score_a = get_table_a_score(score.neck_score, score.trunk_score, score.legs_score)
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

        # Add to buffer
        self.score_buffer.append(raw_score.final_score)
        if len(self.score_buffer) > self.buffer_size:
            self.score_buffer.pop(0)

        # Smooth the final score
        avg_score = round(sum(self.score_buffer) / len(self.score_buffer))
        avg_score = max(1, min(12, avg_score))

        # Update the score with smoothed value
        raw_score.final_score = avg_score
        risk_info = RISK_LEVELS.get(avg_score, ("high", (0, 0, 255)))
        raw_score.risk_level = risk_info[0]
        raw_score.risk_color = risk_info[1]

        self.current_score = raw_score
        return raw_score

    def reset(self):
        """Reset the score buffer."""
        self.score_buffer.clear()
        self.current_score = None
