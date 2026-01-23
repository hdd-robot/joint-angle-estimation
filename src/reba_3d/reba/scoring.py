"""
REBA scoring functions for individual body segments.

Converts calibrated angles into REBA posture scores for each body segment.
"""

import math
from typing import Optional, Union

from reba_3d.reba.tables import TABLE_A, TABLE_B, TABLE_C


def is_nan(value) -> bool:
    """Check if a value is NaN."""
    return isinstance(value, float) and math.isnan(value)


def safe_get(liste: list, index: int):
    """Safely get an element from a list, returning None if out of range."""
    if index < 0 or index >= len(liste):
        return None
    return liste[index]


# =============================================================================
# Individual segment scoring functions
# =============================================================================

def score_neck(alpha: Optional[float], beta: Optional[float] = None, gamma: Optional[float] = None) -> tuple:
    """
    Calculate neck score and malus.

    Args:
        alpha: Neck flexion/extension angle (degrees)
        beta: Neck lateral bend angle (degrees)
        gamma: Neck rotation angle (degrees)

    Returns:
        Tuple (base_score, malus)
    """
    if alpha is None or is_nan(alpha):
        return 1, 0

    # Score de base
    if alpha > 15:
        score = 2
    else:
        score = 1

    # Calcul du malus
    malus = 0
    if beta is not None and not is_nan(beta):
        if 15 <= beta <= 30:
            malus += 1
    if gamma is not None and not is_nan(gamma):
        if abs(gamma) > 25:
            malus += 1

    return score, malus


def score_torso(alpha: Optional[float], beta: Optional[float] = None, gamma: Optional[float] = None) -> tuple:
    """
    Calculate torso (trunk) score and malus.

    Args:
        alpha: Torso flexion angle (degrees)
        beta: Torso lateral bend angle (degrees)
        gamma: Torso rotation angle (degrees)

    Returns:
        Tuple (base_score, malus)
    """
    if alpha is None or is_nan(alpha):
        return 1, 0

    # Score de base selon l'inclinaison
    if alpha > 40:
        score = 4
    elif alpha < 20:
        score = 1
    elif alpha > 20:
        score = 3
    else:
        score = 2

    # Calcul du malus
    malus = 0
    if beta is not None and not is_nan(beta):
        if beta < -18:
            malus += 1
    if gamma is not None and not is_nan(gamma):
        if -15 <= gamma < -9:
            malus += 1

    return score, malus


def score_shoulder(
    alpha: Optional[float],
    alpha_g: Optional[float] = None,
    elevation: Optional[float] = None,
    elevation_g: Optional[float] = None
) -> tuple:
    """
    Calculate shoulder (upper arm) score and malus.

    Uses right arm if available, otherwise left arm.

    Args:
        alpha: Right shoulder flexion angle (degrees)
        alpha_g: Left shoulder flexion angle (degrees)
        elevation: Right shoulder elevation angle (degrees)
        elevation_g: Left shoulder elevation angle (degrees)

    Returns:
        Tuple (base_score, malus_right, malus_left)
    """
    # Déterminer la valeur d'angle à utiliser (droite prioritaire)
    angle = alpha if (alpha is not None and not is_nan(alpha)) else alpha_g

    if angle is None or is_nan(angle):
        return 1, 0, 0

    # Score de base
    if angle > 80:
        score = 4
    elif angle < -15:
        score = 3
    elif angle < 20:
        score = 1
    elif angle < 30:
        score = 2
    else:
        score = 3

    # Malus pour élévation de l'épaule
    malus_d = 0
    malus_g = 0

    if elevation is not None and not is_nan(elevation):
        if elevation < -5:
            malus_d = 1

    if elevation_g is not None and not is_nan(elevation_g):
        if elevation_g < -5:
            malus_g = 1

    return score, malus_d, malus_g


def score_elbow(angle: Optional[float], angle_g: Optional[float] = None) -> int:
    """
    Calculate elbow (lower arm) score.

    Uses right arm if available, otherwise left arm.

    Args:
        angle: Right elbow angle (degrees)
        angle_g: Left elbow angle (degrees)

    Returns:
        Elbow score (1 or 2)
    """
    # Déterminer la valeur d'angle à utiliser
    value = angle if (angle is not None and not is_nan(angle)) else angle_g

    if value is None or is_nan(value):
        return 1

    if value > 40:
        return 2
    return 1


def score_knee(angle: Optional[float], angle_g: Optional[float] = None) -> int:
    """
    Calculate knee (legs) base score.

    Uses right leg if available, otherwise left leg.

    Args:
        angle: Right knee flexion angle (degrees)
        angle_g: Left knee flexion angle (degrees)

    Returns:
        Knee score (1 or 2)
    """
    value = angle if (angle is not None and not is_nan(angle)) else angle_g

    if value is None or is_nan(value):
        return 1

    if value > 50:
        return 2
    return 1


def score_feet_contact(contact_state: str) -> int:
    """
    Calculate feet contact malus for legs score.

    Args:
        contact_state: Contact state ("OK", "DROIT", "GAUCHE", "404")

    Returns:
        Malus value (1 if both feet, 2 if single foot)
    """
    if contact_state in ["GAUCHE", "DROIT"]:
        return 2
    return 1


def score_wrist(angle: Optional[float] = None) -> int:
    """
    Calculate wrist score.

    Note: Wrist scoring not fully implemented due to VRAM constraints.
    Returns default value of 1.

    Args:
        angle: Wrist angle (degrees) - currently unused

    Returns:
        Wrist score (always 1)
    """
    return 1


# =============================================================================
# Combined scoring functions
# =============================================================================

def calculate_score_a(trunk_score: int, neck_score: int, legs_score: int) -> Union[int, str]:
    """
    Calculate Score A from Table A.

    Args:
        trunk_score: Total trunk score (base + malus)
        neck_score: Total neck score (base + malus)
        legs_score: Total legs score (base + malus)

    Returns:
        Score A value or "Invalide"
    """
    return TABLE_A.get((trunk_score, neck_score, legs_score), "Invalide")


def calculate_score_b(upper_arm_score: int, lower_arm_score: int, wrist_score: int) -> Union[int, str]:
    """
    Calculate Score B from Table B.

    Args:
        upper_arm_score: Total upper arm score (base + malus)
        lower_arm_score: Lower arm (elbow) score
        wrist_score: Wrist score

    Returns:
        Score B value or "Invalide"
    """
    return TABLE_B.get((upper_arm_score, lower_arm_score, wrist_score), "Invalide")


def calculate_score_c(score_a: int, score_b: int) -> Union[int, str]:
    """
    Calculate Score C from Table C.

    Args:
        score_a: Score A value
        score_b: Score B value

    Returns:
        Score C value or "Invalide"
    """
    return TABLE_C.get((score_a, score_b), "Invalide")


def apply_activity_malus(
    score_c: int,
    score_a: int,
    score_b: int,
    load_malus: int = 1,
    coupling_malus: int = 1,
    activity_malus: int = 1
) -> tuple:
    """
    Apply load, coupling, and activity malus to final score.

    Args:
        score_c: Base Score C
        score_a: Score A
        score_b: Score B
        load_malus: Load/force malus (default: 1)
        coupling_malus: Coupling malus (default: 1)
        activity_malus: Activity malus (default: 1)

    Returns:
        Tuple (final_score, recalculated_flag)
    """
    new_score_a = min(score_a + load_malus, 12)
    new_score_b = min(score_b + coupling_malus, 12)
    new_key = (new_score_a, score_b)
    new_score_c = TABLE_C.get(new_key, score_c)
    final_score = min(new_score_c + activity_malus, 12)

    recalculated = (final_score != score_c)
    return final_score, recalculated


# =============================================================================
# Risk level mapping
# =============================================================================

def score_to_risk_label(score: Union[int, str]) -> str:
    """
    Convert REBA score to risk level label (French).

    Args:
        score: REBA final score (1-12) or "Invalide"

    Returns:
        Risk level label in French
    """
    if score == "Invalide" or score is None:
        return "invalide"
    elif score == 1:
        return "sans risque"
    elif score <= 3:
        return "risque faible"
    elif score < 7:
        return "risque moyen"
    elif score <= 10:
        return "risque élevé"
    else:
        return "très élevé"


def score_to_risk_label_en(score: Union[int, str]) -> str:
    """
    Convert REBA score to risk level label (English).

    Args:
        score: REBA final score (1-12) or "Invalide"

    Returns:
        Risk level label in English
    """
    label_fr = score_to_risk_label(score)
    label_map = {
        "sans risque": "negligible risk",
        "risque faible": "low risk",
        "risque moyen": "medium risk",
        "risque élevé": "high risk",
        "très élevé": "very high risk",
        "invalide": "invalid",
    }
    return label_map.get(label_fr, "invalid")
