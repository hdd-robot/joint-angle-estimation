"""
REBA scoring tables.

Contains the standard REBA lookup tables for calculating posture scores:
- TABLE_A: Trunk, Neck, Legs scores
- TABLE_B: Upper Arm, Lower Arm, Wrist scores
- TABLE_C: Combined scores from Table A and Table B
"""

from typing import Dict, Tuple, Union

# =============================================================================
# Table A: Trunk, Neck, Legs
# =============================================================================
# Key: (trunk_score, neck_score, legs_score)
# Value: Score A

TABLE_A: Dict[Tuple[int, int, int], int] = {
    # Trunk score 1
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 1, 3): 3, (1, 1, 4): 4,
    (1, 2, 1): 1, (1, 2, 2): 2, (1, 2, 3): 3, (1, 2, 4): 4,
    (1, 3, 1): 3, (1, 3, 2): 3, (1, 3, 3): 5, (1, 3, 4): 6,

    # Trunk score 2
    (2, 1, 1): 2, (2, 1, 2): 3, (2, 1, 3): 4, (2, 1, 4): 5,
    (2, 2, 1): 3, (2, 2, 2): 4, (2, 2, 3): 5, (2, 2, 4): 6,
    (2, 3, 1): 4, (2, 3, 2): 5, (2, 3, 3): 6, (2, 3, 4): 7,

    # Trunk score 3
    (3, 1, 1): 2, (3, 1, 2): 4, (3, 1, 3): 5, (3, 1, 4): 6,
    (3, 2, 1): 4, (3, 2, 2): 5, (3, 2, 3): 6, (3, 2, 4): 7,
    (3, 3, 1): 5, (3, 3, 2): 6, (3, 3, 3): 7, (3, 3, 4): 8,

    # Trunk score 4
    (4, 1, 1): 3, (4, 1, 2): 5, (4, 1, 3): 6, (4, 1, 4): 7,
    (4, 2, 1): 5, (4, 2, 2): 6, (4, 2, 3): 7, (4, 2, 4): 8,
    (4, 3, 1): 6, (4, 3, 2): 7, (4, 3, 3): 8, (4, 3, 4): 9,

    # Trunk score 5
    (5, 1, 1): 4, (5, 1, 2): 6, (5, 1, 3): 7, (5, 1, 4): 8,
    (5, 2, 1): 6, (5, 2, 2): 7, (5, 2, 3): 8, (5, 2, 4): 9,
    (5, 3, 1): 7, (5, 3, 2): 8, (5, 3, 3): 9, (5, 3, 4): 9,
}

# =============================================================================
# Table B: Upper Arm, Lower Arm, Wrist
# =============================================================================
# Key: (upper_arm_score, lower_arm_score, wrist_score)
# Value: Score B

TABLE_B: Dict[Tuple[int, int, int], int] = {
    # Upper arm score 1
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 1, 3): 2,
    (1, 2, 1): 2, (1, 2, 2): 2, (1, 2, 3): 3,

    # Upper arm score 2
    (2, 1, 1): 2, (2, 1, 2): 2, (2, 1, 3): 3,
    (2, 2, 1): 2, (2, 2, 2): 3, (2, 2, 3): 4,

    # Upper arm score 3
    (3, 1, 1): 3, (3, 1, 2): 4, (3, 1, 3): 5,
    (3, 2, 1): 4, (3, 2, 2): 5, (3, 2, 3): 5,

    # Upper arm score 4
    (4, 1, 1): 4, (4, 1, 2): 5, (4, 1, 3): 5,
    (4, 2, 1): 5, (4, 2, 2): 6, (4, 2, 3): 7,

    # Upper arm score 5
    (5, 1, 1): 6, (5, 1, 2): 7, (5, 1, 3): 8,
    (5, 2, 1): 7, (5, 2, 2): 8, (5, 2, 3): 8,

    # Upper arm score 6
    (6, 1, 1): 7, (6, 1, 2): 8, (6, 1, 3): 8,
    (6, 2, 1): 8, (6, 2, 2): 9, (6, 2, 3): 9,
}

# =============================================================================
# Table C: Combined Score A and Score B
# =============================================================================
# Key: (score_a, score_b)
# Value: Score C (final REBA score before activity adjustment)

TABLE_C: Dict[Tuple[int, int], int] = {
    # Score A = 1
    (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 2, (1, 5): 3,
    (1, 6): 3, (1, 7): 4, (1, 8): 5, (1, 9): 6, (1, 10): 7,
    (1, 11): 7, (1, 12): 7,

    # Score A = 2
    (2, 1): 1, (2, 2): 2, (2, 3): 2, (2, 4): 3, (2, 5): 4,
    (2, 6): 4, (2, 7): 5, (2, 8): 6, (2, 9): 6, (2, 10): 7,
    (2, 11): 7, (2, 12): 8,

    # Score A = 3
    (3, 1): 2, (3, 2): 3, (3, 3): 3, (3, 4): 3, (3, 5): 4,
    (3, 6): 5, (3, 7): 6, (3, 8): 7, (3, 9): 7, (3, 10): 8,
    (3, 11): 8, (3, 12): 8,

    # Score A = 4
    (4, 1): 3, (4, 2): 4, (4, 3): 4, (4, 4): 4, (4, 5): 5,
    (4, 6): 6, (4, 7): 7, (4, 8): 8, (4, 9): 8, (4, 10): 9,
    (4, 11): 9, (4, 12): 9,

    # Score A = 5
    (5, 1): 4, (5, 2): 4, (5, 3): 4, (5, 4): 5, (5, 5): 6,
    (5, 6): 7, (5, 7): 8, (5, 8): 8, (5, 9): 9, (5, 10): 9,
    (5, 11): 9, (5, 12): 9,

    # Score A = 6
    (6, 1): 6, (6, 2): 6, (6, 3): 6, (6, 4): 7, (6, 5): 8,
    (6, 6): 8, (6, 7): 9, (6, 8): 9, (6, 9): 10, (6, 10): 10,
    (6, 11): 10, (6, 12): 10,

    # Score A = 7
    (7, 1): 7, (7, 2): 7, (7, 3): 7, (7, 4): 8, (7, 5): 9,
    (7, 6): 9, (7, 7): 9, (7, 8): 10, (7, 9): 10, (7, 10): 11,
    (7, 11): 11, (7, 12): 11,

    # Score A = 8
    (8, 1): 8, (8, 2): 8, (8, 3): 8, (8, 4): 9, (8, 5): 10,
    (8, 6): 10, (8, 7): 10, (8, 8): 10, (8, 9): 10, (8, 10): 11,
    (8, 11): 11, (8, 12): 11,

    # Score A = 9
    (9, 1): 9, (9, 2): 9, (9, 3): 9, (9, 4): 10, (9, 5): 10,
    (9, 6): 10, (9, 7): 11, (9, 8): 11, (9, 9): 11, (9, 10): 12,
    (9, 11): 12, (9, 12): 12,

    # Score A = 10
    (10, 1): 10, (10, 2): 10, (10, 3): 10, (10, 4): 11, (10, 5): 11,
    (10, 6): 11, (10, 7): 11, (10, 8): 12, (10, 9): 12, (10, 10): 12,
    (10, 11): 12, (10, 12): 12,

    # Score A = 11
    (11, 1): 11, (11, 2): 11, (11, 3): 11, (11, 4): 11, (11, 5): 12,
    (11, 6): 12, (11, 7): 12, (11, 8): 12, (11, 9): 12, (11, 10): 12,
    (11, 11): 12, (11, 12): 12,

    # Score A = 12
    (12, 1): 12, (12, 2): 12, (12, 3): 12, (12, 4): 12, (12, 5): 12,
    (12, 6): 12, (12, 7): 12, (12, 8): 12, (12, 9): 12, (12, 10): 12,
    (12, 11): 12, (12, 12): 12,
}


def lookup_table_a(trunk: int, neck: int, legs: int) -> Union[int, str]:
    """
    Look up score from Table A.

    Args:
        trunk: Trunk posture score (1-5)
        neck: Neck posture score (1-3)
        legs: Legs posture score (1-4)

    Returns:
        Score A value, or "Invalide" if combination not found
    """
    return TABLE_A.get((trunk, neck, legs), "Invalide")


def lookup_table_b(upper_arm: int, lower_arm: int, wrist: int) -> Union[int, str]:
    """
    Look up score from Table B.

    Args:
        upper_arm: Upper arm posture score (1-6)
        lower_arm: Lower arm posture score (1-2)
        wrist: Wrist posture score (1-3)

    Returns:
        Score B value, or "Invalide" if combination not found
    """
    return TABLE_B.get((upper_arm, lower_arm, wrist), "Invalide")


def lookup_table_c(score_a: int, score_b: int) -> Union[int, str]:
    """
    Look up final score from Table C.

    Args:
        score_a: Score from Table A (1-12)
        score_b: Score from Table B (1-12)

    Returns:
        Score C value, or "Invalide" if combination not found
    """
    return TABLE_C.get((score_a, score_b), "Invalide")
