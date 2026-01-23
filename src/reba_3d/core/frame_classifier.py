"""
Frame classification based on visible keypoints.

Classifies frames into face view, right profile, or left profile
based on which keypoints are visible and valid.
"""

import pandas as pd
from typing import Dict, List

from reba_3d.config.settings import (
    MANDATORY_KEYPOINTS_FACE,
    MANDATORY_KEYPOINTS_PROFIL_DROIT,
    MANDATORY_KEYPOINTS_PROFIL_GAUCHE,
)


def check_keypoints(df_row: pd.Series, keypoint_list: List[str]) -> bool:
    """
    Check if all required keypoints are valid in a DataFrame row.

    A keypoint is considered valid if it has non-NaN coordinates
    and non-zero depth (z != 0).

    Args:
        df_row: A row from the keypoints DataFrame
        keypoint_list: List of required keypoint names

    Returns:
        True if all keypoints are valid, False otherwise
    """
    return all(
        not pd.isna(df_row.get(f"{kp}_z", float('nan'))) and
        df_row.get(f"{kp}_z", 0) != 0
        for kp in keypoint_list
    )


def check_keypoints_dict(
    person_keypoints: Dict[str, Dict[str, float]],
    keypoint_list: List[str]
) -> bool:
    """
    Check if all required keypoints are valid in a keypoint dictionary.

    Args:
        person_keypoints: Dictionary mapping keypoint names to position dicts
        keypoint_list: List of required keypoint names

    Returns:
        True if all keypoints are valid, False otherwise
    """
    for kp in keypoint_list:
        if kp not in person_keypoints:
            return False
        if person_keypoints[kp].get("z", 0) == 0:
            return False
    return True


def classify_frame(df_row: pd.Series) -> str:
    """
    Classify a single frame based on visible keypoints.

    Args:
        df_row: A row from the keypoints DataFrame

    Returns:
        Classification string: "face", "profil_droit", "profil_gauche", or "invalid"
    """
    if check_keypoints(df_row, MANDATORY_KEYPOINTS_FACE):
        return "face"
    elif check_keypoints(df_row, MANDATORY_KEYPOINTS_PROFIL_DROIT):
        return "profil_droit"
    elif check_keypoints(df_row, MANDATORY_KEYPOINTS_PROFIL_GAUCHE):
        return "profil_gauche"
    return "invalid"


def classify_frames(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Classify all frames in a DataFrame by view type.

    Args:
        df: DataFrame containing keypoint data

    Returns:
        Dictionary with keys "face", "profil_droit", "profil_gauche"
        mapping to lists of frame numbers
    """
    pertinent_frames = {
        "face": [],
        "profil_droit": [],
        "profil_gauche": []
    }

    for _, row in df.iterrows():
        frame_number = row["frame"]
        classification = classify_frame(row)

        if classification in pertinent_frames:
            pertinent_frames[classification].append(frame_number)

    # Trier les listes
    for key in pertinent_frames:
        pertinent_frames[key] = sorted(pertinent_frames[key])

    return pertinent_frames


def get_all_pertinent_frames(pertinent_frames: Dict[str, List[int]]) -> List[int]:
    """
    Get a sorted list of all pertinent frames from classification results.

    Args:
        pertinent_frames: Dictionary from classify_frames()

    Returns:
        Sorted list of all unique frame numbers
    """
    all_frames = set()
    for frames in pertinent_frames.values():
        all_frames.update(frames)
    return sorted(all_frames)


def filter_pertinent_frames(
    df: pd.DataFrame,
    pertinent_frames: Dict[str, List[int]]
) -> pd.DataFrame:
    """
    Filter DataFrame to keep only pertinent frames.

    Args:
        df: Original DataFrame
        pertinent_frames: Classification results from classify_frames()

    Returns:
        Filtered DataFrame containing only pertinent frames
    """
    all_frames = get_all_pertinent_frames(pertinent_frames)
    return df[df['frame'].isin(all_frames)].sort_values(by='frame').reset_index(drop=True)
