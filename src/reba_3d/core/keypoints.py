"""
Keypoint data loading and processing functions.

Handles loading 3D keypoint data from JSON, filtering by confidence,
and building DataFrames for analysis.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union

from reba_3d.config.settings import KEYPOINT_NAMES, CONFIDENCE_THRESHOLD


def load_keypoints_3d(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load 3D keypoint data from a JSON file.

    Args:
        json_path: Path to the keypoints_3d.json file

    Returns:
        List of frame dictionaries containing keypoint data

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Le fichier {json_path} n'existe pas.")

    with open(json_path, 'r') as f:
        return json.load(f)


def filter_by_confidence(
    keypoint: Dict[str, float],
    threshold: float = CONFIDENCE_THRESHOLD
) -> Dict[str, float]:
    """
    Filter keypoint data based on confidence threshold.

    Args:
        keypoint: Dictionary with x, y, z, confidence keys
        threshold: Minimum confidence to consider valid (default: 0.35)

    Returns:
        Dictionary with x, y, z set to NaN if confidence < threshold
    """
    if keypoint.get("confidence", 0) < threshold:
        return {"x": np.nan, "y": np.nan, "z": np.nan}
    return {
        "x": float(keypoint["x"]),
        "y": float(keypoint["y"]),
        "z": float(keypoint["z"])
    }


def build_dataframe(
    data_json: List[Dict[str, Any]],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    keypoint_names: List[str] = None
) -> pd.DataFrame:
    """
    Build a pandas DataFrame from keypoint JSON data.

    Converts raw JSON keypoint data into a structured DataFrame,
    filtering by confidence and keeping only the most confident
    person per frame.

    Args:
        data_json: List of frame dictionaries from JSON
        confidence_threshold: Minimum confidence threshold (default: 0.35)
        keypoint_names: List of keypoint names (default: KEYPOINT_NAMES)

    Returns:
        DataFrame with columns for each keypoint coordinate (e.g., Nose_x, Nose_y, Nose_z)
    """
    if keypoint_names is None:
        keypoint_names = KEYPOINT_NAMES

    data = []

    for frame_data in data_json:
        frame_number = frame_data["frame"]
        keypoints_3d = frame_data["keypoints_3d"]

        for person_id, person_keypoints in enumerate(keypoints_3d):
            frame_dict = {"frame": frame_number, "person_id": person_id}
            sum_conf = 0.0

            for i, keypoint in enumerate(person_keypoints):
                if i < len(keypoint_names):
                    c = keypoint["confidence"]
                    sum_conf += c

                    if c < confidence_threshold:
                        frame_dict[f"{keypoint_names[i]}_x"] = np.nan
                        frame_dict[f"{keypoint_names[i]}_y"] = np.nan
                        frame_dict[f"{keypoint_names[i]}_z"] = np.nan
                    else:
                        frame_dict[f"{keypoint_names[i]}_x"] = float(keypoint["x"])
                        frame_dict[f"{keypoint_names[i]}_y"] = float(keypoint["y"])
                        frame_dict[f"{keypoint_names[i]}_z"] = float(keypoint["z"])

            frame_dict["sum_conf"] = sum_conf
            data.append(frame_dict)

    df = pd.DataFrame(data).sort_values(
        by=['frame', 'sum_conf'],
        ascending=[True, False]
    )

    # Garder uniquement la personne la plus confiante par frame
    df = df.drop_duplicates(subset='frame', keep='first')
    df = df.sort_values(by='frame').reset_index(drop=True)

    return df


def get_keypoint_position(
    df_row: pd.Series,
    keypoint_name: str
) -> np.ndarray:
    """
    Extract 3D position of a keypoint from a DataFrame row.

    Args:
        df_row: A row from the keypoints DataFrame
        keypoint_name: Name of the keypoint (e.g., "Nose")

    Returns:
        numpy array [x, y, z] or array of NaN if not available
    """
    return np.array([
        df_row.get(f"{keypoint_name}_x", np.nan),
        df_row.get(f"{keypoint_name}_y", np.nan),
        df_row.get(f"{keypoint_name}_z", np.nan)
    ], dtype=float)


def keypoint_is_valid(position: np.ndarray) -> bool:
    """
    Check if a keypoint position is valid (no NaN values and non-zero depth).

    Args:
        position: 3D position array [x, y, z]

    Returns:
        True if position is valid, False otherwise
    """
    return not np.isnan(position).any() and position[2] != 0
