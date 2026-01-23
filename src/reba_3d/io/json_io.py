"""
JSON input/output functions for keypoint and risk data.

Handles reading and writing keypoints_3d.json and risk_times.json files.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple


def read_keypoints_json(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read 3D keypoints from a JSON file.

    Args:
        json_path: Path to the keypoints_3d.json file

    Returns:
        List of frame dictionaries with structure:
        [
            {
                "frame": int,
                "keypoints_3d": [
                    [{"x": float, "y": float, "z": float, "confidence": float}, ...]
                ]
            },
            ...
        ]

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Le fichier {json_path} n'existe pas.")

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_keypoints_json(
    data: List[Dict[str, Any]],
    output_path: Union[str, Path],
    indent: int = 4
) -> None:
    """
    Write 3D keypoints to a JSON file.

    Args:
        data: List of frame dictionaries with keypoint data
        output_path: Path to write the JSON file
        indent: JSON indentation level (default: 4)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def read_risk_times_json(json_path: Union[str, Path]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Read risk time intervals from a JSON file.

    Args:
        json_path: Path to the risk_times.json file

    Returns:
        Dictionary mapping risk labels to time intervals:
        {
            "negligible risk": [(start1, end1), (start2, end2), ...],
            "medium risk": [...],
            ...
        }

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Le fichier {json_path} n'existe pas.")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert lists to tuples
    result = {}
    for label, intervals in data.items():
        result[label] = [(start, end) for start, end in intervals]

    return result


def write_risk_times_json(
    risk_times: Dict[str, List[Tuple[float, float]]],
    output_path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Write risk time intervals to a JSON file.

    Args:
        risk_times: Dictionary mapping risk labels to time intervals
        output_path: Path to write the JSON file
        indent: JSON indentation level (default: 2)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tuples to lists for JSON serialization
    data = {}
    for label, intervals in risk_times.items():
        data[label] = [list(interval) for interval in intervals]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def convert_risk_times_fr_to_en(
    risk_times_fr: Dict[str, List]
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Convert French risk labels to English.

    Args:
        risk_times_fr: Risk times with French labels
            Format: {"label_fr": [(window_num, start, end), ...]}

    Returns:
        Risk times with English labels
            Format: {"label_en": [(start, end), ...]}
    """
    label_map = {
        "sans risque": "negligible risk",
        "risque faible": "low risk",
        "risque moyen": "medium risk",
        "risque élevé": "high risk",
        "très élevé": "very high risk",
    }

    result = {}
    for fr_label, intervals in risk_times_fr.items():
        en_label = label_map.get(fr_label)
        if en_label:
            if en_label not in result:
                result[en_label] = []
            for item in intervals:
                if len(item) == 3:
                    # Format: (window_num, start, end)
                    _, start, end = item
                else:
                    # Format: (start, end)
                    start, end = item
                result[en_label].append((start, end))

    return result
