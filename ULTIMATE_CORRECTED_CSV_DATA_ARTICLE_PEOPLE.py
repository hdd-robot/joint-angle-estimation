import os
import re
import numpy as np
import pandas as pd

# ======================
# Config / constants
# ======================
PEOPLE_DIR = "PEOPLE"              # root folder containing ID_* subfolders
LOG_2D_NAME = "reba_analysis_2d.log"
LOG_3D_NAME = "reba_analysis_3d.log"
OUTPUT_CSV = "ULTIMATE_ALL_DATA_EXPERIMENT.csv"

# Risk label -> numeric score
RISK_MAP = {
    "invalid": np.nan,
    "negligible risk": 0,
    "low risk": 0,
    "medium risk": 1,
    "high risk": 2,
    "very high risk": 3,
}

# ======================
# Parsing helpers
# ======================
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_angle_list(text: str, prefix: str) -> list[float]:
    """
    Extracts an angle list like:
    Alpha_neck : [F1: 0.00°, F2: 0.00°]
    """
    match = re.search(
        rf"{re.escape(prefix)}\s*:\s*\[([^\]]+)\]",
        text,
        flags=re.IGNORECASE
    )
    if not match:
        return []

    values: list[float] = []
    for item in match.group(1).split(","):
        m = re.search(r":\s*([-+]?\d*\.?\d+|nan)\s*°", item.strip(), flags=re.IGNORECASE)
        if m:
            s = m.group(1)
            values.append(np.nan if s.lower() == "nan" else float(s))
    return values

def extract_reba_scores(text: str, dimension: str) -> dict[int, float]:
    """
    Extracts the REBA risks list like:
    2D REBA risks:
    [ F1: negligible risk, F2: medium risk ]
    Returns {window_index -> score} where window_index is the number after F (1-based).
    """
    pattern = rf"{dimension}\s+REBA\s+risks\s*:?[\s\r\n]*\[(.*?)\]"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return {}

    content = match.group(1)
    items = re.findall(r"F(\d+)\*?\s*:\s*([^,\]]+)", content, flags=re.IGNORECASE)

    scores: dict[int, float] = {}
    for f, risk in items:
        risk_clean = re.sub(r"\s+", " ", risk.strip().lower())
        scores[int(f)] = RISK_MAP.get(risk_clean, np.nan)
    return scores

def extract_table_values(text: str, dimension: str, table_letter: str) -> list[float]:
    """
    Extracts table values like:
    2D Table A
    [1, 1]
    or
    3D Table C (Final Score)
    [1, 1]
    """
    pattern = rf"{dimension}\s+Table\s+{table_letter}\b[^\n]*\n\s*\[([^\]]+)\]"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return []

    raw_values = match.group(1).split(",")
    out: list[float] = []
    for v in raw_values:
        try:
            out.append(float(v.strip()))
        except ValueError:
            out.append(np.nan)
    return out

def discover_people_folders(root_dir: str) -> list[str]:
    """Find all ID_* folders inside PEOPLE_DIR."""
    if not os.path.isdir(root_dir):
        return []

    folders: list[str] = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and re.match(r"^ID[_\s-]?\d+$", name, flags=re.IGNORECASE):
            folders.append(full)

    def id_num(path: str) -> int:
        m = re.search(r"(\d+)$", os.path.basename(path))
        return int(m.group(1)) if m else 10**9

    return sorted(folders, key=id_num)

def parse_person_id(folder_path: str) -> int:
    m = re.search(r"(\d+)$", os.path.basename(folder_path))
    return int(m.group(1)) if m else -1

# ======================
# Angle specs (final column prefixes -> possible log prefixes)
# IMPORTANT: the LEFT side is what will appear in the CSV (with _2D/_3D suffix)
# ======================
ANGLE_SPECS: dict[str, list[str]] = {
    # Neck
    "Neck_alpha": ["Alpha_neck", "Neck_alpha"],
    "Neck_beta":  ["Beta_neck", "Neck_beta"],
    "Neck_gamma": ["Gamma_neck", "Neck_gamma"],

    # Torso
    "Torso_alpha": ["Alpha_torso", "Torso_alpha"],
    "Torso_beta":  ["Beta_torso", "Torso_beta"],
    "Torso_gamma": ["Gamma_torso", "Torso_gamma"],

    # Elbows
    "Right_elbow":      ["Angle_right_elbow", "Right_elbow_angle", "Elbow_angle", "Angle_elbow"],
    "Left_elbow": ["Angle_left_elbow",  "Left_elbow_angle"],

    # Knees
    "Right_Knee":      ["Angle_right_knee", "Right_knee_angle", "Knee_angle", "Angle_knee"],
    "Left_knee": ["Angle_left_knee",  "Left_knee_angle"],

    # Right shoulder
    "Right_shoulder_alpha":     ["Alpha_right_shoulder", "Right_shoulder_alpha"],
    "Right_shoulder_beta":      ["Beta_right_shoulder",  "Right_shoulder_beta"],
    "Right_shoulder_gamma":     ["Gamma_right_shoulder", "Right_shoulder_gamma"],
    "Right_shoulder_elevation": ["Elevation_right_shoulder", "Right_shoulder_elevation", "Shoulder_elevation"],

    # Left shoulder
    "Left_shoulder_alpha":     ["Alpha_left_shoulder", "Left_shoulder_alpha"],
    "Left_shoulder_beta":      ["Beta_left_shoulder",  "Left_shoulder_beta"],
    "Left_shoulder_gamma":     ["Gamma_left_shoulder", "Left_shoulder_gamma"],
    "Left_shoulder_elevation": ["Elevation_left_shoulder", "Left_shoulder_elevation"],
}

def extract_angles(text: str) -> dict[str, list[float]]:
    """Return {angle_key: list_of_values} by trying all aliases."""
    out: dict[str, list[float]] = {}
    for angle_key, aliases in ANGLE_SPECS.items():
        vals: list[float] = []
        for alias in aliases:
            vals = extract_angle_list(text, alias)
            if vals:
                break
        out[angle_key] = vals
    return out

def longest_list_len(d: dict[str, list[float]]) -> int:
    lens = [len(v) for v in d.values() if isinstance(v, list)]
    return max(lens) if lens else 0

# ======================
# Extraction / merge
# ======================
all_rows: list[dict] = []
people_folders = discover_people_folders(PEOPLE_DIR)

for folder in people_folders:
    person_id = parse_person_id(folder)

    path_2d = os.path.join(folder, LOG_2D_NAME)
    path_3d = os.path.join(folder, LOG_3D_NAME)

    text_2d = read_text_file(path_2d) if os.path.isfile(path_2d) else ""
    text_3d = read_text_file(path_3d) if os.path.isfile(path_3d) else ""

    # Angles
    angles_2d = extract_angles(text_2d)
    angles_3d = extract_angles(text_3d)

    # Tables A/B/C
    table_a_2d = extract_table_values(text_2d, "2D", "A")
    table_b_2d = extract_table_values(text_2d, "2D", "B")
    table_c_2d = extract_table_values(text_2d, "2D", "C")

    table_a_3d = extract_table_values(text_3d, "3D", "A")
    table_b_3d = extract_table_values(text_3d, "3D", "B")
    table_c_3d = extract_table_values(text_3d, "3D", "C")

    # REBA risks (window-indexed: F1, F2, ...)
    reba_2d = extract_reba_scores(text_2d, "2D")
    reba_3d = extract_reba_scores(text_3d, "3D")

    # Number of windows = max length among available sources
    max_windows = max(
        longest_list_len(angles_2d),
        longest_list_len(angles_3d),
        len(table_a_2d), len(table_b_2d), len(table_c_2d),
        len(table_a_3d), len(table_b_3d), len(table_c_3d),
        (max(reba_2d.keys()) if reba_2d else 0),
        (max(reba_3d.keys()) if reba_3d else 0),
    )

    for window_idx in range(1, max_windows + 1):
        row = {"Person_ID": person_id, "Window_ID": window_idx}

        # 2D angles
        for key, values in angles_2d.items():
            if len(values) >= window_idx:
                row[f"{key}_2D"] = values[window_idx - 1]

        # 3D angles
        for key, values in angles_3d.items():
            if len(values) >= window_idx:
                row[f"{key}_3D"] = values[window_idx - 1]

        # Tables
        if len(table_a_2d) >= window_idx: row["Table_A_2D"] = table_a_2d[window_idx - 1]
        if len(table_b_2d) >= window_idx: row["Table_B_2D"] = table_b_2d[window_idx - 1]
        if len(table_c_2d) >= window_idx: row["Table_C_2D"] = table_c_2d[window_idx - 1]

        if len(table_a_3d) >= window_idx: row["Table_A_3D"] = table_a_3d[window_idx - 1]
        if len(table_b_3d) >= window_idx: row["Table_B_3D"] = table_b_3d[window_idx - 1]
        if len(table_c_3d) >= window_idx: row["Table_C_3D"] = table_c_3d[window_idx - 1]

        # REBA
        row["REBA_2D"] = reba_2d.get(window_idx, np.nan)
        row["REBA_3D"] = reba_3d.get(window_idx, np.nan)

        # Error
        a, b = row["REBA_2D"], row["REBA_3D"]
        row["Error"] = (a - b) if (not pd.isna(a) and not pd.isna(b)) else np.nan

        all_rows.append(row)

# ======================
# DataFrame + ensure all expected columns exist
# ======================
df_final = pd.DataFrame(all_rows).sort_values(by=["Person_ID", "Window_ID"])

expected_angle_cols = [f"{k}_2D" for k in ANGLE_SPECS] + [f"{k}_3D" for k in ANGLE_SPECS]
expected_other_cols = [
    "Table_A_2D", "Table_B_2D", "Table_C_2D",
    "Table_A_3D", "Table_B_3D", "Table_C_3D",
    "REBA_2D", "REBA_3D", "Error",
]

for col in expected_angle_cols + expected_other_cols:
    if col not in df_final.columns:
        df_final[col] = np.nan

# Optional: put columns in a stable, readable order
ordered_cols = (
    ["Person_ID", "Window_ID"] +
    expected_angle_cols +
    expected_other_cols
)
df_final = df_final[ordered_cols]

# Export
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Export completed: {OUTPUT_CSV} ({len(df_final)} rows)")

