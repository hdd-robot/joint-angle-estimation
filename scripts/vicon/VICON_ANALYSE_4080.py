import pandas as pd
from pathlib import Path

# Dossier où se trouve ce script .py
HERE = Path(__file__).resolve().parent

CSV_PATH = HERE / "angles_2d3d_20260211_093659.csv"

COLS_elbow = [
    "2d_right_elbow",
    "3d_right_elbow",
]

#limits included
FRAME_RANGES_elbow = [
    (207, 221),

]

COLS_shoulder = [
    "2d_right_shoulder_gamma",
    "3d_right_shoulder_beta",

]

#limits included
FRAME_RANGES_shoulder = [
    (155, 161),

]
COLS_shoulder = [
    "3d_right_shoulder_alpha",

]
#limits included
FRAME_RANGES_shoulder = [
    (99, 114),


]


COLS_Neck= [
    "2d_neck_gamma",
    "3d_neck_beta",

]
#limits included
FRAME_RANGES_neck = [
    (313, 317),


]

COLS_torso= [
    "2d_trunk_gamma",
    "3d_trunk_beta",

]
#limits included
FRAME_RANGES_torso = [
    (409, 421),


]
#  frame_start  frame_end  n_rows  2d_right_elbow  3d_right_elbow
#          207        221      15       57.672467       63.856813

#  frame_start  frame_end  n_rows  2d_right_shoulder_gamma  3d_right_shoulder_beta
#          155        161       7                 -59.4455              -52.839943

#  frame_start  frame_end  n_rows  3d_right_shoulder_alpha
#           99        114      16                65.426881

#  frame_start  frame_end  n_rows  3d_neck_alpha
#          290        301      12      18.876408

#  frame_start  frame_end  n_rows  2d_neck_gamma  3d_neck_beta
#          313        317       5       -0.98164        8.3223

#  frame_start  frame_end  n_rows  3d_trunk_alpha
#          257        268      12       54.429442

#  frame_start  frame_end  n_rows  2d_trunk_gamma  3d_trunk_beta
#          409        421      13       21.663169     -21.505977

##VICON
CSV_PATH = HERE / "trial_RGBD0002_angles_sync.csv"

COLS_shoulder = [
    "shoulderR_alpha_deg",

]

#limits included
FRAME_RANGES_shoulder = [
    (552, 623),

]

COLS_Neck= [
    "neck_alpha_deg",
    "neck_beta_deg",

]
#limits included
FRAME_RANGES_neck = [
    (1946, 1996),
    (2162, 2212),

]

COLS_torso= [
    "torso_alpha_deg",
    "torso_beta_deg",

]
#limits included
FRAME_RANGES_torso = [
    (1710, 1760),
    (2876, 2919),

]

#  frame_start  frame_end  n_rows  neck_alpha_deg
#         1946       1996      51      40.23

#  frame_start  frame_end  n_rows  neck_beta_deg

#         2162       2212      51       7.37

#  frame_start  frame_end  n_rows  shoulderR_alpha_deg
#          552        623      72           -70.135238


#  frame_start  frame_end  n_rows  torso_alpha_deg
#         1710       1760      51       -52.110861

#  frame_start  frame_end  n_rows  torso_beta_deg
#         2876       2919      44      40.907131

# (Optionnel mais super utile) Debug
# print("CWD:", Path.cwd())
# print("Script dir:", HERE)
# print("CSV exists?:", CSV_PATH.exists())
# print("CSV path:", CSV_PATH)

df = pd.read_csv(CSV_PATH)

# Sécurité: vérifier que les colonnes utilisées existent
missing = [c for c in COLS_shoulder if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

results = []
for start_f, end_f in FRAME_RANGES_torso:
    chunk = df.loc[df["frame"].between(start_f, end_f, inclusive="both"), COLS_torso]
    means = chunk.mean(numeric_only=True)

    results.append({
        "frame_start": start_f,
        "frame_end": end_f,
        "n_rows": len(chunk),
        **means.to_dict(),
    })

summary = pd.DataFrame(results)
print(summary.to_string(index=False))


# Optionnel: sauvegarder
# summary.to_csv("moyennes_par_plage.csv", index=False)
