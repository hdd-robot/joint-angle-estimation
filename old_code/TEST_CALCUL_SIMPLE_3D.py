import os
import json
import pandas as pd
import numpy as np
import re

### Objectif
# Charger les données OpenPose (position des keypoints 3D) depuis keypoints_3d.json
# Identifier les frames pertinentes (vue de face, profil droit, profil gauche)
# Filtrer ce DataFrame pour ne garder que ces frames pertinentes
# Appliquer une fenêtre glissante pour calculer un angle (ici l’angle du cou) en moyenne, avec gestion des valeurs nulles
# Calculer des offsets de calibration
# Appliquer la calibration à des données dynamiques.

# 📌 Définition du chemin du fichier JSON contenant les keypoints 3D
keypoints_json_path = os.path.expanduser("~/openpose/keypoints_3d.json")

# 📌 Vérifier si le fichier existe
if not os.path.exists(keypoints_json_path):
    raise FileNotFoundError(f"Erreur : Le fichier {keypoints_json_path} n'existe pas.")

# 📌 Charger les données du JSON
with open(keypoints_json_path, 'r') as f:
    data_json = json.load(f)

# 📌 Liste des keypoints OpenPose
keypoint_names = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

# 📌 Convertir les données JSON en un format tabulaire
data = []
for frame_data in data_json:
    frame_number = frame_data["frame"]
    keypoints_3d = frame_data["keypoints_3d"]

    for person_id, person_keypoints in enumerate(keypoints_3d):  # Plusieurs personnes possibles
        frame_dict = {"frame": frame_number, "person_id": person_id}  # Initialiser un dict avec frame et ID
        for i, keypoint in enumerate(person_keypoints):
            if i < len(keypoint_names):  # Vérifier que l'index ne dépasse pas la liste des keypoints
                confidence = keypoint["confidence"]
                frame_dict[f"{keypoint_names[i]}_confidence"] = confidence
                if confidence < 0.5:
                    frame_dict[f"{keypoint_names[i]}_x"] = np.nan
                    frame_dict[f"{keypoint_names[i]}_y"] = np.nan
                    frame_dict[f"{keypoint_names[i]}_z"] = np.nan

                else:
                    frame_dict[f"{keypoint_names[i]}_x"] = keypoint["x"]
                    frame_dict[f"{keypoint_names[i]}_y"] = keypoint["y"]
                    frame_dict[f"{keypoint_names[i]}_z"] = keypoint["z"]
        data.append(frame_dict)

# 📌 Convertir en DataFrame Pandas
df = pd.DataFrame(data)



# ----- II) Détection des frames pertinentes -----
mandatory_keypoints_face = ["Nose", "Neck", "RShoulder", "RElbow", "LShoulder", "LElbow"]
mandatory_keypoints_profil_droit = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "RHip"]
mandatory_keypoints_profil_gauche = ["Nose", "Neck", "LShoulder", "LElbow", "LWrist", "LHip"]

pertinent_frames = {"face": [], "profil_droit": [], "profil_gauche": []}


def check_keypoints(df_row, kp_list):
    """ Vérifie si tous les keypoints d'une frame sont valides (z != 0) """
    return all(df_row[f"{kp}_z"] != 0 for kp in kp_list)


# 📌 Parcourir les frames et classer les vues
for index, row in df.iterrows():
    frame_number = row["frame"]

    if check_keypoints(row, mandatory_keypoints_face):
        pertinent_frames["face"].append(frame_number)
    elif check_keypoints(row, mandatory_keypoints_profil_droit):
        pertinent_frames["profil_droit"].append(frame_number)
    elif check_keypoints(row, mandatory_keypoints_profil_gauche):
        pertinent_frames["profil_gauche"].append(frame_number)

# Trier les frames pertinentes
for key in pertinent_frames:
    pertinent_frames[key] = sorted(pertinent_frames[key])
all_frames = sorted(
    set(pertinent_frames["face"] + pertinent_frames["profil_droit"] + pertinent_frames["profil_gauche"]))

# ✅ Affichage des résultats
print(f"✅ {len(pertinent_frames['face'])} frames de vue de face détectées.")
print(f"✅ {len(pertinent_frames['profil_droit'])} frames de vue de profil droit détectées.")
print(f"✅ {len(pertinent_frames['profil_gauche'])} frames de vue de profil gauche détectées.")
print(f"✅ {len(all_frames)} frames pertinentes totales détectées.")

# ----- III) Filtrage du DataFrame df pour ne garder que les frames pertinentes -----
df = df[df['frame'].isin(all_frames)].sort_values(by='frame').reset_index(drop=True)


def local_polynomial_regression(x_frames, y_values, order=3):
    """
    Ajuste un polynôme d'ordre 'order' sur (x_frames, y_values).
    Retourne la valeur prédite au centre de x_frames.
    """
    # Supprimer les valeurs NaN pour éviter les erreurs
    mask = ~np.isnan(y_values)
    x_clean = x_frames[mask]
    y_clean = y_values[mask]

    if len(x_clean) < order + 1:
        # Pas assez de points pour ajuster un polynôme
        return np.nan

    # Ajustement du polynôme
    coefs = np.polyfit(x_clean, y_clean, order)

    # On évalue le polynôme au centre de la fenêtre
    x_center = np.mean(x_clean)
    y_pred = np.polyval(coefs, x_center)
    return y_pred




# ----- IV) Calcul de l'angle du cou -----
window_size = 30  # Taille de la fenêtre glissante
angles_cou = []  # Liste pour stocker les angles moyens du cou
poly_order = 2  # Ordre de la régression polynomiale (2 = quadratique)

# Les noms des keypoints dont on a besoin pour l'angle du cou
keypoints_needed = ["Nose", "Neck", "MidHip"]

for start in range(0, len(df) - window_size + 1, window_size):
    df_window = df.iloc[start:start + window_size]

    # Vecteur d'index pour la régression (ici, les indices de frames)
    x_frames = df_window["frame"].to_numpy()

    # Pour stocker la position polynomiale (x, y, z) de chaque keypoint
    positions = {}

    for kp in keypoints_needed:
        # Régression polynomiale sur X, Y, Z
        x_pred = local_polynomial_regression(x_frames, df_window[f"{kp}_x"].to_numpy(), order=poly_order)
        y_pred = local_polynomial_regression(x_frames, df_window[f"{kp}_y"].to_numpy(), order=poly_order)
        z_pred = local_polynomial_regression(x_frames, df_window[f"{kp}_z"].to_numpy(), order=poly_order)

        positions[kp] = np.array([x_pred, y_pred, z_pred])

    # Récupérer Nose, Neck, MidHip (polynomiaux)
    nose = positions["Nose"]
    neck = positions["Neck"]
    mid_hip = positions["MidHip"]

    # Vérification pour éviter la division par zéro ou si polynôme n'a pas pu s'ajuster
    if any(np.isnan(nose)) or any(np.isnan(neck)) or any(np.isnan(mid_hip)):
        angle_radians_cou = 0.0
    else:
        vector_neck_to_nose = nose - neck
        vector_neck_to_midhip = mid_hip - neck
        dot_product_cou = np.dot(vector_neck_to_nose, vector_neck_to_midhip)
        norm_vector_neck_to_nose = np.linalg.norm(vector_neck_to_nose)
        norm_vector_neck_to_midhip = np.linalg.norm(vector_neck_to_midhip)

        # Éviter la division par zéro
        if norm_vector_neck_to_nose < 1e-8 or norm_vector_neck_to_midhip < 1e-8:
            angle_radians_cou = 0.0
        else:
            cos_theta_cou = np.clip(
                dot_product_cou / (norm_vector_neck_to_nose * norm_vector_neck_to_midhip),
                -1.0, 1.0
            )
            sin_theta_cou = np.sqrt(1 - cos_theta_cou ** 2)
            angle_radians_cou = np.arctan2(sin_theta_cou, cos_theta_cou)

    # Conversion en degrés
    angle_degrees_cou = abs(np.degrees(angle_radians_cou))
    angles_cou.append(angle_degrees_cou)


angles_cou_test = [float(angle) for angle in angles_cou]
print(angles_cou_test)

# 📌 Calibration
angles_cou_cali = [152.48, 153.23, 152.18, 155.10, 154.95]

offsets = {"cou": np.mean([val for val in angles_cou_cali if val != 0])}
angles_effectifs = {"cou": [abs(angle - offsets["cou"]) if angle != 0 else 0 for angle in angles_cou]}

# 📋 Affichage des angles calibrés
print("\n📌 Angles effectifs après calibration :")
for segment, valeurs in angles_effectifs.items():
    angles_format = [f"Fenêtre{i}: {float(val):.2f}°" for i, val in enumerate(valeurs, start=1)]
    print(f"{segment.capitalize()} : [{', '.join(angles_format)}]")


#VISUALISATION METHODE FILTRAGE DATA

def local_polyfit_centered(df, col, half_window=15, order=3):
    """
    Retourne un array de même taille que df[col], avec pour chaque frame i,
    la prédiction polynomiale ajustée sur la fenêtre [i - half_window, i + half_window].
    """
    y_pred_array = np.full(len(df), np.nan)
    frames = df["frame"].values
    values = df[col].values

    for i in range(len(df)):
        start = max(0, i - half_window)
        end = min(len(df), i + half_window + 1)
        x_window = frames[start:end]
        y_window = values[start:end]

        pred = local_polynomial_regression(x_window, y_window, order=order)
        y_pred_array[i] = pred

    return y_pred_array


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # Change le backend vers TkAgg

df_poly = df.copy()

# Exemple sur Nose_x, Nose_y, Nose_z
for axis in ["x", "y", "z"]:
    col = f"Nose_{axis}"
    df_poly[f"Nose_{axis}_poly"] = local_polyfit_centered(df, col, half_window=15, order=2)


#MOFICATION
# IGNORER LA PREMIERE FENETRE ET LA DERNIERE FENETRE
# personne rentre dans le champ, la personne est hors champ


# Tracé : On utilise 'df_poly' pour la colonne polynomiale
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df["frame"], df["Nose_x"], 'o-', alpha=0.4, label="Original")
ax.plot(df["frame"], df_poly["Nose_x_poly"], 'r-', alpha=0.8, label="Polynôme local")
ax.set_xlabel("Frame")
ax.set_ylabel("Nose_x")
ax.legend()
plt.show()

# Rolling mean stockée dans df
df["Nose_x_rolling"] = df["Nose_x"].rolling(window=30, center=True, min_periods=1).mean()

plt.figure(figsize=(10,5))
plt.plot(df["frame"], df["Nose_x"], 'o-', alpha=0.3, label="Original")
plt.plot(df["frame"], df["Nose_x_rolling"], 'g-', alpha=0.8, label="Moyenne mobile")
# Ici on prend la colonne polynomiale dans df_poly
plt.plot(df["frame"], df_poly["Nose_x_poly"], 'r-', alpha=0.8, label="Polynôme local")
plt.legend()
plt.show()