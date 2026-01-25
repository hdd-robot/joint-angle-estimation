import os
import json
import pandas as pd
import numpy as np
import re

### Objectif
# Charger les données OpenPose (position des keypoints 3D) depuis `keypoints_3d.json`
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
                if confidence < 0.6:
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

print("\n🔍 Aperçu des données :")
print(df.head(30))

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

# ----- IV) Calcul de l'angle du cou -----
window_size = 30  # Taille de la fenêtre glissante
angles_cou = []  # Liste pour stocker les angles moyens du cou

for start in range(0, len(df) - window_size + 1, window_size):
    df_window = df.iloc[start:start + window_size]

    # Moyenne des keypoints nécessaires
    nose = df_window[["Nose_x", "Nose_y", "Nose_z"]].mean().to_numpy()
    neck = df_window[["Neck_x", "Neck_y", "Neck_z"]].mean().to_numpy()
    mid_hip = df_window[["MidHip_x", "MidHip_y", "MidHip_z"]].mean().to_numpy()

    # Vérification pour éviter la division par zéro
    if np.allclose(neck[:2], 0.0) or np.allclose(nose[:2], 0.0) or np.allclose(mid_hip[:2], 0.0):
        angle_radians_cou = 0.0
    else:
        vector_neck_to_nose = nose - neck
        vector_neck_to_midhip = mid_hip - neck
        dot_product_cou = np.dot(vector_neck_to_nose, vector_neck_to_midhip)
        norm_vector_neck_to_nose = np.linalg.norm(vector_neck_to_nose)
        norm_vector_neck_to_midhip = np.linalg.norm(vector_neck_to_midhip)

        cos_theta_cou = np.clip(dot_product_cou / (norm_vector_neck_to_nose * norm_vector_neck_to_midhip), -1.0, 1.0)
        sin_theta_cou = np.sqrt(1 - cos_theta_cou ** 2)
        angle_radians_cou = np.arctan2(sin_theta_cou, cos_theta_cou)

    # Conversion en degrés
    angle_degrees_cou = abs(np.degrees(angle_radians_cou))
    angles_cou.append(angle_degrees_cou)

# 📌 Calibration
angles_cou_cali = [154.33937456570123, 154.23085141315445, 152.5082740820906, 155.86675557448706, 154.9812675647632]
offsets = {"cou": np.mean([val for val in angles_cou_cali if val != 0])}
angles_effectifs = {"cou": [abs(angle - offsets["cou"]) if angle != 0 else 0 for angle in angles_cou]}

# 📋 Affichage des angles calibrés
print("\n📌 Angles effectifs après calibration :")
for segment, valeurs in angles_effectifs.items():
    angles_format = [f"Fenêtre{i}: {float(val):.2f}°" for i, val in enumerate(valeurs, start=1)]
    print(f"{segment.capitalize()} : [{', '.join(angles_format)}]")


#VISUALISATION METHODE FILTRAGE DATA

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # Change le backend vers TkAgg


# 📌 Tracé des positions des keypoints sur le temps
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)

for i, keypoint in enumerate(["Nose", "Neck", "MidHip"]):
    for j, axis in enumerate(["x", "y", "z"]):
        axes[i, j].plot(df["frame"], df[f"{keypoint}_{axis}"], marker="o", linestyle="-", alpha=0.6)
        axes[i, j].set_title(f"{keypoint} {axis}")
        axes[i, j].set_ylabel("Position")
        axes[i, j].set_xlabel("Frame")

plt.tight_layout()
plt.show()




def apply_moving_average(df, keypoints, window_size=30):
    """
    Applique une moyenne glissante sur les coordonnées x, y, z des keypoints donnés.
    """
    df_smooth = df.copy()
    for keypoint in keypoints:
        for axis in ["x", "y", "z"]:
            col = f"{keypoint}_{axis}"
            df_smooth[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
    return df_smooth

# 📌 Appliquer la moyenne glissante aux données
df_smooth = apply_moving_average(df, ["Nose", "Neck", "MidHip"], window_size=30)


# 📌 Tracé avant/après lissage
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)

for i, keypoint in enumerate(["Nose", "Neck", "MidHip"]):
    for j, axis in enumerate(["x", "y", "z"]):
        axes[i, j].plot(df["frame"], df[f"{keypoint}_{axis}"], label="Original", alpha=0.5)
        axes[i, j].plot(df["frame"], df_smooth[f"{keypoint}_{axis}"], label="Lissé", alpha=0.8)
        axes[i, j].set_title(f"{keypoint} {axis}")
        axes[i, j].legend()

plt.tight_layout()
plt.show()


def remove_outliers_iqr(df, keypoints, alpha=3):
    """
    Supprime les valeurs aberrantes en utilisant la méthode IQR avec un facteur ajustable.

    :param df: DataFrame contenant les données.
    :param keypoints: Liste des keypoints à filtrer.
    :param alpha: Facteur multiplicatif pour l'IQR (par défaut 1.5, peut être augmenté ou diminué).
    :return: DataFrame filtré avec NaN à la place des valeurs aberrantes.
    """
    df_filtered = df.copy()

    for keypoint in keypoints:
        for axis in ["x", "y", "z"]:
            col = f"{keypoint}_{axis}"

            q1, q3 = np.percentile(df[col].dropna(), [25, 75])  # Calcul des quartiles
            iqr = q3 - q1  # Calcul de l'intervalle interquartile

            lower_bound = q1 - alpha * iqr  # Seuil bas
            upper_bound = q3 + alpha * iqr  # Seuil haut

            # Remplacer les valeurs aberrantes par NaN
            df_filtered[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), np.nan, df[col])

    return df_filtered

df_filtered = remove_outliers_iqr(df, ["Nose", "Neck", "MidHip"], alpha=2)


from scipy.ndimage import median_filter



# def remove_outliers_iqr_median(df, keypoints, alpha=1.5, median_window=5):
#     """
#     Applique un filtrage IQR suivi d'un filtrage médian pour supprimer les valeurs aberrantes.
#
#     :param df: DataFrame contenant les données.
#     :param keypoints: Liste des keypoints à filtrer.
#     :param alpha: Facteur multiplicatif pour l'IQR (1.5 par défaut).
#     :param median_window: Taille de la fenêtre pour le filtrage médian (5 par défaut).
#     :return: DataFrame filtré.
#     """
#     df_filtered = df.copy()
#
#     for keypoint in keypoints:
#         for axis in ["x", "y", "z"]:
#             col = f"{keypoint}_{axis}"
#
#             # Étape 1 : Filtrage IQR
#             q1, q3 = np.percentile(df[col].dropna(), [25, 75])
#             iqr = q3 - q1
#             lower_bound = q1 - alpha * iqr
#             upper_bound = q3 + alpha * iqr
#             df_filtered[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), np.nan, df[col])
#
#             # Étape 2 : Appliquer un filtre médian pour supprimer les diracs
#             df_filtered[col] = median_filter(df_filtered[col], size=median_window, mode='nearest')
#
#     return df_filtered
#
#
# fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
#
#
#
#
# # 📌 Appliquer le filtrage combiné
# df_filtered = remove_outliers_iqr_median(df, ["Nose", "Neck", "MidHip"], alpha=2, median_window=3)
#
# for i, keypoint in enumerate(["Nose", "Neck", "MidHip"]):
#     for j, axis in enumerate(["x", "y", "z"]):
#         col = f"{keypoint}_{axis}"
#         axes[i, j].plot(df["frame"], df[col], label="Original", alpha=0.3, marker="o", linestyle="-")
#         axes[i, j].plot(df["frame"], df_filtered[col], label="Filtré (IQR + Médian)", alpha=0.9, marker="o", linestyle="-")
#         axes[i, j].set_title(f"{keypoint} {axis}")
#         axes[i, j].legend()
#
# plt.tight_layout()
# plt.show()


def apply_median_filter(df, keypoints, median_window=5):
    """
    Applique un filtrage médian sur les coordonnées x, y, z des keypoints donnés.

    :param df: DataFrame contenant les données.
    :param keypoints: Liste des keypoints à filtrer.
    :param median_window: Taille de la fenêtre du filtre médian (5 par défaut).
    :return: DataFrame filtré.
    """
    df_filtered = df.copy()

    for keypoint in keypoints:
        for axis in ["x", "y", "z"]:
            col = f"{keypoint}_{axis}"

            # Appliquer le filtre médian
            df_filtered[col] = median_filter(df[col], size=median_window, mode='nearest')

    return df_filtered

# 📌 Appliquer le filtrage médian uniquement
df_filtered = apply_median_filter(df, ["Nose", "Neck", "MidHip"], median_window=3)

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)

for i, keypoint in enumerate(["Nose", "Neck", "MidHip"]):
    for j, axis in enumerate(["x", "y", "z"]):
        col = f"{keypoint}_{axis}"
        axes[i, j].plot(df["frame"], df[col], label="Original", alpha=0.3, marker="o", linestyle="-")
        axes[i, j].plot(df["frame"], df_filtered[col], label="Filtré (Médian)", alpha=0.9, marker="o", linestyle="-")
        axes[i, j].set_title(f"{keypoint} {axis}")
        axes[i, j].legend()

plt.tight_layout()
plt.show()



# # 📌 Appliquer le filtrage IQR avec un seuil ajustable
#
# df_filtered = remove_outliers_iqr(df, ["Nose", "Neck", "MidHip"])
# #
#
# # 📌 Tracé avant/après filtrage IQR
# fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
#
# for i, keypoint in enumerate(["Nose", "Neck", "MidHip"]):
#     for j, axis in enumerate(["x", "y", "z"]):
#         col = f"{keypoint}_{axis}"
#         axes[i, j].plot(df["frame"], df[col], label="Original", alpha=0.5, marker="o", linestyle="-")
#         axes[i, j].plot(df["frame"], df_filtered[col], label="Filtré (IQR)", alpha=0.8, marker="o", linestyle="-")
#         axes[i, j].set_title(f"{keypoint} {axis}")
#         axes[i, j].legend()
#
# plt.tight_layout()
# plt.show()



# ----- Recalcul de l'angle du cou avec les données filtrées -----
angles_cou_smooth = []  # Liste pour stocker les angles recalculés

for start in range(0, len(df_filtered) - window_size + 1, window_size):
    df_window = df_filtered.iloc[start:start + window_size]

    # Moyenne des keypoints filtrés
    nose = df_window[["Nose_x", "Nose_y", "Nose_z"]].mean().to_numpy()
    neck = df_window[["Neck_x", "Neck_y", "Neck_z"]].mean().to_numpy()
    mid_hip = df_window[["MidHip_x", "MidHip_y", "MidHip_z"]].mean().to_numpy()

    # Vérification pour éviter la division par zéro
    if np.allclose(neck[:2], 0.0) or np.allclose(nose[:2], 0.0) or np.allclose(mid_hip[:2], 0.0):
        angle_radians_cou = 0.0
    else:
        vector_neck_to_nose = nose - neck
        vector_neck_to_midhip = mid_hip - neck
        dot_product_cou = np.dot(vector_neck_to_nose, vector_neck_to_midhip)
        norm_vector_neck_to_nose = np.linalg.norm(vector_neck_to_nose)
        norm_vector_neck_to_midhip = np.linalg.norm(vector_neck_to_midhip)

        cos_theta_cou = np.clip(dot_product_cou / (norm_vector_neck_to_nose * norm_vector_neck_to_midhip), -1.0, 1.0)
        sin_theta_cou = np.sqrt(1 - cos_theta_cou ** 2)
        angle_radians_cou = np.arctan2(sin_theta_cou, cos_theta_cou)

    # Conversion en degrés
    angle_degrees_cou = abs(np.degrees(angle_radians_cou))
    angles_cou_smooth.append(angle_degrees_cou)

# 📌 Appliquer la calibration aux nouveaux angles
angles_effectifs_smooth = {"cou": [abs(angle - offsets["cou"]) if angle != 0 else 0 for angle in angles_cou_smooth]}

# 📋 Affichage des angles recalculés après filtrage
print("\n📌 Angles effectifs après filtrage et calibration :")
for segment, valeurs in angles_effectifs_smooth.items():
    angles_format = [f"Fenêtre{i}: {float(val):.2f}°" for i, val in enumerate(valeurs, start=1)]
    print(f"{segment.capitalize()} : [{', '.join(angles_format)}]")



# def detect_outliers_iqr(data):
#     """ Détection des valeurs aberrantes via la méthode des interquartiles """
#     q1, q3 = np.percentile(data, [25, 75])
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     return (data < lower_bound) | (data > upper_bound)
#
# #
# # 📌 Visualisation des données des keypoints (x, y, z) pour identifier les outliers
# keypoints_to_check = ["Nose", "Neck", "MidHip"]
# fig, axes = plt.subplots(len(keypoints_to_check), 3, figsize=(15, 10))
#
# for i, keypoint in enumerate(keypoints_to_check):
#     for j, axis in enumerate(["x", "y", "z"]):
#         sns.boxplot(data=df_window[f"{keypoint}_{axis}"], ax=axes[i, j])
#         axes[i, j].set_title(f"{keypoint} {axis}")
#
# plt.tight_layout()
# plt.show()
#
# # 📌 Comparaison de différentes méthodes pour gérer les valeurs aberrantes
# methods = {
#     "mean": lambda x: np.mean(x),
#     "median": lambda x: np.median(x),
#     "iqr_filtered_mean": lambda x: np.mean(x[~detect_outliers_iqr(x)])
# }
#
# angles_cou_methods = {method: [] for method in methods}
#
# for start in range(0, len(df) - window_size + 1, window_size):
#     df_window = df.iloc[start:start + window_size]
#
#     keypoints = {}
#     for method_name, method_func in methods.items():
#         keypoints[method_name] = {}
#         for kp in keypoints_to_check:
#             keypoints[method_name][kp] = np.array([
#                 method_func(df_window[f"{kp}_x"].dropna().to_numpy()),
#                 method_func(df_window[f"{kp}_y"].dropna().to_numpy()),
#                 method_func(df_window[f"{kp}_z"].dropna().to_numpy())
#             ])
#
#     for method_name in methods:
#         nose = keypoints[method_name]["Nose"]
#         neck = keypoints[method_name]["Neck"]
#         mid_hip = keypoints[method_name]["MidHip"]
#
#         if np.allclose(neck[:2], 0.0) or np.allclose(nose[:2], 0.0) or np.allclose(mid_hip[:2], 0.0):
#             angle_radians_cou = 0.0
#         else:
#             vector_neck_to_nose = nose - neck
#             vector_neck_to_midhip = mid_hip - neck
#             dot_product_cou = np.dot(vector_neck_to_nose, vector_neck_to_midhip)
#             norm_vector_neck_to_nose = np.linalg.norm(vector_neck_to_nose)
#             norm_vector_neck_to_midhip = np.linalg.norm(vector_neck_to_midhip)
#             cos_theta_cou = np.clip(dot_product_cou / (norm_vector_neck_to_nose * norm_vector_neck_to_midhip), -1.0,
#                                     1.0)
#             sin_theta_cou = np.sqrt(1 - cos_theta_cou ** 2)
#             angle_radians_cou = np.arctan2(sin_theta_cou, cos_theta_cou)
#
#         angle_degrees_cou = abs(np.degrees(angle_radians_cou))
#         angles_cou_methods[method_name].append(angle_degrees_cou)
#
# # 📊 Comparaison des résultats des différentes méthodes
# fig, ax = plt.subplots(figsize=(10, 5))
# for method, angles in angles_cou_methods.items():
#     ax.plot(range(len(angles)), angles, label=method)
# ax.set_title("Comparaison des méthodes de filtrage pour l'angle du cou")
# ax.set_xlabel("Fenêtre de temps")
# ax.set_ylabel("Angle (°)")
# ax.legend()
# plt.show()




## NON EFFICACE

# from numpy.polynomial.polynomial import Polynomial
#
# # 📌 Ajustement par Moindres Carrés (modèle quadratique)
# df_trend = df.copy()
# poly_order = 2  # 1 pour linéaire, 2 pour quadratique
#
# for keypoint in ["Nose", "Neck", "MidHip"]:
#     for axis in ["x", "y", "z"]:
#         x_vals = df["frame"]
#         y_vals = df[f"{keypoint}_{axis}"]
#
#         # Supprimer NaN
#         mask = ~np.isnan(y_vals)
#         poly_model = Polynomial.fit(x_vals[mask], y_vals[mask], poly_order)
#
#         df_trend[f"{keypoint}_{axis}"] = poly_model(x_vals)



# # 📌 Tracé avant/après régression
# fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
#
# for i, keypoint in enumerate(["Nose", "Neck", "MidHip"]):
#     for j, axis in enumerate(["x", "y", "z"]):
#         axes[i, j].plot(df["frame"], df[f"{keypoint}_{axis}"], label="Original", alpha=0.5)
#         axes[i, j].plot(df["frame"], df_trend[f"{keypoint}_{axis}"], label="Régression LSQ", alpha=0.8)
#         axes[i, j].set_title(f"{keypoint} {axis}")
#         axes[i, j].legend()
#
# plt.tight_layout()
# plt.show()


