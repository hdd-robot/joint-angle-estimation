import os
import json
import pandas as pd
import numpy as np

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
                #frame_dict[f"{keypoint_names[i]}_x"] = keypoint["x"]
                # frame_dict[f"{keypoint_names[i]}_y"] = keypoint["y"]
                # frame_dict[f"{keypoint_names[i]}_z"] = keypoint["z"]
                frame_dict[f"{keypoint_names[i]}_confidence"] = keypoint["confidence"]
        data.append(frame_dict)

# 📌 Convertir en DataFrame Pandas
df = pd.DataFrame(data)

# 📌 Trier le DataFrame par numéro de frame
df = df.sort_values(by=["frame", "person_id"]).reset_index(drop=True)

# 📌 Afficher toutes les colonnes et les 5 premières lignes
pd.set_option('display.width', 400)  # Largeur maximale d'affichage
pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
pd.set_option('display.expand_frame_repr', False)  # Éviter le retour à la ligne
pd.set_option('display.max_rows', None)  # Affiche toutes les lignes

print("\n🔍 Aperçu des données :")
print(df.head(30))

print(df.info())


# 📌 Définir l'intervalle de frames à afficher
frame_min = 131
frame_max = 160

# 📌 Filtrer le DataFrame pour ne garder que les frames de l'intervalle
df_filtered = df[(df["frame"] >= frame_min) & (df["frame"] <= frame_max)]

# 📌 Afficher les résultats
print(f"\n🔍 Données filtrées entre les frames {frame_min} et {frame_max}:")
print(df_filtered)
#
# # 📌 Vérifier les types des colonnes
# print("\n📌 Types des colonnes :")
# print(df.dtypes)
#
# # 📌 Vérifier les valeurs manquantes
# print("\n📌 Analyse des valeurs manquantes :")
# print(df.isnull().sum())
#
# # 📌 Affichage statistique des coordonnées X, Y, Z
# print("\n📌 Statistiques des keypoints 3D :")
# print(df.describe())
