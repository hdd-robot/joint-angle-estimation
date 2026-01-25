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
mandatory_keypoints_face = ["Nose", "Neck", "RShoulder", "RElbow", "LShoulder", "LElbow","MidHip","RHip","LHip","RAnkle","LAnkle"]
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

# ----- III) Filtrer le DataFrame df pour ne garder que all_frames -----
df = df[df['frame'].isin(all_frames)]
df = df.sort_values(by='frame').reset_index(drop=True)


# colonnes qui sont utilisées pour le calcul angulaire général
angular_keypoints = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "LHip",
    "RKnee","LKnee","RAnkle","LAnkle"
]

# Créer la liste des colonnes à vérifier (x et y)
cols_to_check = []
for kp in angular_keypoints:
    cols_to_check.append(f"{kp}_x")
    cols_to_check.append(f"{kp}_y")
    cols_to_check.append(f"{kp}_z")



# 📌 Fenêtre glissante pour calculer l'angle du cou
window_size = 30  # Taille de la fenêtre
seuil_nul = 10  # Seuil de tolérance (%) pour imposer une moyenne nulle (calibration 10, action 25)
angles_cou = []  # Liste pour stocker les angles moyens du cou
angles_coude = []
angles_genou = []
angles_epaule = []
angles_hanche = [] #IMPOSSIBLE EN 2D et VUE DE FACE ou 3/4

windows_info = [] #pour savoir les frames dans chaque fenêtre


for start in range(0, len(df) - window_size + 1, window_size):  # Avancer par fenêtres
    # Extraire les données pour les frames dans la fenêtre
    df_window = df.iloc[start:start + window_size]
    frames_in_window = df_window['frame'].tolist()
    # Remplacer les valeurs 0 par la moyenne des autres valeurs de la fenêtre
    df_window_replaced = df_window.replace(0, np.nan)  # Remplace temporairement 0 par NaN
    moyennes = df_window_replaced.mean()  # Calcule la moyenne sans les 0



    # Extraire les coordonnées moyennes des keypoints nécessaires
    nose = np.array([moyennes["Nose_x"], moyennes["Nose_y"], moyennes["Nose_z"]])
    neck = np.array([moyennes["Neck_x"], moyennes["Neck_y"], moyennes["Neck_z"]])
    mid_hip = np.array([moyennes["MidHip_x"], moyennes["MidHip_y"], moyennes["MidHip_z"]])
    r_shoulder = np.array([moyennes["RShoulder_x"], moyennes["RShoulder_y"], moyennes["RShoulder_z"]])
    r_elbow = np.array([moyennes["RElbow_x"], moyennes["RElbow_y"], moyennes["RElbow_z"]])
    r_wrist = np.array([moyennes["RWrist_x"], moyennes["RWrist_y"], moyennes["RWrist_z"]])

    l_shoulder = np.array([moyennes["LShoulder_x"], moyennes["LShoulder_y"], moyennes["LShoulder_z"]])
    l_elbow = np.array([moyennes["LElbow_x"], moyennes["LElbow_y"], moyennes["LElbow_z"]])
    l_wrist = np.array([moyennes["LWrist_x"], moyennes["LWrist_y"], moyennes["LWrist_z"]])

    r_hip = np.array([moyennes["RHip_x"], moyennes["RHip_y"], moyennes["RHip_z"]])
    r_knee = np.array([moyennes["RKnee_x"], moyennes["RKnee_y"], moyennes["RKnee_z"]])
    r_ankle = np.array([moyennes["RAnkle_x"], moyennes["RAnkle_y"], moyennes["RAnkle_z"]])


    l_hip = np.array([moyennes["LHip_x"], moyennes["LHip_y"], moyennes["LHip_z"]])
    l_knee = np.array([moyennes["LKnee_x"], moyennes["LKnee_y"], moyennes["LKnee_z"]])
    l_ankle = np.array([moyennes["LAnkle_x"], moyennes["LAnkle_y"], moyennes["LAnkle_z"]])

    r_eye = np.array([moyennes["REye_x"], moyennes["REye_y"], moyennes["REye_z"]])
    l_eye = np.array([moyennes["LEye_x"], moyennes["LEye_y"], moyennes["LEye_z"]])
    r_ear = np.array([moyennes["REar_x"], moyennes["REar_y"], moyennes["REar_z"]])
    l_ear = np.array([moyennes["LEar_x"], moyennes["LEar_y"], moyennes["LEar_z"]])

    l_big_toe = np.array([moyennes["LBigToe_x"], moyennes["LBigToe_y"], moyennes["LBigToe_z"]])
    l_small_toe = np.array([moyennes["LSmallToe_x"], moyennes["LSmallToe_y"], moyennes["LSmallToe_z"]])
    l_heel = np.array([moyennes["LHeel_x"], moyennes["LHeel_y"], moyennes["LHeel_z"]])

    r_big_toe = np.array([moyennes["RBigToe_x"], moyennes["RBigToe_y"], moyennes["RBigToe_z"]])
    r_small_toe = np.array([moyennes["RSmallToe_x"], moyennes["RSmallToe_y"], moyennes["RSmallToe_z"]])
    r_heel = np.array([moyennes["RHeel_x"], moyennes["RHeel_y"], moyennes["RHeel_z"]])

    # Calcul des vecteurs
    #cou => initialisation => soustraire à 180 °
    #a partir d'un même point
    vector_neck_to_nose = nose - neck
    vector_neck_to_midhip = mid_hip - neck

    # hanche => initialisation => soustraire à 180 °
    vector_mid_hip_to_neck = neck - mid_hip
    vector_mid_hip_to_l_hip = l_hip - mid_hip

    #epaule => initialisation => soustraire à 90 °
    vector_r_shoulder_to_midhip = mid_hip - r_shoulder
    vector_r_shoulder_to_r_elbow = r_elbow - r_shoulder

    #genou => initialisation => soustraire à 180°
    vector_r_knee_to_r_hip = r_hip - r_knee
    vector_r_knee_to_r_ankle= r_ankle - r_knee


    #coude => initialisation => soustraire à 180°
    vector_r_elbow_to_r_shoulder = r_shoulder - r_elbow
    vector_r_elbow_to_r_wrist = r_wrist - r_elbow

    #### HANCHE ###
    dot_product_hanche = np.dot(vector_mid_hip_to_neck, vector_mid_hip_to_l_hip )

# Calcul des normes
    norm_vector_mid_hip_to_neck = np.linalg.norm(vector_mid_hip_to_neck)
    norm_vector_mid_hip_to_l_hip = np.linalg.norm(vector_mid_hip_to_l_hip )

    #eviter la division par 0 si données manquantes

    if (np.allclose(mid_hip[:2], 0.0) or np.allclose(neck[:2], 0.0) or np.allclose(l_hip [:2], 0.0)):
        print("Impossible de calculer un angle hanche : l’un des vecteurs a une norme nulle.")
        angle_radians_hanche = 0.0
    else:

    # Calcul de l'angle en radians
    #     angle_radians_hanche = np.arccos(dot_product_hanche / (norm_vector_neck_to_r_hip * norm_vector_r_hip_to_l_hip ))

    #   Calcul avec atan2d
        cos_theta_hanche = dot_product_hanche / (norm_vector_mid_hip_to_neck * norm_vector_mid_hip_to_l_hip )
        cos_theta_hanche = np.clip(cos_theta_hanche, -1.0, 1.0) #Erreur numérique → Petites imprécisions dues aux flottants.

        sin_theta_hanche = np.sqrt(1 - cos_theta_hanche ** 2)
        angle_radians_hanche = np.arctan2(sin_theta_hanche, cos_theta_hanche)

    # Conversion en degrés
    angle_degrees_hanche = np.degrees(angle_radians_hanche)

    # Ajouter l'angle à la liste
    angles_hanche.append(angle_degrees_hanche)

    #### EPAULE ###
    dot_product_epaule = np.dot(vector_r_shoulder_to_midhip, vector_r_shoulder_to_r_elbow)

    # Calcul des normes
    norm_vector_r_shoulder_to_midhip= np.linalg.norm(vector_r_shoulder_to_midhip)
    norm_r_shoulder_to_r_elbow = np.linalg.norm(vector_r_shoulder_to_r_elbow)
    #eviter la division par 0 si données manquantes


    if (np.allclose(r_shoulder[:2], 0.0) or np.allclose(mid_hip[:2], 0.0) or np.allclose(r_elbow[:2], 0.0)):

        print("Impossible de calculer un angle epaule : l’un des vecteurs a une norme nulle.")
        angle_radians_epaule = 0.0
    else:

    # Calcul de l'angle en radians
    #     angle_radians_epaule = np.arccos(dot_product_epaule / (norm_vector_r_shoulder_to_mid_hip * norm_r_shoulder_to_r_wrist))

    #   Calcul avec atan2d
        cos_theta_epaule = dot_product_epaule / (norm_vector_r_shoulder_to_midhip * norm_r_shoulder_to_r_elbow)
        cos_theta_epaule = np.clip(cos_theta_epaule, -1.0, 1.0) #Erreur numérique → Petites imprécisions dues aux flottants.

        sin_theta_epaule = np.sqrt(1 - cos_theta_epaule ** 2)
        angle_radians_epaule = np.arctan2(sin_theta_epaule, cos_theta_epaule)

    # Conversion en degrés
    angle_degrees_epaule = np.degrees(angle_radians_epaule)

    # Ajouter l'angle à la liste
    angles_epaule.append(angle_degrees_epaule)

    #### GENOU #####
    dot_product_genou = np.dot(vector_r_knee_to_r_hip, vector_r_knee_to_r_ankle)

    # Calcul des normes
    norm_vector_r_knee_to_r_hip = np.linalg.norm(vector_r_knee_to_r_hip)
    norm_r_knee_to_r_ankle = np.linalg.norm(vector_r_knee_to_r_ankle)
    #eviter la division par 0 si données manquantes

    if (np.allclose(r_knee[:2], 0.0) or np.allclose(r_hip[:2], 0.0) or np.allclose(r_ankle[:2], 0.0)):
        #print("Impossible de calculer un angle : l’un des vecteurs a une norme nulle.")
        angle_radians_genou = 0.0
    else:

    # Calcul de l'angle en radians
        #angle_radians_genou = np.arccos(dot_product_genou / (norm_vector_r_hip_to_r_knee * norm_r_knee_to_r_ankle))
    #   Calcul avec atan2d
        cos_theta_genou = dot_product_genou / (norm_vector_r_knee_to_r_hip * norm_r_knee_to_r_ankle)
        cos_theta_genou = np.clip(cos_theta_genou, -1.0, 1.0)  # Erreur numérique → Petites imprécisions dues aux flottants.

        sin_theta_genou = np.sqrt(1 - cos_theta_genou ** 2)
        angle_radians_genou = np.arctan2(sin_theta_genou, cos_theta_genou)

    # Conversion en degrés
    angle_degrees_genou = np.degrees(angle_radians_genou)

    # Ajouter l'angle à la liste
    angles_genou.append(angle_degrees_genou)

    ######## COUDE ######
    dot_product_coude = np.dot(vector_r_elbow_to_r_shoulder, vector_r_elbow_to_r_wrist)

    # Calcul des normes
    norm_vector_r_elbow_to_r_shoulder = np.linalg.norm(vector_r_elbow_to_r_shoulder)
    norm_vector_r_elbow_to_r_wrist = np.linalg.norm(vector_r_elbow_to_r_wrist)

    #eviter la division par 0 si données manquantes

    if (np.allclose(r_elbow[:2], 0.0) or np.allclose(r_shoulder[:2], 0.0) or np.allclose(r_wrist[:2], 0.0)):
        #print("Impossible de calculer un angle : l’un des vecteurs a une norme nulle.")
        angle_radians_coude = 0.0
    else:

    # Calcul de l'angle en radians
        #angle_radians_coude = np.arccos(dot_product_coude / (norm_l_shoulder_to_l_elbow * norm_l_elbow_to_l_wrist))
    #   Calcul avec atan2d
        cos_theta_coude = dot_product_coude / (norm_vector_r_elbow_to_r_shoulder * norm_vector_r_elbow_to_r_wrist)
        cos_theta_coude = np.clip(cos_theta_coude, -1.0, 1.0) #Erreur numérique → Petites imprécisions dues aux flottants.

        sin_theta_coude = np.sqrt(1 - cos_theta_coude ** 2)
        angle_radians_coude = np.arctan2(sin_theta_coude, cos_theta_coude)

    # Conversion en degrés
    angle_degrees_coude = np.degrees(angle_radians_coude)

    # Ajouter l'angle à la liste
    angles_coude.append(angle_degrees_coude)

    ###### COU ####

    # Calcul du produit scalaire
    dot_product_cou = np.dot(vector_neck_to_nose, vector_neck_to_midhip)
    # Calcul des normes
    norm_vector_neck_to_nose = np.linalg.norm(vector_neck_to_nose)
    norm_vector_neck_to_midhip = np.linalg.norm(vector_neck_to_midhip)

    #eviter la division par 0 si données manquantes

    if (np.allclose(neck[:2], 0.0) or np.allclose(nose[:2], 0.0) or np.allclose(mid_hip[:2], 0.0)):
        #print("Impossible de calculer un angle : l’un des vecteurs a une norme nulle.")
        angle_radians_cou = 0.0
    else:

    # Calcul de l'angle en radians
        #angle_radians_cou = np.arccos(dot_product_cou / (norm_nose_to_neck * norm_neck_to_midhip))
    #   Calcul avec atan2d
        cos_theta_cou = dot_product_cou / (norm_vector_neck_to_nose * norm_vector_neck_to_midhip)
        cos_theta_cou = np.clip(cos_theta_cou, -1.0, 1.0)  # Erreur numérique → Petites imprécisions dues aux flottants.

        sin_theta_cou = np.sqrt(1 - cos_theta_cou ** 2)
        angle_radians_cou = np.arctan2(sin_theta_cou, cos_theta_cou)


    # Conversion en degrés
    angle_degrees_cou = np.degrees(angle_radians_cou)

    # Ajouter l'angle à la liste
    angles_cou.append(angle_degrees_cou)
    # Mémoriser info de cette fenêtre
    windows_info.append({
        "start_index": start,
        "end_index": start + window_size - 1,
        "frames": frames_in_window,
    })

# Afficher le résumé final avec un numéro de fenêtre
for i, w in enumerate(windows_info, start=1):  # `start=1` pour commencer à 1 au lieu de 0
    print(f"Fenêtre {i:<3} | Frames={w['frames']}")


angles_cou = [float(angle) for angle in angles_cou]
angles_coude = [float(angle) for angle in angles_coude]
angles_genou = [float(angle) for angle in angles_genou]
angles_epaule = [float(angle) for angle in angles_epaule]
angles_hanche = [float(angle) for angle in angles_hanche]

#UTILE UNIQUEMENT POUR CALIBRATION
# # Afficher les angles pour chaque fenêtre
print("Angles pour chaque fenêtre glissante :")
print(angles_cou)
print(angles_coude)
print(angles_genou)
print(angles_epaule)
print(angles_hanche)


#Angles (Position debout) pour la calibration pour chaque fenêtre glissante :

angles_cou_cali = [145.23288185186254, 155.51361197093328, 152.80980824750682, 154.69932272435503, 155.7907113746624]
angles_coude_cali = [156.82726249637682, 162.41863464383155, 162.99109099266363, 161.32192024239404, 161.8683546357431]
angles_genou_cali = [148.2444394598532, 177.0693962700636, 179.08393594155913, 178.07629352033604, 174.31848044360203]
angles_epaule_cali = [34.78183930938488, 31.999961686844994, 33.03967339308437, 34.78622226320265, 33.41754051885848]
angles_hanche_cali = [110.0533083029013, 93.08029182289043, 92.94257979487107, 92.63630794452114, 99.38144777283786]


# 📌 Fonction pour calculer la moyenne sans les zéros
def moyenne_sans_zero(liste, skip_n=1):
    # Exclure les n premiers éléments
    liste_filtrée = liste[skip_n:]
    # Supprimer les NaN et les 0.0
    valeurs_filtrees = [val for val in liste_filtrée if val != 0.0 and not np.isnan(val)]

    # Vérifier si la liste n'est pas vide après filtrage
    if valeurs_filtrees:
        return np.mean(valeurs_filtrees)
    else:
        return 0.0  # Si tout est NaN ou 0.0, renvoyer 0.0



# 📌 Calcul des offsets pour calibration
offsets = {
    "cou": moyenne_sans_zero(angles_cou_cali),
    "coude": moyenne_sans_zero(angles_coude_cali),
    "genou": moyenne_sans_zero(angles_genou_cali),
    "epaule": moyenne_sans_zero(angles_epaule_cali),
    "hanche": moyenne_sans_zero(angles_hanche_cali)
}

#POUR LA VERIFICATION DE CALIBRATION
# # Affichage des valeurs de calibration
# print("\nOffsets de calibration (moyenne sans les zéros) :")
# for segment, valeur in offsets.items():
#     print(f"{segment.capitalize()} : {valeur:.2f}°")

# Offsets de calibration (moyenne sans les zéros) :
# Cou : 154.39°
# Coude : 161.99°
# Genou : 178.43°
# Epaule : 33.13°
# Hanche : 161.30°

# 📌 Fonction pour appliquer la calibration à de nouveaux angles dynamiques
def appliquer_calibration(angles_dynamiques, offset):
    return [abs(angle - offset) if angle != 0.0 else 0.0 for angle in angles_dynamiques] #APPLICATION DE LA VALEUR ABSOLUE




angles_dynamiques = {
    "cou": angles_cou,
    "coude": angles_coude,
    "genou": angles_genou,
    "epaule": angles_epaule,
    "hanche": angles_hanche
}

# 📌 Calcul des angles effectifs après calibration
angles_effectifs = {segment: appliquer_calibration(angles, offsets[segment]) for segment, angles in angles_dynamiques.items()}

# 📋 Affichage des angles effectifs
print("\n📌 Angles effectifs après calibration :")

###AFFICHAGE SOBRE
for segment, valeurs in angles_effectifs.items():
    angles_format = [f" {float(val):.2f}°" for val in valeurs]
    print(f"{segment.capitalize()} : [{', '.join(angles_format)}]\n")

###AFFICHAGE AVEC FENETRE (evite de compter chaque element pour comparer avec la vidéo
for segment, valeurs in angles_effectifs.items():
    angles_format = [f"F{i}: {float(val):.2f}°" for i, val in enumerate(valeurs, start=1)]
    print(f"{segment.capitalize()} : [{', '.join(angles_format)}]")


# 📌 Précision Post calibration : (sans implémentation des scores de confiance)
# Cou : [ 0.05°,  0.15°,  1.88°,  1.48°,  0.60°]
#
# Coude : [ 0.84°,  0.18°,  1.17°,  1.71°,  1.19°]
#
# Genou : [ 0.00°,  1.14°,  0.84°,  0.29°,  0.00°]
#
# Epaule : [ 0.43°,  0.51°,  0.01°,  2.15°,  2.09°]
#
# Hanche : [ 1.40°,  0.30°,  1.33°,  0.90°,  3.94°]





### Pour la calibration

# 📌 Calcul des moyennes des angles effectifs (après calibration)
moyenne_angles_effectifs = {
    segment: moyenne_sans_zero(angles) for segment, angles in angles_effectifs.items()
}
# 📋 Affichage des moyennes des angles effectifs
print("\n📌 Moyenne des angles effectifs après calibration :")
for segment, valeur in moyenne_angles_effectifs.items():
    print(f"{segment.capitalize()} : {valeur:.2f}°")


#APRES FILTRAGE DES PREMIERS VALEURS on obtient les moyennes


# Cou : [F1: 2.97°, F2: 0.60°, F3: 1.91°, F4: 0.93°, F5: 8.93°]
# Coude : [F1: 2.05°, F2: 0.49°, F3: 0.87°, F4: 2.12°, F5: 4.94°]
# Genou : [F1: 35.76°, F2: 1.68°, F3: 0.42°, F4: 1.24°, F5: 19.93°]
# Epaule : [F1: 2.72°, F2: 1.00°, F3: 0.10°, F4: 2.10°, F5: 3.95°]
# Hanche : [F1: 15.54°, F2: 1.43°, F3: 1.57°, F4: 1.87°, F5: 4.87°]


# 📌 Moyenne des angles effectifs après calibration :
# Cou : 3.09°
# Coude : 2.10°
# Genou : 5.82°
# Epaule : 1.79°
# Hanche : 2.44°


#RESULTAT ACTION

# Cou : [ 4.57°,  6.31°,  9.74°,  9.49°,  9.86°,  8.13°,  12.06°,  1.63°,  80.92°,  4.65°,  54.59°,  103.07°,  9.97°,  5.40°,  7.27°,  7.93°,  7.64°]
#
# Coude : [ 6.51°,  6.17°,  5.21°,  2.40°,  83.06°,  49.82°,  7.26°,  5.75°,  8.68°,  8.13°,  0.25°,  0.97°,  2.22°,  2.89°,  0.00°,  10.70°,  57.44°]
#
# Genou : [ 4.72°,  0.08°,  1.66°,  1.09°,  0.95°,  0.88°,  0.47°,  1.29°,  2.47°,  3.35°,  0.24°,  4.14°,  1.27°,  2.02°,  2.33°,  1.32°,  1.11°]
#
# Epaule : [ 0.24°,  0.80°,  46.61°,  55.55°,  46.58°,  48.36°,  2.97°,  0.55°,  1.13°,  3.66°,  0.64°,  0.46°,  1.51°,  0.11°,  51.39°,  71.98°,  66.48°]
#
# Hanche : [ 7.77°,  1.87°,  8.49°,  8.78°,  9.20°,  8.33°,  7.25°,  6.42°,  7.69°,  2.58°,  22.74°,  29.12°,  7.01°,  3.57°,  3.11°,  2.99°,  2.93°]
#
# Cou : [F1: 4.57°, F2: 6.31°, F3: 9.74°, F4: 9.49°, F5: 9.86°, F6: 8.13°, F7: 12.06°, F8: 1.63°, F9: 80.92°, F10: 4.65°, F11: 54.59°, F12: 103.07°, F13: 9.97°, F14: 5.40°, F15: 7.27°, F16: 7.93°, F17: 7.64°]
# Coude : [F1: 6.51°, F2: 6.17°, F3: 5.21°, F4: 2.40°, F5: 83.06°, F6: 49.82°, F7: 7.26°, F8: 5.75°, F9: 8.68°, F10: 8.13°, F11: 0.25°, F12: 0.97°, F13: 2.22°, F14: 2.89°, F15: 0.00°, F16: 10.70°, F17: 57.44°]
# Genou : [F1: 4.72°, F2: 0.08°, F3: 1.66°, F4: 1.09°, F5: 0.95°, F6: 0.88°, F7: 0.47°, F8: 1.29°, F9: 2.47°, F10: 3.35°, F11: 0.24°, F12: 4.14°, F13: 1.27°, F14: 2.02°, F15: 2.33°, F16: 1.32°, F17: 1.11°]
# Epaule : [F1: 0.24°, F2: 0.80°, F3: 46.61°, F4: 55.55°, F5: 46.58°, F6: 48.36°, F7: 2.97°, F8: 0.55°, F9: 1.13°, F10: 3.66°, F11: 0.64°, F12: 0.46°, F13: 1.51°, F14: 0.11°, F15: 51.39°, F16: 71.98°, F17: 66.48°]
# Hanche : [F1: 7.77°, F2: 1.87°, F3: 8.49°, F4: 8.78°, F5: 9.20°, F6: 8.33°, F7: 7.25°, F8: 6.42°, F9: 7.69°, F10: 2.58°, F11: 22.74°, F12: 29.12°, F13: 7.01°, F14: 3.57°, F15: 3.11°, F16: 2.99°, F17: 2.93°]
