import os
import json
import re
import cv2
import numpy as np

# 📌 Définition du fichier JSON contenant les keypoints 3D
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

# 📌 Liste de keypoints obligatoires pour chaque vue
mandatory_keypoints_face = ["Nose", "Neck", "RShoulder", "RElbow", "LShoulder", "LElbow", "MidHip", "LWrist", "RWrist"]
mandatory_keypoints_profil_droit = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "RHip", "MidHip"]
mandatory_keypoints_profil_gauche = ["Nose", "Neck", "LShoulder", "LElbow", "LWrist", "LHip", "MidHip"]

# 📋 Dictionnaire pour stocker les frames en fonction de la vue
pertinent_frames = {"face": [], "profil_droit": [], "profil_gauche": []}

# 📌 Fonction pour vérifier si une frame contient tous les keypoints obligatoires
def check_keypoints(person_keypoints, mandatory_list):
    """
    Vérifie si tous les keypoints dans mandatory_list sont valides (i.e., z != 0).
    """
    for kp in mandatory_list:
        if kp in person_keypoints:
            if person_keypoints[kp]["z"] == 0:  # Si le keypoint a une profondeur nulle, il est invalide
                return False
        else:
            return False  # Si le keypoint est absent, il est invalide
    return True

# 📌 Analyse du JSON et classification des frames
for frame_data in data_json:
    frame_number = frame_data["frame"]
    keypoints_3d = frame_data["keypoints_3d"]

    if len(keypoints_3d) > 0:  # Vérifier qu'au moins une personne est détectée
        person_keypoints = {keypoint_names[i]: kp for i, kp in enumerate(keypoints_3d[0]) if i < len(keypoint_names)}

        # Vérification des keypoints pour classer la frame
        if check_keypoints(person_keypoints, mandatory_keypoints_face):
            pertinent_frames["face"].append(frame_number)
        elif check_keypoints(person_keypoints, mandatory_keypoints_profil_droit):
            pertinent_frames["profil_droit"].append(frame_number)
        elif check_keypoints(person_keypoints, mandatory_keypoints_profil_gauche):
            pertinent_frames["profil_gauche"].append(frame_number)

# 📌 Trier les listes
for key in pertinent_frames:
    pertinent_frames[key] = sorted(pertinent_frames[key])

# 📌 Fusionner toutes les frames pertinentes et les trier
all_frames = sorted(set(pertinent_frames["face"] + pertinent_frames["profil_droit"] + pertinent_frames["profil_gauche"]))

# ✅ Affichage des résultats
print(f"✅ {len(pertinent_frames['face'])} frames de vue de face détectées.")
print(f"✅ {len(pertinent_frames['profil_droit'])} frames de vue de profil droit détectées.")
print(f"✅ {len(pertinent_frames['profil_gauche'])} frames de vue de profil gauche détectées.")
print(f"✅ {len(all_frames)} frames pertinentes totales détectées.")

# ---------------------------------------------------------
# 📌 Lecture de la vidéo OpenPose et affichage des frames
# ---------------------------------------------------------
video_path = os.path.expanduser("~/openpose/output_openpose.avi")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Impossible d'ouvrir la vidéo : {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ Vidéo chargée : {video_path}")
print(f"- FPS : {fps}")
print(f"- Nombre de frames : {frame_count}")
print(f"- Résolution : {frame_width}x{frame_height}")

# Vérifier qu'il y a bien des frames pertinentes
if not all_frames:
    print("❌ Aucune frame pertinente détectée, arrêt du programme.")
    cap.release()
    exit()

# 📌 Lecture et navigation dans les frames pertinentes
current_index = 0
while current_index < len(all_frames):
    frame_number = all_frames[current_index]

    # Positionner la vidéo à la frame spécifique
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number))
    ret, frame = cap.read()

    if not ret:
        print(f"📋 Frame {frame_number} introuvable ou hors limite.")
        break

    # 📌 Ajouter une annotation sur la frame
    cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame Pertinente', frame)

    # 📌 Contrôle du clavier
    key = cv2.waitKey(0)
    if key == ord('q'):
        print("📋 Interruption par l'utilisateur (touche 'q').")
        break
    elif key == 81:  # Flèche gauche
        current_index = max(0, current_index - 1)
    elif key == 83:  # Flèche droite
        current_index = min(len(all_frames) - 1, current_index + 1)

cap.release()
cv2.destroyAllWindows()
