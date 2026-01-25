import pyrealsense2 as rs
import numpy as np
import cv2
import os

#Affichage du répertoire de travail

print("Répertoire de travail actuel :", os.getcwd())

# Définir le chemin du fichier .bag
bag_file = os.path.expanduser("~/20250223_015719.bag")





# Vérifier si le fichier .bag existe
if not os.path.exists(bag_file):
    raise FileNotFoundError(f"Erreur : Le fichier {bag_file} n'existe pas. Vérifiez son emplacement.")

# Configuration du pipeline RealSense pour lire le fichier .bag
# Charger le fichier .bag
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file)

# Créer un gestionnaire d'alignement pour synchroniser profondeur et couleur
align_to = rs.stream.color
align = rs.align(align_to)

# Démarrer le pipeline
pipeline.start(config)

# Initialiser l'écriture vidéo
output_video_path = os.path.expanduser("~/openpose/output.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None  # Initialisé après la première image

#Variables auxiliaires
frame_count = 0  # Compteur de frames
first_frame = None  # Sauvegarde de la première image

#Boucle principale de lecture
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if not color_frame:
            print("Aucune image détectée, fin du fichier .bag ?")
            break
        # Récup data de la frame couleur
        frame = np.asanyarray(color_frame.get_data())

        # ✅ Correction de la couleur (RGB → BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #OpenCV utilise par défaut le format BGR

        # Vérifier si la première image revient (détection de boucle) pour stopper l'enregistrement
        if first_frame is None:
            first_frame = frame.copy()
        elif np.array_equal(first_frame, frame):
            print("Détection d'une boucle dans la vidéo, arrêt de l'enregistrement.")
            break

        # Initialiser VideoWriter après avoir récupéré la taille réelle de l'image
        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
        # Écriture de la frame dans le fichier de sortie
        out.write(frame)
        frame_count += 1

        # Affichage de la frame dans une fenêtre OpenCV
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interruption de l'utilisateur, arrêt de l'enregistrement.")
            break
# Gestion de la sortie (bloc finally)
finally:
    pipeline.stop()
    if out:
        out.release()
    cv2.destroyAllWindows()
print(f"✅ Vidéo enregistrée dans {output_video_path} avec {frame_count} frames.")

##Pour résumé
# Le script lit un fichier .bag RealSense (enregistrement couleur/profondeur).
# Il aligne la vidéo de profondeur sur la vidéo couleur (même dimensions).
# Il récupère les frames couleur pour en faire une vidéo .avi.
# Il effectue une conversion de l’espace de couleurs (RGB → BGR).
# Il détecte si la vidéo repart à zéro (loop) en comparant chaque frame avec la première frame.
# Il permet aussi à l’utilisateur d’arrêter l’enregistrement manuellement (touche q).
# À la fin, il sauvegarde la vidéo et ferme toutes les ressources.

