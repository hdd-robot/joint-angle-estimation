import os
import cv2
import numpy as np

import pyrealsense2 as rs
import json

# ✅ Importer OpenPose
OPENPOSE_PATH = "/home/dwayne/openpose"
os.environ["PYTHONPATH"] += os.pathsep + OPENPOSE_PATH + "/build/python"
import pyopenpose as op

# ✅ Configurer OpenPose
params = {"model_folder": OPENPOSE_PATH + "/models/"}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# ✅ Définir le chemin du fichier .bag
bag_file = os.path.expanduser("~/20250424_143820.bag")

if not os.path.exists(bag_file):
    raise FileNotFoundError(f"Erreur : Le fichier {bag_file} n'existe pas. Vérifiez son emplacement.")

# ✅ Configurer RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file)

# Aligner la profondeur avec la couleur
align_to = rs.stream.color
align = rs.align(align_to)

# Démarrer le pipeline
profile = pipeline.start(config)

# Récupérer les paramètres de la caméra
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# ✅ Définir les chemins des fichiers
output_video_path = os.path.expanduser("~/openpose/output.avi")  # Vidéo brute
output_openpose_video_path = os.path.expanduser("~/openpose/output_openpose.avi")  # Vidéo OpenPose
keypoints_json_path = os.path.expanduser("~/openpose/keypoints_3d.json")

# ✅ Supprimer les fichiers existants pour éviter d'avoir de vieilles données
for file_path in [output_video_path, output_openpose_video_path, keypoints_json_path]:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"🗑 Fichier supprimé : {file_path}")

# ✅ Initialiser les écritures vidéo
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_raw = None  # Vidéo brute
out_openpose = None  # Vidéo OpenPose

frame_count = 0  # Compteur de frames
first_frame = None  # Stocker la première image pour détection de boucle
keypoints_3d_data = []  # Stocker les keypoints 3D
THRESHOLD_DIFFERENCE = 1.0  # Tolérance pour détecter la boucle (peut être ajustée)


def get_3d_point(depth_frame, x, y):
    """ Convertir les coordonnées 2D (x, y) en coordonnées 3D (x, y, z) avec RealSense. """
    # ✅ Vérifier les dimensions de l'image de profondeur
    depth_width = depth_frame.get_width()
    depth_height = depth_frame.get_height()

    # ✅ Clamper les valeurs (empêcher x, y d'être en dehors de l'image)
    x = max(0, min(x, depth_width - 1))
    y = max(0, min(y, depth_height - 1))

    # ✅ Obtenir la distance en profondeur
    depth_value = depth_frame.get_distance(x, y)
    if depth_value > 0:
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
        return point_3d
    return [0, 0, 0]



try:
    while True:
        # Lire les frames du .bag
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Aucune image détectée, fin du fichier .bag ?")
            break

        # Convertir la frame en tableau NumPy
        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ✅ Détection de boucle améliorée
        if first_frame is None:
            first_frame = frame.copy()
        else:
            # Calculer la différence absolue entre la première frame et la frame actuelle
            frame_difference = cv2.absdiff(first_frame, frame)
            mean_diff = np.mean(frame_difference)  # Moyenne des différences de pixels

            # Si la différence est inférieure au seuil, la boucle est détectée
            if mean_diff < THRESHOLD_DIFFERENCE:
                print(
                    f"🔁 Détection d'une boucle dans la vidéo (différence {mean_diff:.2f}), arrêt de l'enregistrement.")
                break

        # ✅ Détecter les keypoints 2D avec OpenPose
        datum = op.Datum()
        datum.cvInputData = frame
        datum_ptr = op.VectorDatum()
        datum_ptr.append(datum)
        opWrapper.emplaceAndPop(datum_ptr)

        keypoints_3d = []
        if datum_ptr[0].poseKeypoints is not None:
            for person in datum_ptr[0].poseKeypoints:
                keypoints_person_3d = []
                for kp in person:
                    x, y, confidence = kp
                    x, y = int(x), int(y)
                    point_3d = get_3d_point(depth_frame, x, y)
                    keypoints_person_3d.append({
                        "x": float(point_3d[0]), "y": float(point_3d[1]), "z": float(point_3d[2]),
                        "confidence": float(confidence)
                    })

                keypoints_3d.append(keypoints_person_3d)

            # Stocker les données des keypoints 3D
            keypoints_3d_data.append({"frame": frame_count, "keypoints_3d": keypoints_3d})
            print(f"Frame {frame_count}: Keypoints 3D - {keypoints_3d}")

        # ✅ Affichage des résultats
        cv2.imshow("OpenPose + RealSense", datum_ptr[0].cvOutputData)

        # ✅ Initialiser l'écriture vidéo après la première frame
        if out_raw is None:
            height, width, _ = frame.shape
            out_raw = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
            out_openpose = cv2.VideoWriter(output_openpose_video_path, fourcc, 30.0, (width, height))

        # ✅ Écrire les vidéos
        out_raw.write(frame)  # Vidéo brute
        out_openpose.write(datum_ptr[0].cvOutputData)  # Vidéo avec OpenPose

        frame_count += 1

        # ✅ Arrêter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interruption de l'utilisateur, arrêt de l'enregistrement.")
            break

finally:
    pipeline.stop()
    if out_raw:
        out_raw.release()
    if out_openpose:
        out_openpose.release()
    cv2.destroyAllWindows()

    # ✅ Sauvegarder les nouveaux keypoints 3D
    with open(keypoints_json_path, 'w') as f:
        json.dump(keypoints_3d_data, f, indent=4)

    print(f"✅ Vidéo brute enregistrée dans {output_video_path} avec {frame_count} frames.")
    print(f"✅ Vidéo OpenPose enregistrée dans {output_openpose_video_path} avec {frame_count} frames.")
    print(f"✅ Données 3D enregistrées dans {keypoints_json_path}")
