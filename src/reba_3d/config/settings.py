"""
Configuration settings for reba_3d package.

Default paths, thresholds, and constants that can be modified via CLI or environment variables.
"""

import os

# =============================================================================
# Paths (modifiable via CLI or environment variables)
# =============================================================================

OPENPOSE_PATH = os.environ.get("REBA_OPENPOSE_PATH", "/home/hdd/openpose")
OUTPUT_DIR = os.environ.get("REBA_OUTPUT_DIR", os.path.expanduser("~/openpose_output"))

# =============================================================================
# OpenPose execution mode
# =============================================================================

# Mode d'exécution:
# - "local": OpenPose installé sur la machine (pyopenpose)
# - "v4l2": OpenPose via Docker avec streaming V4L2 (temps réel)
OPENPOSE_MODE = os.environ.get("REBA_OPENPOSE_MODE", "local")

# Configuration V4L2 - streaming temps réel
# Note: /dev/shm est un tmpfs (RAM) pour une latence minimale
V4L2_DEVICE = os.environ.get("REBA_V4L2_DEVICE", "/dev/video10")
V4L2_OPENPOSE_JSON_DIR = os.environ.get("REBA_V4L2_JSON_DIR", "/dev/shm/openpose_json")
V4L2_DEPTH_BUFFER_SIZE = int(os.environ.get("REBA_V4L2_BUFFER_SIZE", "60"))  # Buffer circulaire depth

# =============================================================================
# Recording settings
# =============================================================================

# Activer l'enregistrement des vidéos (par défaut: non)
RECORDING_ENABLED = os.environ.get("REBA_RECORDING_ENABLED", "false").lower() == "true"
RECORDING_CODEC = os.environ.get("REBA_RECORDING_CODEC", "XVID")
SAVE_RAW_VIDEO = os.environ.get("REBA_SAVE_RAW_VIDEO", "true").lower() == "true"
SAVE_OPENPOSE_VIDEO = os.environ.get("REBA_SAVE_OPENPOSE_VIDEO", "true").lower() == "true"
SAVE_ANNOTATED_VIDEO = os.environ.get("REBA_SAVE_ANNOTATED_VIDEO", "true").lower() == "true"

# =============================================================================
# Processing thresholds
# =============================================================================

# Seuil de confiance minimum pour considérer un keypoint valide
CONFIDENCE_THRESHOLD = 0.35

# Taille de la fenêtre de lissage (en frames)
WINDOW_SIZE = 30

# Ordre du polynôme pour la régression locale
POLY_ORDER = 2

# Seuil de contact au sol pour les pieds (en mètres)
FEET_CONTACT_THRESHOLD = 0.10

# Frames par seconde par défaut
FPS = 15

# =============================================================================
# OpenPose BODY_25 keypoint names
# =============================================================================

KEYPOINT_NAMES = [
    "Nose",       # 0
    "Neck",       # 1
    "RShoulder",  # 2
    "RElbow",     # 3
    "RWrist",     # 4
    "LShoulder",  # 5
    "LElbow",     # 6
    "LWrist",     # 7
    "MidHip",     # 8
    "RHip",       # 9
    "RKnee",      # 10
    "RAnkle",     # 11
    "LHip",       # 12
    "LKnee",      # 13
    "LAnkle",     # 14
    "REye",       # 15
    "LEye",       # 16
    "REar",       # 17
    "LEar",       # 18
    "LBigToe",    # 19
    "LSmallToe",  # 20
    "LHeel",      # 21
    "RBigToe",    # 22
    "RSmallToe",  # 23
    "RHeel",      # 24
]

# =============================================================================
# Keypoints required for different view classifications
# =============================================================================

# Keypoints obligatoires pour une vue de face
MANDATORY_KEYPOINTS_FACE = [
    "Nose", "Neck", "RShoulder", "RElbow", "LShoulder", "LElbow",
    "MidHip", "RHip", "LHip", "RAnkle", "LAnkle"
]

# Keypoints obligatoires pour une vue de profil droit
MANDATORY_KEYPOINTS_PROFIL_DROIT = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "RHip"
]

# Keypoints obligatoires pour une vue de profil gauche
MANDATORY_KEYPOINTS_PROFIL_GAUCHE = [
    "Nose", "Neck", "LShoulder", "LElbow", "LWrist", "LHip"
]

# Keypoints nécessaires pour construire les repères (Tête, Buste, Épaule)
NEEDED_KEYPOINTS = [
    "Nose", "Neck", "MidHip", "REye", "LEye", "RShoulder", "LShoulder",
    "LElbow", "LWrist", "RElbow", "RWrist", "RHip", "LHip",
    "RKnee", "LKnee", "RAnkle", "LAnkle"
]

# =============================================================================
# Risk level labels (French to English mapping)
# =============================================================================

RISK_LABELS_FR = {
    1: "sans risque",
    2: "risque faible",
    3: "risque faible",
    4: "risque moyen",
    5: "risque moyen",
    6: "risque moyen",
    7: "risque élevé",
    8: "risque élevé",
    9: "risque élevé",
    10: "risque élevé",
    11: "très élevé",
    12: "très élevé",
}

RISK_LABELS_EN = {
    "sans risque": "negligible risk",
    "risque faible": "low risk",
    "risque moyen": "medium risk",
    "risque élevé": "high risk",
    "très élevé": "very high risk",
}

# Couleurs ANSI pour affichage terminal
ANSI_COLORS = {
    "sans risque": "\033[92m",      # Vert
    "risque faible": "\033[94m",    # Bleu
    "risque moyen": "\033[93m",     # Jaune
    "risque élevé": "\033[91m",     # Rouge clair
    "très élevé": "\033[41m\033[97m",  # Fond rouge, texte blanc
    "invalide": "\033[90m",         # Gris
    "reset": "\033[0m",
}

# Couleurs BGR pour annotation vidéo
VIDEO_COLORS = {
    "negligible risk": (0, 200, 0),     # Vert
    "low risk": (200, 150, 0),          # Cyan
    "medium risk": (0, 150, 255),       # Orange
    "high risk": (0, 0, 255),           # Rouge
    "very high risk": (136, 47, 99),    # Violet foncé
    "invalid": (100, 100, 100),         # Gris
}
