=============
Configuration
=============

Le fichier ``config.yaml`` contient tous les paramètres configurables de l'application REBA 3D.

Interface graphique (gui)
=========================

Paramètres de la fenêtre principale de l'application.

.. code-block:: yaml

   gui:
     width: 1280          # Largeur de la fenêtre en pixels
     height: 720          # Hauteur de la fenêtre en pixels
     title: "REBA 3D"     # Titre affiché
     fps: 30              # Fréquence de rafraîchissement de l'interface
     left_panel_width: 180    # Largeur du panneau gauche (boutons)
     right_panel_width: 250   # Largeur du panneau droit (logs)

Caméra RealSense
================

Configuration de la caméra Intel RealSense pour la capture RGB-D.

.. code-block:: yaml

   realsense:
     color_width: 640     # Résolution horizontale (couleur)
     color_height: 480    # Résolution verticale (couleur)
     color_fps: 30        # FPS du flux couleur
     depth_width: 640     # Résolution horizontale (profondeur)
     depth_height: 480    # Résolution verticale (profondeur)
     depth_fps: 30        # FPS du flux profondeur
     realtime_playback: true  # Lecture temps réel des fichiers .bag

.. note::
   ``realtime_playback: true`` lit les fichiers .bag à vitesse normale.
   Mettre à ``false`` pour une lecture accélérée (traitement par lots).

Chemins et répertoires (paths)
==============================

Emplacements des fichiers d'entrée et de sortie.

.. code-block:: yaml

   paths:
     bag_directory: "bag"           # Dossier contenant les fichiers .bag
     default_bag_file: "recording.bag"  # Fichier .bag par défaut
     output_directory: "output"     # Dossier de sortie des résultats

OpenPose
========

Configuration de l'installation OpenPose locale.

.. code-block:: yaml

   openpose:
     path: "/home/hdd/openpose"   # Chemin vers l'installation OpenPose

.. warning::
   Le chemin doit pointer vers le répertoire racine d'OpenPose contenant
   les dossiers ``models/`` et ``build/``.

Variables d'environnement alternatives :

- ``REBA_OPENPOSE_PATH`` : Chemin OpenPose
- ``REBA_OUTPUT_DIR`` : Répertoire de sortie

Enregistrement vidéo (recording)
================================

Options d'enregistrement des vidéos de sortie.

.. code-block:: yaml

   recording:
     enabled: false              # Activer/désactiver l'enregistrement
     codec: "XVID"               # Codec vidéo (XVID, MJPG, mp4v)
     save_raw_video: true        # Sauvegarder la vidéo brute
     save_openpose_video: true   # Sauvegarder avec squelette OpenPose
     save_annotated_video: true  # Sauvegarder avec annotations REBA

Pour activer l'enregistrement :

- Via ``config.yaml`` : ``recording.enabled: true``
- Via CLI : ``reba3d gui --save-video``
- Via environnement : ``REBA_RECORDING_ENABLED=true``

Traitement REBA
===============

Paramètres du calcul REBA.

.. code-block:: yaml

   reba:
     confidence_threshold: 0.35   # Confiance minimale des keypoints (0.0-1.0)
     window_size: 30              # Fenêtre de lissage (frames)
     poly_order: 2                # Ordre du polynôme de lissage
     feet_contact_threshold: 0.10 # Seuil contact au sol (mètres)
     video_fps: 15                # FPS pour calcul temporel

     # Scores manuels (voir page "Scores REBA")
     load_score: 0        # Score charge/force (0-3)
     coupling_score: 0    # Score qualité de prise (0-3)
     activity_score: 0    # Score activité (0-3)

Lissage des données
-------------------

Le système utilise **deux méthodes de lissage différentes** selon le mode d'utilisation :

Mode temps réel (caméra live)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

En temps réel, le lissage utilise une **moyenne mobile causale** des 10 derniers scores :

.. code-block:: text

   Score affiché = moyenne(score[-10], score[-9], ..., score[-1], score[0])
                            ↑ passé                              ↑ présent

- **Pas de décalage temporel** : seules les frames passées sont utilisées
- **Buffer de 10 frames** : lissage léger pour éviter les oscillations
- Les paramètres ``window_size`` et ``poly_order`` ne sont **pas utilisés** en temps réel

Mode batch (fichiers .bag)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Pour l'analyse offline de fichiers .bag, le lissage utilise une **régression polynomiale centrée** :

.. code-block:: text

   Frames:  [0  1  2  ... 14  15  16 ... 28  29]
                            ↑
                         Centre
                     (résultat calculé)

- ``window_size: 30`` : Fenêtre de 30 frames
- ``poly_order: 2`` : Polynôme quadratique
- **Décalage temporel** : ``window_size / 2`` = 15 frames

Calcul du décalage (mode batch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------+------------+-------------------+
| window_size    | video_fps  | Décalage          |
+================+============+===================+
| 30             | 15         | 15 frames = 1.0s  |
+----------------+------------+-------------------+
| 30             | 30         | 15 frames = 0.5s  |
+----------------+------------+-------------------+
| 10             | 15         | 5 frames = 0.33s  |
+----------------+------------+-------------------+

.. tip::
   En mode batch, augmenter ``window_size`` pour des trajectoires plus fluides
   mais avec plus de décalage. Diminuer pour plus de réactivité mais plus de bruit.

Calibration
===========

Paramètres de la phase de calibration posturale.

.. code-block:: yaml

   calibration:
     duration: 5                    # Durée en secondes
     skip_initial_frames: 1         # Frames ignorées au début
     save_file: "calibration_data.yaml"  # Fichier de sauvegarde

.. note::
   La calibration enregistre les angles de référence quand le sujet
   est en position neutre (debout, bras le long du corps).

Affichage (display)
===================

Options d'affichage des scores et annotations.

.. code-block:: yaml

   display:
     show_scores_on_start: true   # Afficher les scores au démarrage
     show_3d_angles: true         # Afficher les angles 3D
     show_2d_scores: true         # Afficher les scores 2D

     # Couleurs des niveaux de risque (format BGR)
     risk_colors:
       negligible: [0, 200, 0]    # Vert
       low: [200, 150, 0]         # Cyan
       medium: [0, 150, 255]      # Orange
       high: [0, 0, 255]          # Rouge
       very_high: [136, 47, 99]   # Violet

Logs
====

Configuration du panneau de logs.

.. code-block:: yaml

   logging:
     max_lines: 35          # Nombre maximum de lignes affichées
     font_size: 16          # Taille de la police
     save_to_file: false    # Sauvegarder dans un fichier
     log_file: "reba_3d.log"

Export
======

Options d'export des résultats.

.. code-block:: yaml

   export:
     format: "json"                 # Format: json ou csv
     include_raw_angles: true       # Inclure les angles bruts
     include_detailed_scores: true  # Inclure les scores détaillés
