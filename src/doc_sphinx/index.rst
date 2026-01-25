=====================================
REBA 3D - Analyse Posturale Ergonomique
=====================================

.. image:: https://img.shields.io/badge/version-1.0.0-blue.svg
   :alt: Version 1.0.0

Bienvenue dans la documentation de **REBA 3D**, un système d'analyse posturale ergonomique basé sur la méthodologie REBA (Rapid Entire Body Assessment).

Présentation
============

REBA 3D est un outil d'analyse ergonomique qui utilise :

- **Caméra Intel RealSense** : Capture RGB-D pour obtenir les coordonnées 3D
- **OpenPose** : Détection du squelette en temps réel (modèle BODY_25)
- **Méthodologie REBA** : Évaluation standardisée des risques posturaux

Le système analyse les postures des travailleurs et attribue un score de risque de 1 à 12.

Niveaux de risque
-----------------

+-------------+-------------------+------------------------+
| Score       | Niveau de risque  | Action requise         |
+=============+===================+========================+
| 1           | Négligeable       | Aucune action          |
+-------------+-------------------+------------------------+
| 2-3         | Faible            | Changement possible    |
+-------------+-------------------+------------------------+
| 4-6         | Moyen             | Investigation requise  |
+-------------+-------------------+------------------------+
| 7-10        | Élevé             | Investigation urgente  |
+-------------+-------------------+------------------------+
| 11-12       | Très élevé        | Action immédiate       |
+-------------+-------------------+------------------------+

Installation rapide
===================

.. code-block:: bash

   # Activer l'environnement conda
   conda activate POSTURE_POSE

   # Installer le package
   pip install -e .

   # Lancer l'interface graphique
   reba3d gui

Sommaire
========

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   configuration
   scores_reba

Indices et tables
=================

* :ref:`genindex`
* :ref:`search`
