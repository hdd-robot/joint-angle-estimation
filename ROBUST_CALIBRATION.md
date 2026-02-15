# Calibration Robuste avec Filtrage MAD

Ce document explique comment utiliser les nouvelles fonctions de calibration robuste qui intègrent le filtrage par MAD (Median Absolute Deviation) pour éliminer les valeurs aberrantes.

## Vue d'ensemble

La calibration robuste offre plusieurs avantages par rapport à la calibration classique:

- **Résistance aux outliers**: Utilise la MAD pour détecter et éliminer les valeurs aberrantes
- **Statistiques circulaires**: Traite correctement les angles avec wraparound (±180°)
- **Flexibilité**: Configuration différente pour chaque segment corporel
- **Précision améliorée**: Offsets plus stables et fiables

## Configuration via YAML (Recommandé)

Le paramètre `N_Neutre` (nombre de frames utilisées pour le calcul des offsets) est désormais **configurable via le fichier `config.yaml`** :

```yaml
# config.yaml
calibration:
  # Durée de la calibration (en secondes)
  duration: 10

  # Nombre de frames à utiliser pour le calcul des offsets (N_Neutre)
  # Recommandations: 30 (minimum), 60 (optimal), 90 (maximum)
  n_neutre: 60

  # Nombre de fenêtres à ignorer au début (pour stabilité)
  skip_windows: 1

  # Fichier de sauvegarde
  save_file: "calibration_data.yaml"
```

### Valeurs recommandées selon le contexte

| `n_neutre` | Durée @30fps | Qualité | Usage |
|------------|--------------|---------|-------|
| **30** | ~1 seconde | Stable | Minimum acceptable |
| **60** | ~2 secondes | Très stable | **Recommandé** |
| **90** | ~3 secondes | Excellente | Maximum optimal |

### Paramètre `skip_windows`

- **`skip_windows: 0`** : Utilise toutes les frames depuis le début
- **`skip_windows: 1`** : Ignore la première fenêtre (plus stable, recommandé)
- **`skip_windows: 2`** : Ignore les 2 premières fenêtres

**Note** : La valeur `skip_windows` détermine combien de fenêtres de taille `n_neutre` sont ignorées au début de la calibration pour éviter les frames instables.

## Fonctions principales

### 1. Calcul d'offsets robustes

```python
from reba_3d import calculer_offsets_depuis_neutre

# Dictionnaire d'angles mesurés en position neutre
angles_dynamiques_cou = {
    "alpha_cou": [180.0, 180.0, 180.0, ...],  # Liste de mesures
    "beta_cou": [4.5, 3.4, 2.7, ...],
    "gamma_cou": [0.0, 0.0, 0.0, ...],
}

# Calculer les offsets sur les N premières frames
# Note: N_NEUTRE est configurable via config.yaml (calibration.n_neutre)
N_NEUTRE = 30  # Nombre de frames de calibration (minimum recommandé)
offsets_cou = calculer_offsets_depuis_neutre(
    angles_dynamiques_cou,
    N_NEUTRE,
    keys_circulaires={"alpha_cou", "beta_cou", "gamma_cou"}
)

# Résultat: {"alpha_cou": 180.0, "beta_cou": 3.2, "gamma_cou": 0.0}
```

### 2. Application de la calibration

```python
from reba_3d import appliquer_calibration

# Appliquer la calibration à une séquence d'angles
angles_effectifs_cou = {
    k: appliquer_calibration(
        v,                      # Séquence d'angles bruts
        offsets_cou[k],        # Offset calculé
        circulaire=True,       # Normaliser à [-180, 180)
        absval=False,          # Conserver le signe
        invert=False           # angle - offset (ou offset - angle si True)
    )
    for k, v in angles_dynamiques_cou.items()
}
```

### 3. Calibration complète (tous les segments)

```python
from reba_3d import calibrate_all_angles_robust

# Structure complète des angles pour tous les segments
angles_dynamiques = {
    "cou": {
        "alpha": [...],
        "beta": [...],
        "gamma": [...]
    },
    "buste": {
        "alpha": [...],
        "beta": [...],
        "gamma": [...]
    },
    "epaule_droite": {
        "alpha": [...],
        "beta": [...],
        "gamma": [...],
        "elevation": [...]
    },
    "coude_droit": {
        "angle": [...]
    },
    "genou_droit": {
        "angle": [...]
    },
    # ... autres segments
}

# Calibration automatique de tous les segments
# Note: N_NEUTRE est configurable via config.yaml (calibration.n_neutre)
N_NEUTRE = 60  # Valeur optimale recommandée
offsets, angles_calibres = calibrate_all_angles_robust(angles_dynamiques, N_NEUTRE)

# offsets: dictionnaire des offsets calculés
# angles_calibres: dictionnaire des angles après calibration
```

## Configuration par segment

La fonction `calibrate_all_angles_robust` applique automatiquement la configuration appropriée pour chaque segment:

| Segment | Angles circulaires | Valeur absolue | Inversion |
|---------|-------------------|----------------|-----------|
| **Cou** | alpha, beta, gamma | Non | Non |
| **Buste** | alpha, beta, gamma | Non | Non |
| **Épaule droite** | alpha, beta, gamma | Non | **Oui** |
| **Épaule gauche** | alpha, beta, gamma | Non | **Oui** |
| **Coude droit** | - | **Oui** | Non |
| **Coude gauche** | - | **Oui** | Non |
| **Genou droit** | - | **Oui** | Non |
| **Genou gauche** | - | **Oui** | Non |

## Exemple complet d'utilisation

```python
import numpy as np
from reba_3d import calibrate_all_angles_robust

# 1. Collecte des données en position neutre (30 frames)
# puis mouvement dynamique (frames suivantes)
angles_dynamiques_cou = {
    "alpha": [180.0] * 30 + [175.0, 170.0, 165.0] * 20,
    "beta": [4.0] * 30 + [10.0, 15.0, 20.0] * 20,
    "gamma": [0.0] * 30 + [-5.0, -10.0, -15.0] * 20,
}

angles_dynamiques_buste = {
    "alpha": [90.0] * 30 + [85.0, 80.0, 75.0] * 20,
    "beta": [3.0] * 30 + [5.0, 7.0, 10.0] * 20,
    "gamma": [3.0] * 30 + [0.0, -3.0, -5.0] * 20,
}

angles_dynamiques_epaule = {
    "alpha": [0.0] * 30 + [10.0, 20.0, 30.0] * 20,
    "beta": [10.0] * 30 + [20.0, 30.0, 40.0] * 20,
    "gamma": [0.0] * 30 + [5.0, 10.0, 15.0] * 20,
    "elevation": [94.0] * 30 + [80.0, 70.0, 60.0] * 20,
}

angles_dynamiques_coude = {
    "angle": [170.0] * 30 + [160.0, 150.0, 140.0] * 20,
}

angles_dynamiques_genou = {
    "angle": [178.0] * 30 + [160.0, 150.0, 140.0] * 20,
}

# 2. Regrouper tous les segments
angles_dynamiques = {
    "cou": angles_dynamiques_cou,
    "buste": angles_dynamiques_buste,
    "epaule_droite": angles_dynamiques_epaule,
    "coude_droit": angles_dynamiques_coude,
    "genou_droit": angles_dynamiques_genou,
}

# 3. Appliquer la calibration robuste
# Note: N_NEUTRE est configurable via config.yaml (calibration.n_neutre)
N_NEUTRE = 60  # Valeur optimale recommandée
offsets, angles_effectifs = calibrate_all_angles_robust(angles_dynamiques, N_NEUTRE)

# 4. Utiliser les angles calibrés
print("Offsets calculés:")
for segment, segment_offsets in offsets.items():
    print(f"  {segment}:")
    for angle_name, offset_val in segment_offsets.items():
        print(f"    {angle_name}: {offset_val:.2f}°")

print("\nAngles effectifs (premières valeurs après calibration):")
for segment, angles_dict in angles_effectifs.items():
    print(f"  {segment}:")
    for angle_name, angle_seq in angles_dict.items():
        print(f"    {angle_name}: {angle_seq[30:33]}")  # Valeurs après calibration
```

## Paramètres de configuration avancés

### Seuil MAD (k_mad)

Le paramètre `k_mad` contrôle la sensibilité du filtrage des outliers:

```python
from reba_3d import offset_robuste_lineaire, robust_circular_offset

# Plus strict (rejette plus de valeurs)
offset = offset_robuste_lineaire(angles, k_mad=2.5)

# Par défaut (bon compromis)
offset = offset_robuste_lineaire(angles, k_mad=3.5)

# Plus permissif (garde plus de valeurs)
offset = offset_robuste_lineaire(angles, k_mad=4.5)
```

### Choix entre moyenne et médiane

Pour les angles non-circulaires:

```python
# Utiliser la moyenne après filtrage MAD (défaut)
offset = offset_robuste_lineaire(angles, use_mean=True)

# Utiliser la médiane après filtrage MAD (plus robuste)
offset = offset_robuste_lineaire(angles, use_mean=False)
```

## Avantages de la méthode MAD

### 1. Résistance aux outliers

```python
# Séquence avec outliers
angles = [170.0, 171.0, 169.0, 250.0, 170.5, 50.0, 172.0]

# Moyenne simple (affectée par les outliers)
mean_simple = np.mean(angles)  # 164.6°

# Offset robuste (outliers rejetés)
offset = offset_robuste_lineaire(angles)  # 170.5°
```

### 2. Gestion correcte du wraparound

```python
# Angles près de ±180°
angles = [170.0, 175.0, 178.0, -179.0, -175.0, -170.0]

# Moyenne linéaire (INCORRECT)
mean_linear = np.mean(angles)  # 0.0° (faux!)

# Offset circulaire (CORRECT)
offset = robust_circular_offset(angles)  # 175.0°
```

## Intégration dans le pipeline REBA

La calibration robuste s'intègre directement dans le pipeline d'analyse:

```python
from reba_3d import REBAAssessor, calibrate_all_angles_robust

# 1. Charger les keypoints
keypoints_data = load_keypoints("data.json")

# 2. Extraire les angles bruts
raw_angles = extract_angles_from_keypoints(keypoints_data)

# 3. Appliquer la calibration robuste
# Note: N_NEUTRE est configurable via config.yaml (calibration.n_neutre)
N_NEUTRE = 60  # Frames de calibration au début (valeur optimale)
offsets, calibrated_angles = calibrate_all_angles_robust(raw_angles, N_NEUTRE)

# 4. Analyser avec REBA
assessor = REBAAssessor()
results = assessor.analyze_from_calibrated_angles(calibrated_angles)
```

## Comparaison: Classique vs Robuste

| Aspect | Calibration classique | Calibration robuste |
|--------|----------------------|-------------------|
| **Méthode** | Moyenne simple | Filtrage MAD + moyenne/médiane |
| **Outliers** | Affectent le résultat | Automatiquement rejetés |
| **Angles circulaires** | Problèmes de wraparound | Traitement correct |
| **Stabilité** | Variable selon les données | Très stable |
| **Vitesse** | Rapide | Légèrement plus lent |
| **Usage recommandé** | Données très propres | Données réelles (avec bruit) |

## Notes importantes

1. **Configuration de N_NEUTRE**:
   - **Méthode recommandée** : Modifier `config.yaml` (section `calibration.n_neutre`)
   - **Valeurs recommandées** :
     - Minimum : 30 frames (~1 seconde à 30 fps)
     - Optimal : **60 frames** (~2 secondes à 30 fps) - **Recommandé**
     - Maximum : 90 frames (~3 secondes à 30 fps)
   - Plus c'est long, plus c'est stable (mais attention à la patience de l'utilisateur)
   - **Important** : Assurez-vous que `calibration.duration` (durée totale) soit suffisant pour collecter au moins `n_neutre` frames

   **Exemple de configuration optimale** :
   ```yaml
   calibration:
     duration: 10          # 10 secondes (300 frames @30fps)
     n_neutre: 60          # Utiliser 60 frames (2 secondes)
     skip_windows: 1       # Ignorer la 1ère fenêtre
   ```

2. **Position neutre**:
   - Debout, bras le long du corps
   - Regarder droit devant
   - Pieds écartés à largeur d'épaules
   - Rester immobile et détendu

3. **Angles circulaires**:
   - Toujours utiliser `keys_circulaires` pour alpha, beta, gamma
   - Ne pas l'utiliser pour l'élévation (angle 0-180°)

4. **Inversion (épaules)**:
   - Pour les épaules: `invert=True` (offset - angle)
   - Pour les autres: `invert=False` (angle - offset)

## Dépannage

### Problème: Offsets aberrants

```python
# Vérifier les données brutes
print(f"Frames neutres: {angles_dict['alpha'][:N_NEUTRE]}")
print(f"Min: {min(angles_dict['alpha'][:N_NEUTRE])}")
print(f"Max: {max(angles_dict['alpha'][:N_NEUTRE])}")

# Augmenter N_NEUTRE si trop de variation
# Méthode 1: Via config.yaml (RECOMMANDÉ)
# Éditez config.yaml:
#   calibration:
#     n_neutre: 90  # Au lieu de 30

# Méthode 2: En code (pour tests)
N_NEUTRE = 90  # Au lieu de 30
```

### Problème: Angles négatifs inattendus

```python
# Pour les angles qui doivent être positifs, utiliser absval=True
angles_calibres = appliquer_calibration(
    angles_bruts,
    offset,
    circulaire=False,
    absval=True,  # Force la valeur absolue
    invert=False
)
```

## Références

- **MAD (Median Absolute Deviation)**: Méthode statistique robuste pour détecter les outliers
- **Statistiques circulaires**: Traitement mathématique des angles avec wraparound
- **Facteur 1.4826**: Conversion MAD → écart-type pour distribution normale

## Configuration complète de la calibration

Voici un exemple de configuration complète dans `config.yaml` :

```yaml
# config.yaml
calibration:
  # Durée totale de la calibration (en secondes)
  # L'utilisateur doit rester immobile pendant cette durée
  duration: 10

  # Nombre de frames utilisées pour calculer les offsets (N_Neutre)
  # Ces frames sont prises au début de la période de calibration
  # Recommandations:
  #   - 30 frames = ~1 seconde @30fps (minimum)
  #   - 60 frames = ~2 secondes @30fps (optimal, recommandé)
  #   - 90 frames = ~3 secondes @30fps (maximum)
  n_neutre: 60

  # Nombre de fenêtres à ignorer au début
  # Permet d'éviter les frames instables au démarrage
  # Recommandé: 1 (ignore les 30 premières frames si n_neutre=30)
  skip_windows: 1

  # Fichier de sauvegarde des offsets calculés
  save_file: "calibration_data.yaml"
```

### Impact de la configuration

**Scénario avec `duration=10s`, `n_neutre=60`, `skip_windows=1` @30fps** :

1. **Collecte** : 10s × 30fps = **300 frames** au total
2. **Fenêtres** : 300 frames ÷ 60 (window_size) = 5 fenêtres possibles
3. **Utilisation** :
   - Fenêtre 0 (frames 0-59) : **IGNORÉE** (`skip_windows=1`)
   - Fenêtre 1 (frames 60-119) : **UTILISÉE** pour calcul des offsets
   - Fenêtres 2-4 : Non utilisées (mais disponibles si needed)
4. **Résultat** : Offsets calculés sur 60 frames (frames 60-119) avec filtrage MAD

## Support

Pour plus d'informations, consultez:
- [README.md](src/README.md) - Documentation principale du projet
- [USAGE_2D_3D_ANGLES.md](USAGE_2D_3D_ANGLES.md) - Guide d'utilisation des angles 2D/3D
- [config.yaml](src/config.yaml) - Fichier de configuration (section `calibration`)
- Code source: [src/reba_3d/core/robust_calibration.py](src/reba_3d/core/robust_calibration.py)
