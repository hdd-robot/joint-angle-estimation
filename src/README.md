# REBA 3D - Ergonomic Posture Analysis System

3D Ergonomic Posture Analysis System using REBA (Rapid Entire Body Assessment) methodology.

## Features

- Process RGB-D video from Intel RealSense cameras
- Extract 3D skeleton using OpenPose
- Calculate REBA risk scores for posture assessment
- Annotate videos with risk level overlays
- Interactive frame viewer for analysis

## Installation

```bash
# Activate conda environment
conda activate POSTURE_POSE

# Install in development mode
pip install -e .
```

## Quick Start

### Complete Pipeline

```bash
reba3d pipeline --bag recording.bag --output ./output
```

### Individual Commands

```bash
# Step 1: Extract 3D skeleton
reba3d capture --bag recording.bag --output ./output

# Step 2: Calculate REBA scores
reba3d analyze --input keypoints_3d.json --output risk_times.json

# Step 3: Annotate video
reba3d annotate --video output_openpose.avi --risks risk_times.json

# Interactive viewer
reba3d view --video output_openpose.avi --keypoints keypoints_3d.json
```

### Python API

```python
from reba_3d import REBAAssessor

assessor = REBAAssessor()
results = assessor.analyze("keypoints_3d.json")
assessor.print_summary()

# Export risk times
risk_times = assessor.get_risk_times_for_video()
```

## Risk Levels

| REBA Score | Risk Level |
|------------|------------|
| 1          | Negligible |
| 2-3        | Low        |
| 4-6        | Medium     |
| 7-10       | High       |
| 11+        | Very High  |

## OpenPose Configuration

Le système supporte deux modes d'exécution pour OpenPose :

| Mode     | Description                               | Temps réel  |
|----------|-------------------------------------------|-------------|
| `v4l2`   | Streaming via webcam virtuelle V4L2       | **Oui**     |
| `local`  | OpenPose installé localement (pyopenpose) | Oui         | # pas encore testé 

### Mode V4L2 (Recommandé)

Ce mode utilise une webcam virtuelle Linux (V4L2 loopback) pour streamer les frames RGB vers OpenPose en temps réel.

**Étape 1 : Créer le device V4L2 loopback**

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="RGB_OpenPose" exclusive_caps=1
```

> **Important :** L'option `exclusive_caps=1` est nécessaire pour que OpenPose puisse lire le device.

> **Note :** Pour charger automatiquement au démarrage, ajoutez dans `/etc/modprobe.d/v4l2loopback.conf` :
> ```
> options v4l2loopback devices=1 video_nr=10 card_label="RGB_OpenPose" exclusive_caps=1
> ```

**Étape 2 : Créer le répertoire RAM pour les JSON**

```bash
mkdir -p /dev/shm/openpose_json
```

> **Important :** `/dev/shm` est un `tmpfs` (système de fichiers en RAM) déjà monté par défaut sur Linux. Il ne faut **PAS** le créer manuellement. Seul le sous-répertoire `openpose_json` doit être créé.
>
> Vérification :
> ```bash
> df -h /dev/shm
> # Type: tmpfs → les données sont en RAM
> ```

**Étape 3 : Lancer OpenPose Docker avec --camera**

```bash
docker run -it \
  --privileged \
  --gpus all \
  --ipc=host \
  --device=/dev/video10 \
  -v /dev/shm/openpose_json:/data/out \
  cwaffles/openpose \
  ./build/examples/openpose/openpose.bin \
  --camera 10 \
  --write_json /data/out \
  --model_pose COCO \
  --net_resolution 320x176 \
  --number_people_max 1 \
  --display 0
```

> **Important :** L'option `--privileged` est nécessaire pour que Docker puisse accéder au device V4L2.

> **Note :** Si vous obtenez l'erreur suivante :
> ```
> docker: Error response from daemon: Conflict. The container name "/openpose_srv" is already in use...
> ```
> Supprimez d'abord le conteneur existant :
> ```bash
> docker rm -f openpose_srv
> ```

**Étape 4 : Lancer l'application**

```bash
reba3d pipeline --bag recording.bag --openpose-mode v4l2
```

Ou avec des paramètres personnalisés :

```bash
reba3d pipeline --bag recording.bag \
  --openpose-mode v4l2 \
  --v4l2-device /dev/video10 \
  --v4l2-json-dir /dev/shm/openpose_json
```

**Architecture V4L2 :**

```
Caméra 3D (RealSense)
  ├── RGB  ──> V4L2 loopback (/dev/video10) ──> OpenPose Docker (--camera)
  │                                                    │
  │                                              JSON keypoints
  │                                              (/dev/shm = RAM)
  │                                                    ↓
  └── Depth ──> Buffer circulaire ──────────────> Projection 3D
```

### Mode Local (pyopenpose)

Si OpenPose est installé sur votre machine avec les bindings Python :

```yaml
# config.yaml
openpose:
  mode: "local"
  local:
    path: "~/openpose_alt"
```

```bash
reba3d pipeline --bag recording.bag --openpose-mode local
```

## Dependencies

- Python 3.8+
- numpy, pandas, opencv-python
- pyrealsense2 (Intel RealSense SDK)
- pyopenpose (OpenPose Python bindings) - *ou Docker*
- ffmpeg

## License

MIT
