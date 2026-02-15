# REBA 3D - Ergonomic Posture Analysis System

3D Ergonomic Posture Analysis System using REBA (Rapid Entire Body Assessment) methodology. Processes RGB-D video from Intel RealSense cameras with OpenPose skeleton detection to assess worker posture risk levels in occupational settings.

![Software Screenshot](../img/software.png)

## Features

- Process RGB-D video from Intel RealSense cameras (.bag files)
- Extract 3D skeleton using OpenPose (BODY_25 model - 25 keypoints)
- Calculate REBA risk scores for posture assessment
- Real-time 3D angle calculation with depth projection (`rs2_deproject_pixel_to_point`)
- Comparison mode: 2D vs 3D scores side-by-side
- Real-time score graph visualization
- Calibration system with YAML persistence
- Video annotation with risk level overlays

## Project Structure

```
reba_3d/
├── __init__.py                 # Package exports
├── __main__.py                 # Entry point: python -m reba_3d
├── cli.py                      # Command-line interface
│
├── config/
│   ├── settings.py             # Paths, thresholds, constants
│   ├── yaml_config.py          # YAML configuration loader
│   ├── calibration_store.py    # Calibration persistence
│   └── calibration_data.py     # Calibration offsets
│
├── capture/
│   └── realsense_capture.py    # RealSense + OpenPose integration
│
├── core/
│   ├── angles.py               # 2D/3D angle calculation
│   ├── geometry.py             # Gram-Schmidt, nautical angles
│   ├── smoothing.py            # Polynomial regression
│   ├── keypoints.py            # JSON loading, confidence filtering
│   └── frame_classifier.py     # Face/profile classification
│
├── reba/
│   ├── tables.py               # TABLE_A, TABLE_B, TABLE_C constants
│   ├── angles.py               # Angle calculation per segment
│   ├── scoring.py              # REBA scoring functions
│   ├── calibration.py          # Calibration offset application
│   ├── realtime_scorer.py      # Real-time scoring with smoothing
│   └── risk_assessment.py      # Main REBAAssessor class
│
├── visualization/
│   ├── annotator.py            # Video annotation with risks
│   └── viewer.py               # Interactive frame viewer
│
├── gui/
│   ├── app.py                  # Main Pygame application
│   └── components.py           # UI components (buttons, panels, graph)
│
├── io/
│   ├── json_io.py              # JSON read/write helpers
│   └── video_io.py             # VideoReader/VideoWriter classes
│
└── utils/
    └── logger.py               # Logging configuration
```

## Prerequisites

### System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support (required for OpenPose)
- **CUDA**: 11.x or compatible version
- **cuDNN**: Compatible with your CUDA version

### Hardware

- Intel RealSense D400 series camera (D415, D435, D455)

## Installation

### 1. Create Conda Environment

```bash
conda create -n POSTURE_POSE python=3.10
conda activate POSTURE_POSE
```

### 2. Install OpenPose

OpenPose must be installed locally with Python bindings (pyopenpose):

```bash
# Install dependencies
sudo apt-get install cmake libopencv-dev

# Clone OpenPose
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose
git submodule update --init --recursive

# Build with Python bindings
mkdir build && cd build
cmake .. \
    -DBUILD_PYTHON=ON \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DGPU_MODE=CUDA
make -j$(nproc)

# Download models
cd .. && ./models/getModels.sh
```

### 3. Install REBA 3D

```bash
cd /path/to/joint-angle-estimation/src
pip install -e .
```

### 4. Configure OpenPose Path

Edit `config.yaml`:

```yaml
openpose:
  path: "/path/to/openpose"
```

Or set environment variable:

```bash
export REBA_OPENPOSE_PATH="/path/to/openpose"
```

## Quick Start

### GUI Application

```bash
# Launch the GUI
python main.py

# Or via CLI
reba3d gui

# With specific .bag directory
reba3d gui --bag-dir /path/to/bags --bag recording.bag
```

### Complete Pipeline (CLI)

```bash
reba3d pipeline --bag recording.bag --output ./output
```

### Individual Commands

```bash
# Step 1: Extract 3D skeleton from .bag file
reba3d capture --bag recording.bag --output ./output

# Step 2: Calculate REBA scores
reba3d analyze --input ./output/keypoints_3d.json --output ./output/risk_times.json

# Step 3: Annotate video with risk levels
reba3d annotate --video ./output/output_openpose.avi --risks ./output/risk_times.json

# Interactive frame viewer
reba3d view --video ./output/output_openpose.avi --keypoints ./output/keypoints_3d.json
```

### Python API

```python
from reba_3d import REBAAssessor

# Analyze keypoints
assessor = REBAAssessor()
results = assessor.analyze("keypoints_3d.json")
assessor.print_summary()

# Export risk times for video annotation
risk_times = assessor.get_risk_times_for_video()
```

## Keyboard Shortcuts

| Key     | Action                          |
|---------|--------------------------------|
| `SPACE` | Start/Stop capture             |
| `P`     | Pause/Resume (offline mode)    |
| `ESC`   | Quit application               |

## GUI Features

- **Mode Offline**: Read .bag files from RealSense recordings
- **Mode Inline**: Live capture from RealSense camera (requires connected camera)
- **Calibration**: 5-second neutral position calibration (stand in T-pose)
- **3D Mode**: Depth-based angle calculation using `rs2_deproject_pixel_to_point`
- **Comparison**: Side-by-side 2D vs 3D REBA scores
- **Graph**: Real-time score visualization (140px height, green=3D, blue=2D)
- **Scores Toggle**: Show/hide REBA scores overlay

## Output Files

After running the pipeline, the following files are generated:

| File | Description |
|------|-------------|
| `keypoints_3d.json` | 3D skeleton keypoints for each frame |
| `risk_times.json` | REBA scores and risk levels per time window |
| `output.avi` | Raw video (if recording enabled) |
| `output_openpose.avi` | Video with skeleton overlay |
| `output_openpose_annotated.avi` | Video with REBA risk annotations |

## Configuration

All settings can be configured in `config.yaml`:

```yaml
# GUI settings
gui:
  width: 1280
  height: 720
  fps: 30

# RealSense camera
realsense:
  color_width: 640
  color_height: 480
  realtime_playback: true

# REBA processing
reba:
  confidence_threshold: 0.35  # Keypoint confidence threshold
  window_size: 30             # Smoothing window (frames)
  video_fps: 15               # FPS for time calculations

# Video recording (disabled by default)
recording:
  enabled: false
  codec: "XVID"
  save_raw_video: true
  save_openpose_video: true
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `REBA_OPENPOSE_PATH` | Path to OpenPose installation |
| `REBA_OUTPUT_DIR` | Default output directory |
| `REBA_RECORDING_ENABLED` | Enable video recording (`true`/`false`) |

## Risk Levels

| REBA Score | Risk Level  | Action Required |
|------------|-------------|-----------------|
| 1          | Negligible  | None necessary |
| 2-3        | Low         | May be necessary |
| 4-6        | Medium      | Necessary |
| 7-10       | High        | Necessary soon |
| 11+        | Very High   | Necessary immediately |

## OpenPose BODY_25 Keypoints

The system uses OpenPose's BODY_25 model with 25 keypoints:

```
 0: Nose           1: Neck           2: RShoulder
 3: RElbow         4: RWrist         5: LShoulder
 6: LElbow         7: LWrist         8: MidHip
 9: RHip          10: RKnee         11: RAnkle
12: LHip          13: LKnee         14: LAnkle
15: REye          16: LEye          17: REar
18: LEar          19: LBigToe       20: LSmallToe
21: LHeel         22: RBigToe       23: RSmallToe
24: RHeel
```

## Dependencies

### Python Packages

```
numpy>=1.20.0
pandas>=1.3.0
opencv-python>=4.5.0
pygame>=2.1.0
pyyaml>=6.0
pyrealsense2>=2.50.0
```

### System Dependencies

- **OpenPose**: With Python bindings (pyopenpose)
- **ffmpeg**: For video recalibration
- **Intel RealSense SDK**: librealsense2

## Troubleshooting

### OpenPose not found

```bash
# Verify pyopenpose is accessible
python -c "import sys; sys.path.insert(0, '/path/to/openpose/build/python/openpose'); import pyopenpose"
```

### CUDA out of memory

Reduce OpenPose resolution in `realsense_capture.py`:

```python
params["net_resolution"] = "-1x128"  # Lower resolution
```

### RealSense camera not detected

```bash
# Check if camera is recognized
rs-enumerate-devices

# Install udev rules if needed
sudo cp /path/to/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

## License

GPL-3.0
