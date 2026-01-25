# REBA 3D - Ergonomic Posture Analysis System

3D Ergonomic Posture Analysis System using REBA (Rapid Entire Body Assessment) methodology.

## Features

- Process RGB-D video from Intel RealSense cameras
- Extract 3D skeleton using OpenPose
- Calculate REBA risk scores for posture assessment
- Real-time 3D angle calculation with depth projection
- Comparison mode: 2D vs 3D scores side-by-side
- Real-time score graph visualization
- Calibration system with YAML persistence

## Installation

```bash
# Activate conda environment
conda activate POSTURE_POSE

# Install in development mode
pip install -e .
```

### OpenPose Installation

OpenPose must be installed locally with Python bindings (pyopenpose):

```bash
# Clone and build OpenPose
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON
make -j$(nproc)
```

Configure the path in `config.yaml`:

```yaml
openpose:
  path: "/home/hdd/openpose"
```

## Quick Start

### GUI Application

```bash
# Launch the GUI
python main.py

# Or via CLI
reba3d gui
```

### Complete Pipeline (CLI)

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

## GUI Features

- **Mode Offline**: Read .bag files from RealSense recordings
- **Mode Inline**: Live capture from RealSense camera
- **Calibration**: 5-second neutral position calibration
- **3D Mode**: Depth-based angle calculation with `rs2_deproject_pixel_to_point`
- **Comparison**: Side-by-side 2D vs 3D REBA scores
- **Graph**: Real-time score visualization (140px height)

## Dependencies

- Python 3.8+
- numpy, pandas, opencv-python
- pyrealsense2 (Intel RealSense SDK)
- pyopenpose (OpenPose Python bindings)
- pygame (GUI)
- ffmpeg (video processing)

## License

MIT
