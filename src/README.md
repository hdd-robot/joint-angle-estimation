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
| 1 | Negligible |
| 2-3 | Low |
| 4-6 | Medium |
| 7-10 | High |
| 11+ | Very High |

## Dependencies

- Python 3.8+
- numpy, pandas, opencv-python
- pyrealsense2 (Intel RealSense SDK)
- pyopenpose (OpenPose Python bindings)
- ffmpeg

## License

MIT
