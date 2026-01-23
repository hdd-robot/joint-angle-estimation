# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D Ergonomic Posture Analysis System using REBA (Rapid Entire Body Assessment) methodology. Processes RGB-D video from Intel RealSense cameras with OpenPose skeleton detection to assess worker posture risk levels in occupational settings.

## Package Structure

```
reba_3d/
├── __init__.py                 # Package exports
├── __main__.py                 # Entry point: python -m reba_3d
├── cli.py                      # Command-line interface
│
├── config/
│   ├── settings.py             # Paths, thresholds, constants
│   └── calibration_data.py     # Calibration offsets
│
├── capture/
│   └── realsense_capture.py    # RealSense + OpenPose integration
│
├── core/
│   ├── geometry.py             # orthonormaliser_repere(), extraire_angles_nautiques()
│   ├── smoothing.py            # local_polynomial_regression()
│   ├── keypoints.py            # JSON loading, confidence filtering
│   └── frame_classifier.py     # Face/profile classification
│
├── reba/
│   ├── tables.py               # TABLE_A, TABLE_B, TABLE_C constants
│   ├── angles.py               # Angle calculation per segment
│   ├── scoring.py              # REBA scoring functions
│   ├── calibration.py          # Calibration offset application
│   └── risk_assessment.py      # Main REBAAssessor class
│
├── visualization/
│   ├── annotator.py            # Video annotation with risks
│   └── viewer.py               # Interactive frame viewer
│
└── io/
    ├── json_io.py              # JSON read/write helpers
    └── video_io.py             # VideoReader/VideoWriter classes
```

## Commands

### Environment Setup
```bash
conda activate POSTURE_POSE
pip install -e .  # Install package in development mode
```

### CLI Usage

```bash
# Complete pipeline (capture → analyze → annotate)
reba3d pipeline --bag recording.bag --output ./output

# Individual steps:
reba3d capture --bag recording.bag --output ./output
reba3d analyze --input keypoints_3d.json --output risk_times.json
reba3d annotate --video output_openpose.avi --risks risk_times.json
reba3d view --video output_openpose.avi --keypoints keypoints_3d.json
```

### Alternative: Python Module
```bash
python -m reba_3d pipeline --bag recording.bag
```

### Python API

```python
from reba_3d import REBAAssessor

# Analyze keypoints
assessor = REBAAssessor()
results = assessor.analyze("keypoints_3d.json")
assessor.print_summary()

# Get risk times for video annotation
risk_times = assessor.get_risk_times_for_video()
```

## Architecture

### Data Flow
```
.bag (RealSense) → capture → keypoints_3d.json → analyze → risk_times.json → annotate → annotated video
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `reba/risk_assessment.py` | Main REBAAssessor class orchestrating analysis |
| `core/geometry.py` | Gram-Schmidt orthonormalization, nautical angles |
| `core/smoothing.py` | Polynomial regression for trajectory smoothing |
| `reba/tables.py` | REBA lookup tables A, B, C |
| `reba/scoring.py` | Individual segment scoring functions |
| `capture/realsense_capture.py` | RealSense .bag processing |

### Key Algorithm Components

- **`orthonormaliser_repere()`** (`core/geometry.py`): Gram-Schmidt orthonormalization for body segment reference frames
- **`extraire_angles_nautiques()`** (`core/geometry.py`): Calculates α, β, γ angles between body segments
- **`local_polynomial_regression()`** (`core/smoothing.py`): Smooths keypoint trajectories over 30-frame windows
- **REBA Tables** (`reba/tables.py`): Table A (torso+neck+legs) + Table B (arms+wrist) → Table C (final score 1-15)

### Risk Level Mapping
- Score 1: negligible risk (sans risque)
- Score 2-3: low risk (risque faible)
- Score 4-6: medium risk (risque moyen)
- Score 7-10: high risk (risque élevé)
- Score 11+: very high risk (très élevé)

## Configuration

Default settings in `reba_3d/config/settings.py`:
- **OPENPOSE_PATH**: `~/openpose_alt`
- **OUTPUT_DIR**: `~/openpose_alt`
- **CONFIDENCE_THRESHOLD**: 0.35
- **WINDOW_SIZE**: 30 frames (~2 seconds at 15 FPS)
- **POLY_ORDER**: 2 (polynomial order for smoothing)
- **Keypoint model**: OpenPose 25-point (BODY_25)

Settings can be overridden via:
1. CLI arguments (`--openpose`, `--output`, etc.)
2. Environment variables: `REBA_OPENPOSE_PATH`, `REBA_OUTPUT_DIR`

## Dependencies

- Python 3.8+
- numpy, pandas
- opencv-python
- pyrealsense2 (RealSense SDK)
- pyopenpose (OpenPose Python bindings - manual installation required)
- ffmpeg (video processing)

## Notes

- Comments in code are in French
- Docstrings are in English for public API
- Wrist angle scoring not implemented (VRAM constraints)
- Frame classification distinguishes face view vs left/right profile
