# Export of Calibrated Angles in Offline Mode

## Feature

During REBA analysis in offline mode, the system automatically generates a log file containing all **calibrated angles** for each analysis window.

## File Location

The `calibrated_angles.log` file is created in the **same folder as the `keypoints.json` file**.

```
example_video/
├── keypoints.json          # Input file
├── calibrated_angles.log   # <- Calibrated angles (new)
└── risk_times/
    └── risk_times.json     # REBA intervals
```

## File Format

### General Structure

```
================================================================================
CALIBRATED ANGLES - REBA ANALYSIS
================================================================================

NECK Angles (Windows):
Alpha_neck: [F1: 0.00°, F2: 0.92°, F3: 3.62°, F4: -2.83°, ...]
Beta_neck: [F1: 0.00°, F2: 0.83°, F3: 1.05°, F4: -19.54°, ...]
Gamma_neck: [F1: 0.00°, F2: 2.84°, F3: 6.45°, F4: 135.21°, ...]

TORSO Angles (Windows):
Alpha_torso: [F1: 90.12°, F2: 91.45°, F3: 89.78°, F4: 92.33°]
Beta_torso: [F1: 2.45°, F2: 3.12°, F3: 1.89°, F4: 2.67°]
Gamma_torso: [F1: 0.56°, F2: 1.23°, F3: -0.45°, F4: 0.89°]

RIGHT SHOULDER Angles (Windows):
Alpha_right_shoulder: [F1: 2.30°, F2: 3.45°, F3: 1.89°, F4: 4.12°]
Beta_right_shoulder: [F1: 9.60°, F2: 10.23°, F3: 8.97°, F4: 11.45°]
Gamma_right_shoulder: [F1: 1.20°, F2: 2.34°, F3: 0.56°, F4: 3.12°]
Elevation_right_shoulder: [F1: 94.00°, F2: 95.23°, F3: 93.45°, F4: 96.78°]

LEFT SHOULDER Angles (Windows):
Alpha_left_shoulder: [F1: 21.00°, F2: 22.45°, F3: 20.12°, F4: 23.89°]
Beta_left_shoulder: [F1: -15.30°, F2: -14.67°, F3: -16.23°, F4: -13.89°]
Gamma_left_shoulder: [F1: 174.20°, F2: 175.34°, F3: 173.56°, F4: 176.12°]
Elevation_left_shoulder: [F1: 89.50°, F2: 90.23°, F3: 88.67°, F4: 91.45°]

RIGHT ELBOW Angles (Windows):
Angle_right_elbow: [F1: 170.50°, F2: 168.30°, F3: 165.20°, F4: 172.10°]

LEFT ELBOW Angles (Windows):
Angle_left_elbow: [F1: 170.50°, F2: 169.23°, F3: 167.89°, F4: 171.34°]

RIGHT KNEE Angles (Windows):
Angle_right_knee: [F1: 178.00°, F2: 177.45°, F3: 178.23°, F4: 177.89°]

LEFT KNEE Angles (Windows):
Angle_left_knee: [F1: 178.00°, F2: 177.67°, F3: 178.34°, F4: 177.12°]

================================================================================
Total windows analyzed: 24
================================================================================
```

### Included Segments

The file contains angles for all body segments:

| Segment | Exported Angles |
|---------|----------------|
| **NECK** | alpha, beta, gamma |
| **TORSO** | alpha, beta, gamma |
| **RIGHT SHOULDER** | alpha, beta, gamma, elevation |
| **LEFT SHOULDER** | alpha, beta, gamma, elevation |
| **RIGHT ELBOW** | angle |
| **LEFT ELBOW** | angle |
| **RIGHT KNEE** | angle |
| **LEFT KNEE** | angle |

### Window Nomenclature

- **F1**: Window 1 (frames 0-29, 30 frames per window by default)
- **F2**: Window 2 (frames 30-59)
- **F3**: Window 3 (frames 60-89)
- etc.

## Usage

### Offline Mode (automatic)

During offline REBA analysis, the file is created automatically:

```bash
# Run REBA analysis
python3 src/main.py analyze path/to/keypoints.json

# The calibrated_angles.log file is created automatically
```

### Via Python (API)

```python
from reba_3d.reba.risk_assessment import assess_video

# Analyze a video
results = assess_video("path/to/keypoints.json")

# The file is automatically saved in:
# path/to/calibrated_angles.log
```

### Manual Save

```python
from reba_3d.reba.risk_assessment import REBAAssessor

# Create assessor
assessor = REBAAssessor()

# Analyze
results = assessor.analyze("path/to/keypoints.json")

# Save manually
assessor.save_calibrated_angles_log("custom_path/angles.log")
```

## Angle Interpretation

### Nautical Angles (alpha, beta, gamma)

For **NECK**, **TORSO**, **SHOULDERS** segments:

- **Alpha**: Flexion/extension (forward/backward)
  - Neutral position: ~0° (after calibration)
  - Positive: forward flexion
  - Negative: backward extension

- **Beta**: Lateral inclination (left/right)
  - Neutral position: ~0° (after calibration)
  - Positive: right inclination
  - Negative: left inclination

- **Gamma**: Rotation (twist)
  - Neutral position: ~0° (after calibration)
  - Positive: right rotation
  - Negative: left rotation

### Simple Angles (elbows, knees)

- **Angle**: Flexion angle
  - 0°: Full extension (arm/leg straight)
  - 90°: 90° flexion
  - 180°: Complete alignment (standing neutral position)

### Shoulder Elevation

- **Elevation**: Shoulder height
  - ~90°: Neutral position
  - <90°: Lowered shoulder
  - >90°: Elevated shoulder (shrugging)

## Example Data Usage

### Manual Analysis

```bash
# Open file
cat path/to/calibrated_angles.log

# Search for neck angles
grep "Alpha_neck" path/to/calibrated_angles.log
```

### Python Analysis

```python
import re

# Read file
with open("path/to/calibrated_angles.log", "r") as f:
    content = f.read()

# Extract alpha values for neck
alpha_match = re.search(r"Alpha_neck : \[(.*?)\]", content)
if alpha_match:
    values_str = alpha_match.group(1)
    # Parse values: F1: 0.00°, F2: 0.92°, ...
    angles = []
    for match in re.finditer(r"F\d+: ([-\d.]+)°", values_str):
        angles.append(float(match.group(1)))

    print(f"Neck alpha angles: {angles}")
    print(f"Max: {max(angles):.2f}°")
    print(f"Min: {min(angles):.2f}°")
    print(f"Average: {sum(angles)/len(angles):.2f}°")
```

### Visualization with Pandas

```python
import pandas as pd
import re

def parse_angles_log(log_path):
    """Parse the angles log file."""
    with open(log_path, "r") as f:
        content = f.read()

    data = {}

    # Extract all angles
    pattern = r"(\w+)_(\w+) : \[(.*?)\]"
    for match in re.finditer(pattern, content):
        angle_type = match.group(1)
        segment = match.group(2)
        values_str = match.group(3)

        # Parse values
        values = []
        for val_match in re.finditer(r"F\d+: ([-\d.]+)°", values_str):
            values.append(float(val_match.group(1)))

        column_name = f"{segment}_{angle_type}"
        data[column_name] = values

    return pd.DataFrame(data)

# Load data
df = parse_angles_log("path/to/calibrated_angles.log")

# Display statistics
print(df.describe())

# Display neck angles
print(df[['neck_alpha', 'neck_beta', 'neck_gamma']])
```

## Use Cases

### 1. Calibration Validation

Verify that angles in neutral position are close to 0°:

```bash
# Extract first window (neutral position)
grep "F1:" calibrated_angles.log
```

### 2. Extreme Angle Detection

Identify windows with dangerous angles:

```python
# Find windows where alpha_torso > 60° (strong flexion)
import re

with open("calibrated_angles.log", "r") as f:
    content = f.read()

alpha_match = re.search(r"Alpha_torso : \[(.*?)\]", content)
if alpha_match:
    values_str = alpha_match.group(1)
    for match in re.finditer(r"F(\d+): ([-\d.]+)°", values_str):
        window = match.group(1)
        angle = float(match.group(2))
        if abs(angle) > 60:
            print(f"Warning: Window {window}: excessive flexion ({angle:.2f}°)")
```

### 3. Export to Excel

```python
import pandas as pd

# Parse log
df = parse_angles_log("calibrated_angles.log")

# Export to Excel
df.to_excel("angles_analysis.xlsx", index=False)
print("Export complete: angles_analysis.xlsx")
```

## Configuration

### Modify Window Count

The number of windows depends on:
- **Video duration**: Longer = more windows
- **Window size**: Default 30 frames (2 seconds at 15 FPS)

Configuration in `src/reba_3d/config/settings.py`:

```python
WINDOW_SIZE = 30  # Number of frames per window
FPS = 15          # Frames per second
```

### Modify Filename

```python
from pathlib import Path

# Custom path
custom_path = Path("my_folder") / "custom_angles.log"
assessor.save_calibrated_angles_log(str(custom_path))
```

## Limitations

1. **0.00° values**: May indicate:
   - Perfect neutral position (rare)
   - Keypoint detection error
   - Invalid frame

2. **2D Mode**: Alpha and Beta always at 0.00° (no depth)

3. **Incomplete windows**: Last window may have fewer than 30 frames

## Associated Files

In the same folder as `keypoints.json`:

```
example_video/
├── keypoints.json              # OpenPose detections
├── calibrated_angles.log       # Calibrated angles (THIS FILE)
└── risk_times/
    └── risk_times.json         # REBA intervals
```

## Technical Notes

- **Encoding**: UTF-8 with special character support (°, etc.)
- **Precision**: 2 decimal places (e.g., 12.34°)
- **Separator**: Comma + space (`, `)
- **Format**: Plain text (.log) for universal compatibility

---

**Creation date**: 2026-01-26
**Version**: 1.0
**Status**: Implemented and tested
