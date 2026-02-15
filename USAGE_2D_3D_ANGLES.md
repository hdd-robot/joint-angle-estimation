# Usage Guide: 2D vs 3D Angle Calculations

This document explains how to use the 2D and 3D angle calculation functions in the project.

## Overview

The project now offers **two methods** for calculating body segment angles:

### 1. 3D Mode (default) - Complete Nautical Angles
- Calculates all 3 Euler angles (alpha, beta, gamma)
- Uses orthonormal frames and the complete rotation matrix
- **More accurate** for REBA analysis
- Requires 3D keypoint coordinates

### 2. 2D Mode - Planar Projection
- Calculates only gamma angle (rotation in XY plane)
- Alpha and beta are fixed at 0
- **Simpler** but less accurate
- Works with 2D coordinates (x, y)

## Available Functions

### For NECK

```python
from reba_3d.reba.angles import compute_neck_angles, compute_neck_angles_2d

# 3D Mode (complete nautical angles)
angles_3d = compute_neck_angles(positions)
# Returns: (alpha, beta, gamma) in degrees

# 2D Mode (planar projection)
angles_2d = compute_neck_angles_2d(positions)
# Returns: (0.0, 0.0, gamma) in degrees
```

### For TORSO

```python
from reba_3d.reba.angles import compute_torso_angles, compute_torso_angles_2d

# 3D Mode
angles_3d = compute_torso_angles(positions)
# Returns: (alpha, beta, gamma) in degrees

# 2D Mode
angles_2d = compute_torso_angles_2d(positions)
# Returns: (0.0, 0.0, gamma) in degrees
```

### For SHOULDERS

```python
from reba_3d.reba.angles import (
    compute_shoulder_angles_right, compute_shoulder_angles_right_2d,
    compute_shoulder_angles_left, compute_shoulder_angles_left_2d
)

# Right shoulder - 3D Mode
angles_3d = compute_shoulder_angles_right(positions)
# Returns: (alpha, beta, gamma) in degrees

# Right shoulder - 2D Mode
angles_2d = compute_shoulder_angles_right_2d(positions)
# Returns: (0.0, 0.0, gamma) in degrees

# Left shoulder - 3D Mode
angles_3d = compute_shoulder_angles_left(positions)

# Left shoulder - 2D Mode
angles_2d = compute_shoulder_angles_left_2d(positions)
```

## Complete Usage Example

```python
import numpy as np
from reba_3d.reba.angles import (
    compute_neck_angles,
    compute_neck_angles_2d,
    compute_torso_angles,
    compute_torso_angles_2d,
)

# Keypoint positions (example)
positions = {
    "Nose": np.array([0.0, 0.0, 1.7]),
    "Neck": np.array([0.0, 0.0, 1.5]),
    "REye": np.array([0.05, 0.0, 1.72]),
    "LEye": np.array([-0.05, 0.0, 1.72]),
    "MidHip": np.array([0.0, 0.0, 1.0]),
    "RShoulder": np.array([0.2, 0.0, 1.45]),
    "LShoulder": np.array([-0.2, 0.0, 1.45]),
}

# 3D mode calculation
print("=== 3D Mode ===")
neck_3d = compute_neck_angles(positions)
if neck_3d:
    alpha, beta, gamma = neck_3d
    print(f"Neck: alpha={alpha:.2f}°, beta={beta:.2f}°, gamma={gamma:.2f}°")

torso_3d = compute_torso_angles(positions)
if torso_3d:
    alpha, beta, gamma = torso_3d
    print(f"Torso: alpha={alpha:.2f}°, beta={beta:.2f}°, gamma={gamma:.2f}°")

# 2D mode calculation
print("\n=== 2D Mode ===")
neck_2d = compute_neck_angles_2d(positions)
if neck_2d:
    alpha, beta, gamma = neck_2d
    print(f"Neck: alpha={alpha:.2f}°, beta={beta:.2f}°, gamma={gamma:.2f}°")
    print("  Note: alpha and beta are always 0 in 2D mode")

torso_2d = compute_torso_angles_2d(positions)
if torso_2d:
    alpha, beta, gamma = torso_2d
    print(f"Torso: alpha={alpha:.2f}°, beta={beta:.2f}°, gamma={gamma:.2f}°")
```

## Using in a REBA Pipeline

```python
from reba_3d.reba.risk_assessment import REBAAssessor

# By default, REBAAssessor uses 3D mode
assessor = REBAAssessor()

# Analyze a 3D keypoints file
results = assessor.analyze("path/to/keypoints_3d.json")
```

## Choosing Between 2D and 3D

### Use **3D mode** when:
- You have depth data (RealSense camera, Kinect, etc.)
- You want accurate REBA analysis
- You're analyzing complex 3D movements

### Use **2D mode** when:
- You're working with classic 2D images/videos
- You don't have depth data
- You want quick, simplified analysis
- You're comparing with existing 2D results

## Technical Differences

| Aspect | 3D Mode | 2D Mode |
|--------|---------|---------|
| **Alpha** | Rotation around X | Always 0° |
| **Beta** | Rotation around Y | Always 0° |
| **Gamma** | Rotation around Z | Rotation in XY plane |
| **Function used** | `extract_nautical_angles()` | `extract_nautical_angles_2D()` |
| **Rotation matrix** | Complete 3x3 | 2D projection only |
| **REBA accuracy** | High | Reduced |

## Applied Corrections

Recent modifications include:

1. **Sign consistency** in `orthonormalize_frame()`: avoids 180° flips
2. **Gamma calculation correction**: `R[1,0]` instead of `R[0,1]`
3. **Separate functions**: `*_2d()` to avoid parameter conflicts
4. **New function**: `extract_nautical_angles_2D()` for planar projections

## Important Notes

- Functions return `None` if required keypoints are missing or invalid
- Angles are always returned in **degrees** (not radians)
- 2D mode is a **simplification** of 3D mode, not a completely different method
- For complete and compliant REBA analysis, use **3D mode**
