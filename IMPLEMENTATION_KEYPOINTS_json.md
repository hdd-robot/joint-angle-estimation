# Implementation Notes - REBA Modifications

## Date
2026-01-27 (Update)

---

## New Features (2026-01-27)

### 3. Parallel 2D REBA Analysis alongside 3D

**Modified files**:
- `src/reba_3d/reba/risk_assessment.py`
- `src/reba_3d/reba/angles.py`
- `src/reba_3d/gui/app.py`

**Description**:
The system now generates **two analyses in parallel** when loading a `keypoints_3d.json` file:
- **REBA 3D**: Uses complete 3D nautical angles (alpha, beta, gamma)
- **REBA 2D**: Uses planar angle projection (gamma only)

**Generated files**:
```
keypoints_folder/
├── keypoints_3d.json
├── reba_analysis_3d.log       # Detailed 3D report
├── reba_analysis_2d.log       # Detailed 2D report
└── risk_times/
    ├── risk_times_3d.json     # 3D risk intervals
    └── risk_times_2d.json     # 2D risk intervals
```

**`mode` parameter in REBAAssessor**:
```python
# 3D mode (complete nautical angles)
assessor_3d = REBAAssessor(mode="3d")

# 2D mode (planar projection)
assessor_2d = REBAAssessor(mode="2d")
```

**Calculation differences**:

| Segment | 3D Mode | 2D Mode |
|---------|---------|---------|
| Neck | `compute_neck_angles()` | `compute_neck_angles_2d()` |
| Torso | `compute_torso_angles()` | `compute_torso_angles_2d()` |
| Right Shoulder | `compute_shoulder_angles_right()` | `compute_shoulder_angles_right_2d()` |
| Left Shoulder | `compute_shoulder_angles_left()` | `compute_shoulder_angles_left_2d()` |

2D functions return `(0.0, 0.0, gamma)` where gamma is the angle projected in the XY plane.

---

### 4. Custom Calibration Based on Neutral Frames

**Modified files**:
- `src/reba_3d/reba/risk_assessment.py`
- `src/reba_3d/config/yaml_config.py`
- `src/reba_3d/gui/app.py`

**Description**:
Instead of using pre-defined static calibration offsets, the system can now calculate **custom** offsets from frames where the subject is in neutral posture (standing, arms at sides).

**Configuration via `config.yaml`**:
```yaml
calibration:
  neutral_frame_start: 0    # Start frame of neutral posture
  neutral_frame_end: 30     # End frame of neutral posture
  # Set to null to use static calibration
```

**Configuration via Python code**:
```python
# With custom calibration
assessor = REBAAssessor(
    mode="3d",
    neutral_frames=(0, 30)  # Frames 0 to 30 = neutral posture
)

# Or via assess_video()
assess_video(
    "keypoints_3d.json",
    neutral_frames=(0, 30)
)
```

**Custom calibration algorithm**:

```
1. Identify windows corresponding to specified neutral frames
2. Extract raw angles from these windows
3. Calculate offset[segment][angle] = average(neutral_angles)
4. Apply calibration with these custom offsets
```

**Calibration formulas by segment**:

| Segment | Formula |
|---------|---------|
| Neck | `normalize_angle(angle - offset)` |
| Torso | `angle - offset` |
| Shoulders | `offset - angle` (inverted) |
| Elbows | `abs(angle - offset)` |
| Knees | `abs(angle - offset)` |

**Advantages**:
- Adapts to each subject's morphology
- Compensates for natural posture variations
- Improves REBA scoring accuracy

---

### 5. Detailed Analysis Report (.log)

**Added file**: `save_detailed_analysis_log()` method in `REBAAssessor`

**Description**:
Generates a `.log` file with complete REBA analysis details.

**File format**:
```
================================================================================
REBA 3D ANALYSIS - DETAILED REPORT
================================================================================

3D Angles (30-frame Windows):
----------------------------------------

3D NECK Angles (Windows):
Alpha_neck: [F1: 0.00°, F2: -2.71°, F3: 5.12°]
Beta_neck: [F1: 0.00°, F2: -3.87°, F3: 2.15°]
Gamma_neck: [F1: 0.00°, F2: 1.23°, F3: -0.85°]

3D TORSO Angles (Windows):
...

----------------------------------------
3D Table A
[2, 3, 2]

3D Table B
[1, 2, 1]

3D Table C (Final Score)
[2, 4, 2]

----------------------------------------
3D Risks according to REBA:
[ F1: low risk, F2: medium risk, F3: low risk ]

================================================================================
Total windows analyzed: 3
Calculation mode: 3D
================================================================================
```

---

## Generated Files Summary

When analyzing a `keypoints_3d.json`, the system now generates:

| File | Description |
|------|-------------|
| `risk_times_3d.json` | Risk intervals (3D mode) |
| `risk_times_2d.json` | Risk intervals (2D mode) |
| `reba_analysis_3d.log` | Detailed 3D report |
| `reba_analysis_2d.log` | Detailed 2D report |

---

## Date
2026-01-25

## Changes Made

### 0. Migration to FRAME Intervals in `risk_times.json` (BREAKING CHANGE)

**Modified files**:
- `src/reba_3d/reba/risk_assessment.py:361-398` (`_calculate_scores()` method)
- `src/reba_3d/reba/risk_assessment.py:429-451` (`get_risk_times_for_video()` method)
- `src/reba_3d/reba/risk_assessment.py:452-474` (`print_summary()` method)

**Description**:
The `risk_times.json` file now uses **frame numbers** instead of timestamps in seconds to define risk intervals. This change eliminates temporal drift issues caused by FPS variations and guarantees perfect synchronization with the source video.

**Before (seconds)**:
```json
{
  "medium risk": [[2.0, 3.93], [6.0, 7.93]],
  "high risk": [[4.0, 5.93]]
}
```

**After (frames)**:
```json
{
  "medium risk": [[30, 59], [90, 119]],
  "high risk": [[60, 89]]
}
```

**Advantages**:
- **Anti-drift FPS**: Frames are discrete units, no rounding or temporal drift issues
- **Absolute precision**: Each frame is referenced exactly as in the video
- **OpenCV compatibility**: Video readers use frame numbers (cv2.CAP_PROP_POS_FRAMES)
- **FPS independent**: Works with any FPS without recalculation

**Technical changes**:

1. **Internal storage** (`_calculate_scores()`, line 394-398):
```python
# Old code (seconds)
start_time = round(i * seconds_per_window, 2)
end_time = round((i + 1) * seconds_per_window, 2)
self.risk_times[label].append((i + 1, start_time, end_time))

# New code (frames)
window_frames = self.windows_info[i]["frames"]
start_frame = int(window_frames[0])
end_frame = int(window_frames[-1])
self.risk_times[label].append((i + 1, start_frame, end_frame))
```

2. **JSON export** (`get_risk_times_for_video()`, line 447):
```python
# Now returns [[start_frame, end_frame], ...] instead of [(start_s, end_s), ...]
result[en_label] = [[start_frame, end_frame] for _, start_frame, end_frame in intervals]
```

3. **Console display** (`print_summary()`, line 471-473):
```python
# Converts frames to seconds only for debug display
start_s = start_frame / self.fps
end_s = (end_frame + 1) / self.fps
print(f"  - F{window_num}: frames [{start_frame}..{end_frame}] => {start_s:.2f}s to {end_s:.2f}s")
```

**Impact on other modules**:
- Modules that read `risk_times.json` (video annotator, GUI, etc.) must convert frames to seconds if needed: `timestamp = frame_number / video_fps`
- Timestamps for human display are still calculated on the fly

**Compatibility**:
BREAKING CHANGE: Old `risk_times.json` files with timestamps in seconds are no longer compatible with this version.

### 1. Automatic `risk_times.json` File Organization

**Modified file**: `src/reba_3d/reba/risk_assessment.py:474-512`

**Description**:
`risk_times.json` files generated by REBA analysis are now automatically saved in a dedicated `risk_times/` subfolder.

**Behavior**:
- When `output_path=None` in `assess_video()` function, the system automatically creates a `risk_times/` folder in the same directory as the source `keypoints_3d.json` file
- Output filename is generated from input filename: `<keypoints_name>_risk_times.json`

**Example**:
```
Source file:    /path/to/data/keypoints_3d.json
Generated file: /path/to/data/risk_times/keypoints_3d_risk_times.json
```

**Added code**:
```python
# Determine output path
if output_path is None:
    from pathlib import Path
    keypoints_file = Path(keypoints_path)
    # Create risk_times folder in the same directory as keypoints file
    risk_times_dir = keypoints_file.parent / "risk_times"
    risk_times_dir.mkdir(exist_ok=True)
    # Generate output filename based on keypoints filename
    output_filename = keypoints_file.stem + "_risk_times.json"
    output_path = str(risk_times_dir / output_filename)
```

### 2. Added "low risk" Level in JSON Exports

**Modified files**:
- `src/reba_3d/reba/risk_assessment.py:362-368` (dictionary initialization)
- `src/reba_3d/reba/risk_assessment.py:435-441` (FR/EN mapping)

**Description**:
The "low risk" risk level is now included in JSON exports of risk intervals.

**Changes**:

1. Added in `self.risk_times` dictionary initialization:
```python
self.risk_times = {
    "no risk": [],
    "low risk": [],      # <- ADDED
    "medium risk": [],
    "high risk": [],
    "very high": [],
}
```

2. Added in French → English mapping:
```python
label_map = {
    "no risk": "negligible risk",
    "low risk": "low risk",      # <- ADDED
    "medium risk": "medium risk",
    "high risk": "high risk",
    "very high": "very high risk",
}
```

**Result**:
The `risk_times.json` file can now contain up to 5 risk levels:
- `"negligible risk"` - REBA Score: 1
- `"low risk"` - REBA Scores: 2-3
- `"medium risk"` - REBA Scores: 4-7
- `"high risk"` - REBA Scores: 8-10
- `"very high risk"` - REBA Scores: 11-12

## `risk_times.json` File Structure

**Format** (intervals in FRAMES):
```json
{
  "negligible risk": [[start_frame, end_frame], ...],
  "low risk": [[start_frame, end_frame], ...],
  "medium risk": [[start_frame, end_frame], ...],
  "high risk": [[start_frame, end_frame], ...],
  "very high risk": [[start_frame, end_frame], ...]
}
```

**Concrete example**:
```json
{
  "low risk": [[0, 29], [90, 119]],
  "medium risk": [[30, 59]],
  "high risk": [[60, 89]]
}
```

IMPORTANT: Intervals are expressed in **frame numbers** (integers), not seconds.

**Frame ↔ seconds conversion**:
```python
# Frame → Timestamp
timestamp_seconds = frame_number / video_fps

# Timestamp → Frame
frame_number = int(timestamp_seconds * video_fps)
```

**Example** (FPS = 15):
- Frame 30 → 2.0 seconds
- Frame 59 → 3.93 seconds
- Frame 90 → 6.0 seconds

## Compatibility

These modifications are backward compatible:
- If an explicit `output_path` is provided, it is used as-is (original behavior)
- Existing files are not affected
- JSON format remains identical, with simply an additional possible risk level

## REBA Score Reference

According to `src/reba_3d/config/settings.py:116-129`:

| REBA Score | Risk Level (FR) | Risk Level (EN) |
|------------|-----------------|-----------------|
| 1          | no risk         | negligible risk |
| 2-3        | low risk        | low risk        |
| 4-7        | medium risk     | medium risk     |
| 8-10       | high risk       | high risk       |
| 11-12      | very high       | very high risk  |

## Usage

### Via CLI (analyze command)
```bash
python -m reba_3d analyze input/keypoints_3d.json
# Automatically generates: input/risk_times/keypoints_3d_risk_times.json
```

### Via CLI with custom path
```bash
python -m reba_3d analyze input/keypoints_3d.json --output custom/path/risks.json
```

### Via Python API
```python
from reba_3d.reba.risk_assessment import assess_video

# Automatic save in risk_times/
results = assess_video("path/to/keypoints_3d.json")

# Or with custom path
results = assess_video(
    "path/to/keypoints_3d.json",
    output_path="custom/path/risks.json"
)
```

## Technical Notes

- The `risk_times/` folder is created with `mkdir(exist_ok=True)`, so no error if folder already exists
- Using `Path.stem` ensures the `.json` extension is not duplicated
- Intervals are stored in frames to ensure temporal precision regardless of FPS
