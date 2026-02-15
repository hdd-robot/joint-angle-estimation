# Robust 2D and 3D Separate Calibration

## Problem Solved

**Before**: A single calibration for 2D and 3D
- If calibrated in 2D then switching to 3D → incorrect offsets
- Alpha and Beta at 0° in 2D → unusable calibration in 3D

**Now**: Separate calibrations with automatic detection
- **2D File**: `calibration_data_2d.yaml`
- **3D File**: `calibration_data_3d.yaml`
- **Automatic loading** based on active mode

## Implemented Features

### 1. Automatic Mode Detection

**At calibration start** ([app.py:365-371](src/reba_3d/gui/app.py#L365-L371)):
```python
# Automatically detects if 2D or 3D calibration
self.calibration_mode_3d = self.use_3d and self.depth_intrinsics is not None
mode_str = "3D" if self.calibration_mode_3d else "2D"

# Adapted user message
self.log_panel.add_info(f"Calibration {mode_str} ({self.calibration_duration}s)...")
```

**Detection criteria**:
- **3D Mode**: `use_3d = True` AND depth available (RealSense camera)
- **2D Mode**: Otherwise (no depth or use_3d = False)

### 2. Saving to Separate Files

**Modified function**: `get_calibration_path()` ([calibration_store.py:53-79](src/reba_3d/config/calibration_store.py#L53-L79))

```python
def get_calibration_path(config_dir=None, mode_3d=None) -> Path:
    """
    Args:
        mode_3d: True  → calibration_data_3d.yaml
                 False → calibration_data_2d.yaml
                 None  → calibration_data.yaml (legacy)
    """
    if mode_3d is True:
        filename = "calibration_data_3d.yaml"
    elif mode_3d is False:
        filename = "calibration_data_2d.yaml"
    else:
        filename = "calibration_data.yaml"  # Fallback

    return project_root / filename
```

**Saving with metadata** ([app.py:435-448](src/reba_3d/gui/app.py#L435-L448)):
```python
metadata = {
    'mode_3d': self.calibration_mode_3d,  # NEW: True/False
    'method': 'robust_mad' if use_robust else 'legacy_averaging',
    'k_mad': k_mad,
    # ... other metadata
}

calibration_path = get_calibration_path(mode_3d=self.calibration_mode_3d)
save_calibration(offsets, path=calibration_path, metadata=metadata)
```

### 3. Automatic Loading Based on Mode

**Dynamic reloading** ([app.py:1030-1046](src/reba_3d/gui/app.py#L1030-L1046)):
```python
# Detects mode on each frame
mode_3d = self.use_3d and depth_frame is not None and self.depth_intrinsics is not None

# If mode changes, reload appropriate calibration
if mode_3d != self.current_mode_3d:
    self.current_mode_3d = mode_3d
    calibration_path = get_calibration_path(mode_3d=mode_3d)

    if calibration_path.exists():
        self.calibration_manager.reload(path=calibration_path, mode_3d=mode_3d)
        logger.info(f"Calibration {mode_str} loaded: {calibration_path.name}")
    else:
        logger.warning(f"Calibration file {mode_str} not found")
```

### 4. 2D/3D Comparison Mode

**Dedicated manager for 2D** ([app.py:117-126](src/reba_3d/gui/app.py#L117-L126)):
```python
# Separate manager for 2D calibration (comparison mode)
self.calibration_manager_2d = CalibrationManager.__new__(CalibrationManager)
calib_2d_path = get_calibration_path(mode_3d=False)

if calib_2d_path.exists():
    self.calibration_manager_2d._offsets = load_calibration(calib_2d_path)
else:
    self.calibration_manager_2d._offsets = DEFAULT_NAUTICAL_OFFSETS.copy()
```

**Usage in comparison** ([app.py:1068-1072](src/reba_3d/gui/app.py#L1068-L1072)):
```python
# Comparison mode: 2D calculation in parallel with 3D
if self.show_comparison and depth_frame is not None:
    raw_angles_2d = calculate_nautical_angles_2d(person_keypoints)
    # Uses dedicated 2D manager
    calibrated_2d = self.calibration_manager_2d.apply_nested(raw_angles_2d)
```

## Calibration File Structure

### 2D File: `calibration_data_2d.yaml`

```yaml
created_at: '2026-01-26T14:30:00'
metadata:
  frames_collected: 150
  duration_seconds: 5
  n_neutral: 90
  mode_3d: false          # <- 2D Mode
  method: 'robust_mad'
  k_mad: 3.5

offsets:
  neck:
    alpha: 0.0            # <- Always 0 in 2D
    beta: 0.0             # <- Always 0 in 2D
    gamma: 2.5            # <- Calculated (planar rotation)
  torso:
    alpha: 0.0
    beta: 0.0
    gamma: 3.2
  right_shoulder:
    alpha: 0.0
    beta: 0.0
    gamma: 15.4
    elevation: 92.0       # <- Calculated
  right_elbow:
    angle: 170.5          # <- Calculated
  # ... other segments
```

### 3D File: `calibration_data_3d.yaml`

```yaml
created_at: '2026-01-26T14:35:00'
metadata:
  frames_collected: 150
  duration_seconds: 5
  n_neutral: 90
  mode_3d: true           # <- 3D Mode
  method: 'robust_mad'
  k_mad: 3.5

offsets:
  neck:
    alpha: 180.0          # <- Calculated with depth
    beta: 3.5             # <- Calculated with depth
    gamma: 0.5            # <- Calculated
  torso:
    alpha: 90.0
    beta: 2.8
    gamma: 2.9
  right_shoulder:
    alpha: 2.3
    beta: 9.6
    gamma: 1.2
    elevation: 94.0
  right_elbow:
    angle: 170.5
  # ... other segments
```

## User Workflow

### 2D Calibration (without depth camera)

1. **Start application without RealSense**
   ```bash
   python3 src/main.py
   ```

2. **Click "Start Calibration"**
   - Message: "Calibration 2D (5s)..."
   - Neutral position: standing, arms at sides
   - Stay still for 5 seconds

3. **Result**
   - Message: "Calibration 2D OK (robust MAD)"
   - File: `calibration_data_2d.yaml`
   - Offsets: alpha=0, beta=0, gamma calculated

### 3D Calibration (with RealSense camera)

1. **Start with RealSense connected**
   - Depth automatically detected
   - `use_3d = True` in code

2. **Click "Start Calibration"**
   - Message: "Calibration 3D (5s)..."
   - Neutral position: same instructions
   - Stay still for 5 seconds

3. **Result**
   - Message: "Calibration 3D OK (robust MAD)"
   - File: `calibration_data_3d.yaml`
   - Offsets: alpha, beta, gamma all calculated with depth

### Automatic Mode Switching

**Scenario**: Calibrated in 2D, then depth activation

```
1. Calibration in 2D → calibration_data_2d.yaml created
2. Connect RealSense camera
3. Depth detected
4. Log: "Mode changed: switching to 3D, reloading calibration..."
5. If calibration_data_3d.yaml exists → loaded
6. Otherwise → Warning + default offsets used
```

**Solution**: Perform a separate 3D calibration after connecting

## User Messages

### Calibration Messages

| Situation | GUI Message |
|-----------|-------------|
| **Start 2D calibration** | "Calibration 2D (5s)..." |
| **Start 3D calibration** | "Calibration 3D (5s)..." |
| **2D Success** | "Calibration 2D OK (robust MAD)" |
| **3D Success** | "Calibration 3D OK (robust MAD)" |
| **File saved** | "File: calibration_data_2d.yaml" |

### Loading Messages

| Situation | Log Message |
|-----------|-------------|
| **Change 2D→3D** | "Mode changed: switching to 3D, reloading calibration..." |
| **3D Load OK** | "Calibration 3D loaded: calibration_data_3d.yaml" |
| **File missing** | "Calibration 3D file not found: calibration_data_3d.yaml" |
| **Fallback defaults** | "Using default values" |

## 2D vs 3D Comparison

| Aspect | 2D Mode | 3D Mode |
|--------|---------|---------|
| **Camera** | Standard webcam | RealSense with depth |
| **Alpha (neck, torso, shoulders)** | 0.0 (not calculated) | Calculated with depth |
| **Beta (neck, torso, shoulders)** | 0.0 (not calculated) | Calculated with depth |
| **Gamma (planar rotation)** | Calculated | Calculated |
| **Shoulder elevation** | Calculated | Calculated |
| **Elbow/knee angles** | Calculated | Calculated |
| **Calibration file** | `calibration_data_2d.yaml` | `calibration_data_3d.yaml` |
| **REBA accuracy** | Average (2D projection) | Excellent (real 3D) |

## Configuration

### Disable 3D Mode

To force 2D mode even with RealSense:

```python
# In app.py, line ~129
self.use_3d = False  # Force 2D mode
```

### Default Values

If no calibration file exists ([calibration_store.py:28-37](src/reba_3d/config/calibration_store.py#L28-L37)):

```python
DEFAULT_NAUTICAL_OFFSETS = {
    "neck": {"alpha": 180.0, "beta": 3.5, "gamma": 0.0},
    "torso": {"alpha": 90.0, "beta": 2.8, "gamma": 2.8},
    "right_shoulder": {"alpha": 0.0, "beta": 9.6, "gamma": 0.0, "elevation": 94.0},
    # ... other segments with typical values
}
```

## Tests

### Manual Testing

**Test 1: 2D Calibration**
```bash
# Without RealSense
python3 src/main.py
# 1. Click "Start Calibration"
# 2. Verify message "Calibration 2D (5s)..."
# 3. Verify creation of calibration_data_2d.yaml
# 4. Verify metadata.mode_3d = false
```

**Test 2: 3D Calibration**
```bash
# With RealSense
python3 src/main.py
# 1. Verify depth detected
# 2. Click "Start Calibration"
# 3. Verify message "Calibration 3D (5s)..."
# 4. Verify creation of calibration_data_3d.yaml
# 5. Verify metadata.mode_3d = true
# 6. Verify alpha/beta != 0.0 in offsets
```

**Test 3: Dynamic Mode Change**
```bash
# 1. Calibrate in 2D (without depth)
# 2. Connect RealSense during execution
# 3. Verify log "Mode changed: switching to 3D..."
# 4. Verify loading of calibration_data_3d.yaml
# 5. If absent, verify warning + defaults used
```

**Test 4: Comparison Mode**
```bash
# In 3D mode
# 1. Enable 2D/3D comparison
# 2. Verify that 2D score uses calibration_data_2d.yaml
# 3. Verify that 3D score uses calibration_data_3d.yaml
# 4. Compare scores (2D should be less accurate)
```

## Modified Files

### Main Modifications

1. **src/reba_3d/config/calibration_store.py**
   - `get_calibration_path()`: Added `mode_3d` parameter
   - `CalibrationManager.reload()`: Added `mode_3d` parameter
   - `CalibrationManager._mode_3d`: New field to track mode

2. **src/reba_3d/gui/app.py**
   - Line 114: `self.calibration_mode_3d` to store calibration mode
   - Line 115: `self.current_mode_3d` to detect changes
   - Line 117-126: `self.calibration_manager_2d` for comparison mode
   - Line 365-371: Mode detection at calibration start
   - Line 435-448: Save with mode in metadata and appropriate file
   - Line 1030-1046: Automatic reload if mode changes
   - Line 1068-1072: Use 2D manager for comparison

## Implementation Advantages

### 1. Improved Accuracy
- **2D in 2D**: Offsets adapted to planar projection
- **3D in 3D**: Offsets including real depth
- **No mixing**: No risk of using inappropriate calibration

### 2. Flexibility
- **Automatic switching**: Changes file based on mode
- **Comparison mode**: Can compare 2D vs 3D with appropriate calibrations
- **No confusion**: Clearly named separate files

### 3. Compatibility
- **Legacy supported**: Old `calibration_data.yaml` file still works
- **Smooth migration**: No breaking change
- **Clear messages**: User informed of mode being used

### 4. Robustness
- **Automatic detection**: No manual configuration
- **Graceful fallback**: Default values if file missing
- **Informative logs**: Complete tracking of changes

## Troubleshooting

### Problem: "Calibration 3D file not found"

**Cause**: Not yet calibrated in 3D mode

**Solution**:
```
1. Connect RealSense camera
2. Wait for depth detection
3. Click "Start Calibration"
4. calibration_data_3d.yaml will be created
```

### Problem: Different scores between 2D and 3D

**Normal**: Both modes use different calibrations

**Verify**:
- 2D mode uses `calibration_data_2d.yaml`
- 3D mode uses `calibration_data_3d.yaml`
- Both files exist and are recent

### Problem: 2D calibration used in 3D mode

**Cause**: `calibration_data_3d.yaml` file missing

**Solution**:
```
1. Verify 3D calibration exists
2. Otherwise, perform new calibration in 3D mode
3. Verify logs for "Calibration 3D loaded"
```

### Problem: Alpha/Beta at 0 even in 3D

**Possible causes**:
1. 2D calibration loaded by mistake
2. Depth not available during calibration
3. Corrupted calibration file

**Diagnosis**:
```bash
# Verify calibration file
cat calibration_data_3d.yaml

# Verify metadata
# mode_3d: true  → OK
# mode_3d: false → Wrong file!

# Verify offsets
# alpha != 0 → OK
# alpha = 0  → Recalibrate in 3D mode
```

## References

- **Robust MAD Calibration**: [IMPLEMENTATION_CALIBRATION_ROBUSTE.md](IMPLEMENTATION_CALIBRATION_ROBUSTE.md)
- **Nautical Angles Documentation**: [USAGE_2D_3D_ANGLES.md](USAGE_2D_3D_ANGLES.md)
- **Configuration**: [src/config.yaml](src/config.yaml)

---

**Implementation date**: 2026-01-26
**Version**: 2.0
**Status**: Complete and validated implementation
