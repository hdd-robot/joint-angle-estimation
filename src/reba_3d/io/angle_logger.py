"""
Angle logging module for REBA 3D.

Logs angles to file in offline mode with support for 2D and 3D modes.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from reba_3d.config.settings import OUTPUT_DIR


class AngleLogger:
    """
    Logger for recording angles frame by frame.

    Supports both 2D and 3D modes with formatted output.
    """

    # French internal key -> English CSV column name
    _EN_COLUMN_NAMES: Dict[str, str] = {
        "Alpha_Cou": "neck_alpha",
        "Beta_Cou": "neck_beta",
        "Gamma_Cou": "neck_gamma",
        "Alpha_Buste": "trunk_alpha",
        "Beta_Buste": "trunk_beta",
        "Gamma_Buste": "trunk_gamma",
        "Alpha_Epaule_D": "right_shoulder_alpha",
        "Beta_Epaule_D": "right_shoulder_beta",
        "Gamma_Epaule_D": "right_shoulder_gamma",
        "Elevation_epaule_D": "right_shoulder_elevation",
        "Alpha_Epaule_G": "left_shoulder_alpha",
        "Beta_Epaule_G": "left_shoulder_beta",
        "Gamma_Epaule_G": "left_shoulder_gamma",
        "Elevation_epaule_G": "left_shoulder_elevation",
        "Coude_droit": "right_elbow",
        "Coude_gauche": "left_elbow",
        "Genou_droit": "right_knee",
        "Genou_gauche": "left_knee",
    }

    def __init__(self, output_dir: Optional[str] = None, mode: str = "3d", fps: float = 30.0):
        """
        Initialize the angle logger.

        Args:
            output_dir: Directory to save log files. Defaults to ~/openpose_output/angles/
            mode: "2d" or "3d" mode
            fps: Frames per second (user-defined, used to compute timestamps in CSV)
        """
        self.mode = mode.lower()
        self.fps = fps
        self.output_dir = Path(output_dir) if output_dir else Path(OUTPUT_DIR) / "angles"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for angles per segment
        self._angles_2d: Dict[str, List[float]] = {}
        self._angles_3d: Dict[str, List[float]] = {}
        self._frame_count = 0
        self._is_recording = False

    def start_recording(self) -> None:
        """Start a new recording session, clearing previous data."""
        self._angles_2d.clear()
        self._angles_3d.clear()
        self._frame_count = 0
        self._is_recording = True

    def stop_recording(self) -> None:
        """Stop recording."""
        self._is_recording = False

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def log_frame(
        self,
        angles: Dict[str, Dict[str, float]],
        mode: Optional[str] = None
    ) -> None:
        """
        Log angles for a single frame.

        Args:
            angles: Nested dictionary of angles:
                {
                    "neck": {"alpha": float, "beta": float, "gamma": float},
                    "torso": {"alpha": float, "beta": float, "gamma": float},
                    "right_shoulder": {"alpha": float, "beta": float, "gamma": float, "elevation": float},
                    "left_shoulder": {"alpha": float, "beta": float, "gamma": float, "elevation": float},
                    "right_elbow": {"angle": float},
                    "left_elbow": {"angle": float},
                    "right_knee": {"angle": float},
                    "left_knee": {"angle": float},
                }
            mode: Override mode for this frame ("2d" or "3d"). If None, uses instance mode.
        """
        if not self._is_recording:
            return

        current_mode = (mode or self.mode).lower()
        storage = self._angles_3d if current_mode == "3d" else self._angles_2d

        # Only increment frame count once per actual frame.
        # In comparison mode log_frame is called twice (2d + 3d) for the same frame.
        new_len = (max(len(v) for v in storage.values()) + 1) if storage else 1
        self._frame_count = max(self._frame_count, new_len)

        # Process each segment
        for segment, values in angles.items():
            if not isinstance(values, dict):
                continue

            if segment == "neck":
                self._log_nautical_angles(storage, "Cou", values)
            elif segment == "torso":
                self._log_nautical_angles(storage, "Buste", values)
            elif segment == "right_shoulder":
                self._log_nautical_angles(storage, "Epaule_D", values)
                if "elevation" in values:
                    self._append_angle(storage, "Elevation_epaule_D", values.get("elevation", 0.0))
            elif segment == "left_shoulder":
                self._log_nautical_angles(storage, "Epaule_G", values)
                if "elevation" in values:
                    self._append_angle(storage, "Elevation_epaule_G", values.get("elevation", 0.0))
            elif segment == "right_elbow":
                self._append_angle(storage, "Coude_droit", values.get("angle", 0.0))
            elif segment == "left_elbow":
                self._append_angle(storage, "Coude_gauche", values.get("angle", 0.0))
            elif segment == "right_knee":
                self._append_angle(storage, "Genou_droit", values.get("angle", 0.0))
            elif segment == "left_knee":
                self._append_angle(storage, "Genou_gauche", values.get("angle", 0.0))

    def _log_nautical_angles(
        self,
        storage: Dict[str, List[float]],
        prefix: str,
        values: Dict[str, float]
    ) -> None:
        """Log alpha, beta, gamma for a segment."""
        self._append_angle(storage, f"Alpha_{prefix}", values.get("alpha", 0.0))
        self._append_angle(storage, f"Beta_{prefix}", values.get("beta", 0.0))
        self._append_angle(storage, f"Gamma_{prefix}", values.get("gamma", 0.0))

    def _append_angle(
        self,
        storage: Dict[str, List[float]],
        key: str,
        value: float
    ) -> None:
        """Append an angle value to storage."""
        if value is None or (isinstance(value, float) and (value != value)):  # NaN check
            value = 0.0
        if key not in storage:
            storage[key] = []
        storage[key].append(value)

    def _format_angles_line(self, name: str, values: List[float], max_frames: int = 10) -> str:
        """
        Format a single line of angles.

        Args:
            name: Angle name
            values: List of angle values
            max_frames: Maximum number of frames to show (0 = all)

        Returns:
            Formatted string like "Alpha_torso : [F1: 0.00, F2: 12.34, ...]"
        """
        if not values:
            return f"{name} : []"

        frames_to_show = values if max_frames == 0 else values[:max_frames]
        frame_strs = [f"F{i+1}: {v:.2f}" for i, v in enumerate(frames_to_show)]

        if max_frames > 0 and len(values) > max_frames:
            frame_strs.append(f"... (+{len(values) - max_frames} frames)")

        return f"{name} : [{', '.join(frame_strs)}]"

    def get_formatted_output(self, max_frames: int = 10) -> str:
        """
        Get formatted output string for all logged angles.

        Args:
            max_frames: Maximum frames to display per line (0 = all)

        Returns:
            Formatted string with 2D and 3D sections
        """
        lines = []
        lines.append(f"=" * 60)
        lines.append(f"ANGLE LOG - {self._frame_count} frames recorded")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"=" * 60)

        # 2D Section
        if self._angles_2d:
            lines.append("")
            lines.append("2D")
            lines.append("-" * 40)

            # Order: Nautical angles first, then simple angles
            order_2d = [
                "Alpha_Cou", "Beta_Cou", "Gamma_Cou",
                "Alpha_Buste", "Beta_Buste", "Gamma_Buste",
                "Alpha_Epaule_D", "Beta_Epaule_D", "Gamma_Epaule_D",
                "Alpha_Epaule_G", "Beta_Epaule_G", "Gamma_Epaule_G",
                "Coude_droit", "Coude_gauche",
                "Genou_droit", "Genou_gauche",
            ]

            for key in order_2d:
                if key in self._angles_2d:
                    lines.append(self._format_angles_line(key, self._angles_2d[key], max_frames))

            # Any remaining keys not in order
            for key in sorted(self._angles_2d.keys()):
                if key not in order_2d:
                    lines.append(self._format_angles_line(key, self._angles_2d[key], max_frames))

        # 3D Section
        if self._angles_3d:
            lines.append("")
            lines.append("3D")
            lines.append("-" * 40)

            # Order: Nautical angles, elevations, then simple angles
            order_3d = [
                "Alpha_Cou", "Beta_Cou", "Gamma_Cou",
                "Alpha_Buste", "Beta_Buste", "Gamma_Buste",
                "Alpha_Epaule_D", "Beta_Epaule_D", "Gamma_Epaule_D",
                "Elevation_epaule_D",
                "Alpha_Epaule_G", "Beta_Epaule_G", "Gamma_Epaule_G",
                "Elevation_epaule_G",
                "Coude_droit", "Coude_gauche",
                "Genou_droit", "Genou_gauche",
            ]

            for key in order_3d:
                if key in self._angles_3d:
                    lines.append(self._format_angles_line(key, self._angles_3d[key], max_frames))

            # Any remaining keys not in order
            for key in sorted(self._angles_3d.keys()):
                if key not in order_3d:
                    lines.append(self._format_angles_line(key, self._angles_3d[key], max_frames))

        lines.append("")
        lines.append(f"=" * 60)

        return "\n".join(lines)

    def save_to_file(
        self,
        filename: Optional[str] = None,
        max_frames: int = 0
    ) -> Path:
        """
        Save logged angles to a text file.

        Args:
            filename: Custom filename. If None, generates timestamped name.
            max_frames: Maximum frames per line (0 = all frames)

        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"angles_{self.mode}_{timestamp}.txt"

        filepath = self.output_dir / filename

        content = self.get_formatted_output(max_frames=max_frames)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return filepath

    def save_to_csv(self, filename: Optional[str] = None) -> Path:
        """
        Save logged angles to a CSV file for analysis.

        Columns: frame, timestamp_s, fps, <angle_1>, <angle_2>, ...
        Angle columns use English names (e.g. neck_alpha, right_elbow).
        Timestamps are computed from frame number and self.fps.

        Args:
            filename: Custom filename. If None, generates timestamped name.

        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"angles_{self.mode}_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Combine 2D and 3D data with English column names
        all_data = {}
        for key, values in self._angles_2d.items():
            en_name = self._EN_COLUMN_NAMES.get(key, key)
            col = f"2d_{en_name}" if self._angles_3d else en_name
            all_data[col] = values
        for key, values in self._angles_3d.items():
            en_name = self._EN_COLUMN_NAMES.get(key, key)
            col = f"3d_{en_name}" if self._angles_2d else en_name
            all_data[col] = values

        if not all_data:
            return filepath

        # Get max frame count
        max_len = max(len(v) for v in all_data.values()) if all_data else 0

        # Write CSV
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            angle_cols = list(all_data.keys())
            headers = ["frame", "timestamp_s", "fps"] + angle_cols
            f.write(",".join(headers) + "\n")

            # Data rows
            for i in range(max_len):
                frame_num = i + 1
                timestamp_s = f"{frame_num / self.fps:.6f}"
                row = [str(frame_num), timestamp_s, f"{self.fps:g}"]
                for col in angle_cols:
                    values = all_data[col]
                    if i < len(values):
                        row.append(f"{values[i]:.4f}")
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")

        return filepath

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of logged angles.

        Returns:
            Dictionary with statistics for each angle
        """
        import statistics

        summary = {
            "frame_count": self._frame_count,
            "mode": self.mode,
            "2d": {},
            "3d": {},
        }

        for storage, key in [(self._angles_2d, "2d"), (self._angles_3d, "3d")]:
            for name, values in storage.items():
                if values:
                    summary[key][name] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": statistics.mean(values),
                        "count": len(values),
                    }

        return summary


# Global instance for convenience
_default_logger: Optional[AngleLogger] = None


def get_angle_logger(output_dir: Optional[str] = None, mode: str = "3d") -> AngleLogger:
    """
    Get or create the default angle logger.

    Args:
        output_dir: Output directory (only used on first call)
        mode: "2d" or "3d" mode

    Returns:
        AngleLogger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = AngleLogger(output_dir, mode)
    return _default_logger


def reset_angle_logger() -> None:
    """Reset the default angle logger."""
    global _default_logger
    _default_logger = None
