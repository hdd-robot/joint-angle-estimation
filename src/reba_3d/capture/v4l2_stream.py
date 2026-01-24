"""
V4L2 loopback streaming for real-time OpenPose integration.

This module provides classes to:
1. Stream RGB frames to a V4L2 loopback device (virtual webcam)
2. Read OpenPose JSON outputs and associate them with depth frames
3. Maintain synchronization between RGB and Depth streams

Architecture:
    RealSense Camera
      ├── RGB  ──> V4L2 loopback (/dev/video10) ──> OpenPose Docker (--camera)
      └── Depth ──> Depth buffer (circular) ──> 3D projection

Usage:
    1. Create V4L2 loopback device:
       sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="RGB_OpenPose"

    2. Start OpenPose Docker:
       docker run -it --gpus all --ipc=host --device=/dev/video10 \\
         -v /tmp/openpose/out:/data/out cwaffles/openpose \\
         ./build/examples/openpose/openpose.bin --camera 10 \\
         --write_json /data/out --model_pose COCO ...

    3. Run your application with V4L2 mode
"""

import os
import cv2
import json
import time
import fcntl
import threading
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import pyfakewebcam

from ..config.settings import (
    V4L2_DEVICE,
    V4L2_OPENPOSE_JSON_DIR,
    V4L2_DEPTH_BUFFER_SIZE,
)
from ..utils.logger import get_logger

# Logger pour ce module
logger = get_logger("capture.v4l2")


@dataclass
class DepthFrameData:
    """Container for depth frame with metadata."""
    frame_number: int
    depth_frame: Any  # RealSense depth frame
    timestamp: float


class V4L2Writer:
    """
    Writes frames to a V4L2 loopback device (virtual webcam).

    Uses pyfakewebcam for reliable V4L2 loopback writing.

    The V4L2 loopback device must be created beforehand:
        sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="RGB_OpenPose" exclusive_caps=1

    Example:
        >>> writer = V4L2Writer("/dev/video10", width=640, height=480)
        >>> writer.write(frame)
        >>> writer.close()
    """

    def __init__(
        self,
        device: str = None,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ):
        """
        Initialize V4L2 writer.

        Args:
            device: V4L2 loopback device path (default: /dev/video10)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second (not used by pyfakewebcam but kept for API)
        """
        self.device = device or V4L2_DEVICE
        self.width = width
        self.height = height
        self.fps = fps
        self._cam = None
        self._frame_count = 0

        self._check_device_exists()
        self._init_writer()

    def _check_device_exists(self):
        """Check if the V4L2 loopback device exists."""
        if not os.path.exists(self.device):
            raise FileNotFoundError(
                f"V4L2 loopback device not found: {self.device}\n"
                f"Create it with:\n"
                f"  sudo modprobe v4l2loopback devices=1 video_nr=10 "
                f"card_label=\"RGB_OpenPose\" exclusive_caps=1"
            )

    def _init_writer(self):
        """Initialize the pyfakewebcam FakeWebcam."""
        try:
            self._cam = pyfakewebcam.FakeWebcam(self.device, self.width, self.height)
            logger.info(f"V4L2 Writer initialized: {self.device}")
            logger.info(f"Resolution: {self.width}x{self.height}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to open V4L2 device: {self.device}\n"
                f"Error: {e}\n"
                f"Make sure v4l2loopback is loaded with exclusive_caps=1"
            )

    def write(self, frame: np.ndarray) -> int:
        """
        Write a frame to the V4L2 loopback device.

        Args:
            frame: BGR image (numpy array, OpenCV format)

        Returns:
            Frame number that was written
        """
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        # Convert BGR (OpenCV) to RGB (pyfakewebcam)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._cam.schedule_frame(frame_rgb)
        frame_num = self._frame_count
        self._frame_count += 1
        return frame_num

    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count

    def close(self):
        """Release the V4L2 writer."""
        if self._cam is not None:
            # pyfakewebcam doesn't have an explicit close method
            self._cam = None
            logger.info(f"V4L2 Writer closed: {self._frame_count} frames written")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DepthBuffer:
    """
    Circular buffer for depth frames with frame number indexing.

    Maintains a sliding window of recent depth frames for synchronization
    with OpenPose keypoint outputs.
    """

    def __init__(self, max_size: int = None):
        """
        Initialize depth buffer.

        Args:
            max_size: Maximum number of frames to keep (default from settings)
        """
        self.max_size = max_size or V4L2_DEPTH_BUFFER_SIZE
        self._buffer: deque = deque(maxlen=self.max_size)
        self._lock = threading.Lock()

    def add(self, frame_number: int, depth_frame: Any) -> None:
        """
        Add a depth frame to the buffer.

        Args:
            frame_number: Frame index
            depth_frame: RealSense depth frame object
        """
        with self._lock:
            self._buffer.append(DepthFrameData(
                frame_number=frame_number,
                depth_frame=depth_frame,
                timestamp=time.time(),
            ))

    def get(self, frame_number: int) -> Optional[Any]:
        """
        Get depth frame by frame number.

        Args:
            frame_number: Frame index to retrieve

        Returns:
            Depth frame or None if not found
        """
        with self._lock:
            for data in self._buffer:
                if data.frame_number == frame_number:
                    return data.depth_frame
        return None

    def get_closest(self, frame_number: int) -> Optional[Tuple[int, Any]]:
        """
        Get the closest depth frame to the given frame number.

        Args:
            frame_number: Target frame index

        Returns:
            Tuple (actual_frame_number, depth_frame) or None
        """
        with self._lock:
            if not self._buffer:
                return None

            closest = min(
                self._buffer,
                key=lambda d: abs(d.frame_number - frame_number)
            )
            return closest.frame_number, closest.depth_frame

    def __len__(self) -> int:
        return len(self._buffer)


class OpenPoseJSONReader:
    """
    Reads and parses OpenPose JSON output files.

    Monitors a directory for new JSON files generated by OpenPose
    and extracts keypoint data.
    """

    def __init__(self, json_dir: str = None):
        """
        Initialize JSON reader.

        Args:
            json_dir: Directory where OpenPose writes JSON files
        """
        self.json_dir = Path(json_dir or V4L2_OPENPOSE_JSON_DIR)
        self.json_dir.mkdir(parents=True, exist_ok=True)

        self._processed_files: set = set()
        self._last_frame_number = -1

        # Clean directory at start
        self._clean_directory()

        logger.info(f"OpenPose JSON Reader initialized: {self.json_dir}")

    def _clean_directory(self):
        """Remove old JSON files from the directory."""
        for f in self.json_dir.glob("*_keypoints.json"):
            f.unlink(missing_ok=True)

    def _extract_frame_number(self, filename: str) -> int:
        """
        Extract frame number from OpenPose JSON filename.

        OpenPose names files like: 000000000042_keypoints.json

        Args:
            filename: JSON filename

        Returns:
            Frame number or -1 if cannot parse
        """
        try:
            # Remove _keypoints.json suffix
            base = filename.replace("_keypoints.json", "")
            return int(base)
        except ValueError:
            return -1

    def read_new_keypoints(self) -> List[Tuple[int, Optional[np.ndarray]]]:
        """
        Read all new keypoint JSON files since last call.

        Returns:
            List of (frame_number, keypoints) tuples
        """
        results = []

        json_files = sorted(self.json_dir.glob("*_keypoints.json"))

        for json_path in json_files:
            if json_path.name in self._processed_files:
                continue

            frame_number = self._extract_frame_number(json_path.name)
            if frame_number < 0:
                continue

            keypoints = self._parse_json(json_path)
            results.append((frame_number, keypoints))

            self._processed_files.add(json_path.name)
            self._last_frame_number = max(self._last_frame_number, frame_number)

            # Clean up processed file
            json_path.unlink(missing_ok=True)

        return results

    def _parse_json(self, json_path: Path) -> Optional[np.ndarray]:
        """
        Parse OpenPose JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Keypoints array (N_persons, N_keypoints, 3) or None
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            if not data.get("people"):
                return None

            keypoints_list = []
            for person in data["people"]:
                pose_keypoints = person.get("pose_keypoints_2d", [])
                if pose_keypoints:
                    n_kp = len(pose_keypoints) // 3
                    kp_array = np.array(pose_keypoints).reshape(n_kp, 3)
                    keypoints_list.append(kp_array)

            if keypoints_list:
                return np.array(keypoints_list)

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error parsing {json_path}: {e}")

        return None

    @property
    def last_frame_number(self) -> int:
        """Last processed frame number."""
        return self._last_frame_number


class V4L2OpenPoseStreamer:
    """
    Real-time OpenPose streaming via V4L2 loopback.

    This class coordinates:
    1. Writing RGB frames to V4L2 loopback device
    2. Buffering depth frames for 3D projection
    3. Reading OpenPose JSON outputs
    4. Associating 2D keypoints with depth data for 3D reconstruction

    Example:
        >>> streamer = V4L2OpenPoseStreamer()
        >>> with RealSenseCapture(bag_file) as capture:
        ...     for frame_num, color_frame, depth_frame in capture:
        ...         frame = np.asanyarray(color_frame.get_data())
        ...         streamer.push_frame(frame, depth_frame, frame_num)
        ...
        ...         # Get 3D keypoints for processed frames
        ...         results = streamer.get_3d_keypoints(capture)
    """

    def __init__(
        self,
        v4l2_device: str = None,
        json_dir: str = None,
        buffer_size: int = None,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ):
        """
        Initialize V4L2 OpenPose streamer.

        Args:
            v4l2_device: V4L2 loopback device path
            json_dir: Directory for OpenPose JSON output
            buffer_size: Depth frame buffer size
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.width = width
        self.height = height
        self.fps = fps

        self._v4l2_writer = V4L2Writer(
            device=v4l2_device,
            width=width,
            height=height,
            fps=fps,
        )

        self._depth_buffer = DepthBuffer(max_size=buffer_size)
        self._json_reader = OpenPoseJSONReader(json_dir=json_dir)

        self._frame_count = 0

        logger.info("V4L2 OpenPose Streamer ready")
        logger.info("Waiting for OpenPose Docker to process frames...")

    def push_frame(
        self,
        rgb_frame: np.ndarray,
        depth_frame: Any,
        frame_number: int = None,
    ) -> int:
        """
        Push a frame pair (RGB + Depth) to the streamer.

        RGB is sent to V4L2, Depth is buffered for later 3D projection.

        Args:
            rgb_frame: BGR image (numpy array)
            depth_frame: RealSense depth frame
            frame_number: Optional frame number (auto-incremented if None)

        Returns:
            Frame number
        """
        if frame_number is None:
            frame_number = self._frame_count

        # Write RGB to V4L2
        self._v4l2_writer.write(rgb_frame)

        # Buffer depth for later
        self._depth_buffer.add(frame_number, depth_frame)

        self._frame_count += 1
        return frame_number

    def get_3d_keypoints(
        self,
        capture: Any,  # RealSenseCapture for get_3d_point
    ) -> List[Dict[str, Any]]:
        """
        Get 3D keypoints from processed frames.

        Reads new OpenPose JSON outputs and projects 2D keypoints to 3D
        using buffered depth frames.

        Args:
            capture: RealSenseCapture instance for depth-to-3D projection

        Returns:
            List of dictionaries with frame data:
            [{"frame": int, "keypoints_3d": [...]}]
        """
        results = []

        # Read new keypoints from OpenPose
        new_keypoints = self._json_reader.read_new_keypoints()

        for frame_number, keypoints_2d in new_keypoints:
            if keypoints_2d is None:
                continue

            # Find matching depth frame
            depth_result = self._depth_buffer.get_closest(frame_number)
            if depth_result is None:
                logger.warning(f"No depth frame for keypoints frame {frame_number}")
                continue

            actual_frame, depth_frame = depth_result

            # Project 2D keypoints to 3D
            person_kps = []
            for person in keypoints_2d:
                joints = []
                for kp in person:
                    x, y, conf = kp
                    x, y = int(x), int(y)
                    p3d = capture.get_3d_point(depth_frame, x, y)
                    joints.append({
                        "x": float(p3d[0]),
                        "y": float(p3d[1]),
                        "z": float(p3d[2]),
                        "confidence": float(conf),
                    })
                person_kps.append(joints)

            results.append({
                "frame": frame_number,
                "keypoints_3d": person_kps,
            })

        return results

    @property
    def frames_sent(self) -> int:
        """Number of frames sent to V4L2."""
        return self._frame_count

    @property
    def frames_processed(self) -> int:
        """Number of frames processed by OpenPose."""
        return self._json_reader.last_frame_number + 1

    def close(self):
        """Close the streamer."""
        self._v4l2_writer.close()
        logger.info("V4L2 Streamer closed")
        logger.info(f"Frames sent: {self._frame_count}")
        logger.info(f"Frames processed: {self.frames_processed}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def check_v4l2_loopback() -> Dict[str, Any]:
    """
    Check if v4l2loopback module is loaded and device exists.

    Returns:
        Dictionary with status information
    """
    import subprocess

    result = {
        "module_loaded": False,
        "device_exists": False,
        "device": V4L2_DEVICE,
        "instructions": [],
    }

    # Check if module is loaded
    try:
        lsmod = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True
        )
        result["module_loaded"] = "v4l2loopback" in lsmod.stdout
    except Exception:
        pass

    # Check if device exists
    result["device_exists"] = os.path.exists(V4L2_DEVICE)

    # Generate instructions
    if not result["module_loaded"]:
        result["instructions"].append(
            "Load v4l2loopback module:\n"
            "  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"RGB_OpenPose\""
        )

    if not result["device_exists"] and result["module_loaded"]:
        result["instructions"].append(
            f"Device {V4L2_DEVICE} not found. Check video_nr parameter."
        )

    return result
