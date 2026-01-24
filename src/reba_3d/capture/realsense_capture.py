"""
RealSense capture and OpenPose integration.

Processes RealSense .bag files to extract 3D skeleton keypoints.
Supports:
- local: OpenPose installed locally (pyopenpose)
- v4l2: OpenPose via Docker with V4L2 streaming (real-time)
"""

import os
import cv2
import json
import glob
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np

from ..config.settings import (
    OPENPOSE_MODE,
    V4L2_DEVICE,
    V4L2_OPENPOSE_JSON_DIR,
    RECORDING_ENABLED,
    RECORDING_CODEC,
    SAVE_RAW_VIDEO,
    SAVE_OPENPOSE_VIDEO,
)
from ..utils.logger import get_logger

# Logger pour ce module
logger = get_logger("capture.realsense")

# Lazy imports for optional dependencies
_rs = None
_op = None


def _init_realsense():
    """Lazy initialization of pyrealsense2."""
    global _rs
    if _rs is None:
        import pyrealsense2 as rs
        _rs = rs
    return _rs


def _init_openpose(openpose_path: str):
    """Lazy initialization of OpenPose (mode local uniquement)."""
    global _op
    if _op is None:
        import sys
        # Ajouter le chemin des bindings Python OpenPose
        pyopenpose_path = openpose_path + "/build/python/openpose"
        if pyopenpose_path not in sys.path:
            sys.path.insert(0, pyopenpose_path)
        import pyopenpose as op
        _op = op
    return _op


class RealSenseCapture:
    """
    RealSense capture with context manager support.

    Processes .bag files from Intel RealSense cameras.

    Example:
        >>> with RealSenseCapture("recording.bag") as capture:
        ...     for color_frame, depth_frame in capture:
        ...         process(color_frame, depth_frame)
    """

    def __init__(self, bag_file: Union[str, Path], real_time: bool = False):
        """
        Initialize RealSense capture from a .bag file.

        Args:
            bag_file: Path to the .bag recording
            real_time: Whether to play back in real time (default: False)

        Raises:
            FileNotFoundError: If the bag file doesn't exist
        """
        self.bag_file = Path(bag_file)
        if not self.bag_file.exists():
            raise FileNotFoundError(f"Fichier .bag introuvable: {bag_file}")

        self.real_time = real_time
        self.rs = _init_realsense()
        self.pipeline = None
        self.align = None
        self.depth_intrinsics = None
        self._fps = 30.0
        self._duration = 0.0
        self._expected_frames = 0

    def __enter__(self) -> "RealSenseCapture":
        """Start RealSense pipeline."""
        self.pipeline = self.rs.pipeline()
        config = self.rs.config()
        config.enable_device_from_file(str(self.bag_file))

        self.align = self.rs.align(self.rs.stream.color)

        profile = self.pipeline.start(config)
        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(self.real_time)

        # Get duration
        duration = playback.get_duration()
        self._duration = duration.total_seconds()

        # Get FPS
        try:
            self._fps = profile.get_stream(
                self.rs.stream.color
            ).as_video_stream_profile().fps()
        except:
            self._fps = 30.0

        self._expected_frames = int(self._duration * self._fps)

        # Get depth intrinsics
        self.depth_intrinsics = profile.get_stream(
            self.rs.stream.depth
        ).as_video_stream_profile().get_intrinsics()

        logger.info(f"Duree du .bag: {self._duration:.2f} secondes")
        logger.info(f"Nombre de frames estimees: {self._expected_frames}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop RealSense pipeline."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    @property
    def fps(self) -> float:
        """Frames per second."""
        return self._fps

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self._duration

    @property
    def expected_frame_count(self) -> int:
        """Expected number of frames."""
        return self._expected_frames

    def get_3d_point(
        self,
        depth_frame,
        x: int,
        y: int
    ) -> List[float]:
        """
        Get 3D point from 2D pixel coordinates using depth.

        Args:
            depth_frame: RealSense depth frame
            x: Pixel x coordinate
            y: Pixel y coordinate

        Returns:
            [x, y, z] 3D coordinates in meters
        """
        x = max(0, min(x, depth_frame.get_width() - 1))
        y = max(0, min(y, depth_frame.get_height() - 1))
        depth = depth_frame.get_distance(x, y)

        if depth > 0:
            return self.rs.rs2_deproject_pixel_to_point(
                self.depth_intrinsics, [x, y], depth
            )
        return [0, 0, 0]

    def __iter__(self):
        """Iterate over frames."""
        frame_count = 0
        while frame_count < self._expected_frames:
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    break

                yield frame_count, color_frame, depth_frame
                frame_count += 1

            except Exception as e:
                logger.debug(f"Fin du fichier .bag: {e}")
                break


class OpenPoseDetector:
    """
    OpenPose skeleton detector wrapper (mode local avec pyopenpose).

    Example:
        >>> detector = OpenPoseDetector("/path/to/openpose")
        >>> keypoints = detector.detect(frame)
    """

    def __init__(self, openpose_path: str, net_resolution: str = "-1x160"):
        """
        Initialize OpenPose detector.

        Args:
            openpose_path: Path to OpenPose installation
            net_resolution: Network resolution (default: "-1x160" for low VRAM)
        """
        self.openpose_path = openpose_path
        self.op = _init_openpose(openpose_path)

        params = {
            "model_folder": openpose_path + "/models/",
            "model_pose": "BODY_25",
            "net_resolution": net_resolution,  # Réduire pour économiser VRAM
            "number_people_max": 1,  # Une seule personne
        }
        self.wrapper = self.op.WrapperPython()
        self.wrapper.configure(params)
        self.wrapper.start()

    def detect(self, frame) -> Tuple[Any, Optional[Any]]:
        """
        Detect skeleton keypoints in a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            Tuple (output_frame, keypoints) where keypoints may be None
        """
        datum = self.op.Datum()
        datum.cvInputData = frame
        datum_ptr = self.op.VectorDatum()
        datum_ptr.append(datum)
        self.wrapper.emplaceAndPop(datum_ptr)

        output_frame = datum_ptr[0].cvOutputData
        keypoints = datum_ptr[0].poseKeypoints

        return output_frame, keypoints


def create_openpose_detector(
    mode: str = None,
    openpose_path: str = None,
) -> OpenPoseDetector:
    """
    Factory function to create the appropriate OpenPose detector.

    Args:
        mode: "local" or "v4l2" (default from settings)
        openpose_path: Path to OpenPose (for local mode)

    Returns:
        OpenPoseDetector instance

    Note:
        For "v4l2" mode, use V4L2OpenPoseStreamer directly instead.
    """
    mode = mode or OPENPOSE_MODE

    if mode == "local":
        if not openpose_path:
            from ..config.settings import OPENPOSE_PATH
            openpose_path = OPENPOSE_PATH
        logger.info(f"Mode OpenPose: local ({openpose_path})")
        return OpenPoseDetector(openpose_path)

    elif mode == "v4l2":
        raise ValueError(
            "Mode V4L2: utilisez V4L2OpenPoseStreamer directement.\n"
            "Voir process_bag_file_realtime() ou reba_3d.capture.v4l2_stream"
        )

    else:
        raise ValueError(
            f"Mode OpenPose inconnu: {mode}. "
            f"Utilisez 'local' ou 'v4l2'."
        )


def process_bag_file(
    bag_file: Union[str, Path],
    output_dir: Union[str, Path],
    openpose_path: str = None,
    openpose_mode: str = None,
    show_preview: bool = True,
    save_video: bool = None,
) -> Dict[str, str]:
    """
    Process a .bag file to extract 3D keypoints (mode local uniquement).

    Args:
        bag_file: Path to input .bag file
        output_dir: Directory for output files
        openpose_path: Path to OpenPose installation (for local mode)
        openpose_mode: "local" (default from settings)
        show_preview: Whether to show preview window (default: True)
        save_video: Whether to save video files (default from config)

    Returns:
        Dictionary with output file paths:
        {
            "raw_video": path or None,
            "openpose_video": path or None,
            "keypoints_json": path,
        }

    Note:
        Pour le mode V4L2 (temps réel avec Docker), utilisez process_bag_file_realtime().
    """
    # Use config default if not specified
    if save_video is None:
        save_video = RECORDING_ENABLED

    bag_file = Path(bag_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    output_video_path = output_dir / "output.avi" if save_video and SAVE_RAW_VIDEO else None
    output_openpose_path = output_dir / "output_openpose.avi" if save_video and SAVE_OPENPOSE_VIDEO else None
    keypoints_json_path = output_dir / "keypoints_3d.json"

    # Clean up existing files
    paths_to_clean = [keypoints_json_path]
    if output_video_path:
        paths_to_clean.append(output_video_path)
    if output_openpose_path:
        paths_to_clean.append(output_openpose_path)

    for path in paths_to_clean:
        if path.exists():
            path.unlink()
            logger.debug(f"Fichier supprime: {path}")

    # Initialize OpenPose (mode local uniquement)
    detector = create_openpose_detector(
        mode=openpose_mode,
        openpose_path=openpose_path,
    )

    # Video writers (initialized after first frame if recording enabled)
    fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)
    out_raw = None
    out_openpose = None
    keypoints_3d_data = []

    if save_video:
        logger.info("Enregistrement video active")

    with RealSenseCapture(bag_file) as capture:
        fps = capture.fps

        for frame_count, color_frame, depth_frame in capture:
            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect keypoints
            output_frame, keypoints = detector.detect(frame)

            # Extract 3D keypoints
            if keypoints is not None:
                person_kps = []
                for person in keypoints:
                    joints = []
                    for kp in person:
                        x, y, conf = kp
                        x, y = int(x), int(y)
                        p3d = capture.get_3d_point(depth_frame, x, y)
                        joints.append({
                            "x": float(p3d[0]),
                            "y": float(p3d[1]),
                            "z": float(p3d[2]),
                            "confidence": float(conf)
                        })
                    person_kps.append(joints)
                keypoints_3d_data.append({
                    "frame": frame_count,
                    "keypoints_3d": person_kps
                })
                logger.debug(f"Frame {frame_count}: Keypoints detectes.")

            # Preview
            if show_preview:
                cv2.imshow("OpenPose + RealSense", output_frame)

            # Initialize video writers on first frame (only if recording enabled)
            if save_video and out_raw is None:
                height, width = frame.shape[:2]
                logger.info(f"FPS utilise: {fps}")
                if output_video_path:
                    out_raw = cv2.VideoWriter(
                        str(output_video_path), fourcc, fps, (width, height)
                    )
                if output_openpose_path:
                    out_openpose = cv2.VideoWriter(
                        str(output_openpose_path), fourcc, fps, (width, height)
                    )

            # Write frames (only if writers exist)
            if out_raw:
                out_raw.write(frame)
            if out_openpose:
                out_openpose.write(output_frame)

            if show_preview and cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Interruption utilisateur.")
                break

    # Cleanup
    if out_raw:
        out_raw.release()
    if out_openpose:
        out_openpose.release()
    cv2.destroyAllWindows()

    # Save keypoints
    with open(keypoints_json_path, 'w') as f:
        json.dump(keypoints_3d_data, f, indent=4)

    logger.info(f"{len(keypoints_3d_data)} frames enregistrees.")
    logger.info(f"Donnees 3D: {keypoints_json_path}")

    # Recalibrate video FPS (only if recording was enabled)
    actual_frame_count = len(keypoints_3d_data)
    if save_video and actual_frame_count > 0:
        with RealSenseCapture(bag_file) as capture:
            estimated_fps = actual_frame_count / capture.duration

        logger.info(f"FPS reel estime: {estimated_fps:.2f}")
        if output_video_path and output_video_path.exists():
            _recalibrate_video(output_video_path, estimated_fps)
        if output_openpose_path and output_openpose_path.exists():
            _recalibrate_video(output_openpose_path, estimated_fps)

    return {
        "raw_video": str(output_video_path) if output_video_path else None,
        "openpose_video": str(output_openpose_path) if output_openpose_path else None,
        "keypoints_json": str(keypoints_json_path),
    }


def _recalibrate_video(video_path: Path, target_fps: float) -> None:
    """Recalibrate video FPS using ffmpeg."""
    tmp_path = video_path.with_suffix(".tmp.avi")

    logger.info(f"Recalibrage avec ffmpeg: {video_path}")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter:v", f"fps={target_fps:.2f}",
        str(tmp_path)
    ], capture_output=True)

    tmp_path.replace(video_path)
    logger.info(f"Fichier recalibre: {video_path}")


def process_bag_file_realtime(
    bag_file: Union[str, Path],
    output_dir: Union[str, Path],
    v4l2_device: str = None,
    json_dir: str = None,
    show_preview: bool = True,
    save_video: bool = None,
    collect_delay: float = 0.5,
) -> Dict[str, str]:
    """
    Process a .bag file with real-time V4L2 streaming to OpenPose.

    This function streams RGB frames to a V4L2 loopback device that OpenPose
    reads via --camera mode. Depth frames are buffered and used for 3D
    projection when OpenPose outputs are ready.

    Prerequisites:
        1. V4L2 loopback module loaded:
           sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="RGB_OpenPose"

        2. OpenPose Docker running:
           docker run -it --gpus all --ipc=host --device=/dev/video10 \\
             -v /tmp/openpose/out:/data/out cwaffles/openpose \\
             ./build/examples/openpose/openpose.bin --camera 10 \\
             --write_json /data/out --model_pose COCO --display 0

    Args:
        bag_file: Path to input .bag file
        output_dir: Directory for output files
        v4l2_device: V4L2 device path (default: /dev/video10)
        json_dir: OpenPose JSON output directory (default: /tmp/openpose/out)
        show_preview: Whether to show preview window
        save_video: Whether to save video files (default from config)
        collect_delay: Delay after streaming to collect remaining results

    Returns:
        Dictionary with output file paths
    """
    from .v4l2_stream import V4L2OpenPoseStreamer, check_v4l2_loopback

    # Use config default if not specified
    if save_video is None:
        save_video = RECORDING_ENABLED

    # Check V4L2 setup
    v4l2_status = check_v4l2_loopback()
    if not v4l2_status["device_exists"]:
        for instruction in v4l2_status["instructions"]:
            logger.warning(instruction)
        raise RuntimeError("V4L2 loopback not ready. See instructions above.")

    bag_file = Path(bag_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    output_video_path = output_dir / "output.avi" if save_video and SAVE_RAW_VIDEO else None
    keypoints_json_path = output_dir / "keypoints_3d.json"

    # Clean up existing files
    paths_to_clean = [keypoints_json_path]
    if output_video_path:
        paths_to_clean.append(output_video_path)

    for path in paths_to_clean:
        if path.exists():
            path.unlink()
            logger.debug(f"Fichier supprime: {path}")

    if save_video:
        logger.info("Enregistrement video active")

    keypoints_3d_data = []

    with RealSenseCapture(bag_file, real_time=True) as capture:
        fps = capture.fps
        width = 640  # Default RealSense color width
        height = 480

        # Get frame dimensions from first frame
        frames = capture.pipeline.wait_for_frames()
        aligned = capture.align.process(frames)
        color = aligned.get_color_frame()
        if color:
            width = color.get_width()
            height = color.get_height()

        # Video writer (only if recording enabled)
        fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)
        out_raw = None
        if save_video and output_video_path:
            out_raw = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )

        logger.info(f"Streaming en temps reel: {width}x{height} @ {fps} FPS")

        with V4L2OpenPoseStreamer(
            v4l2_device=v4l2_device,
            json_dir=json_dir,
            width=width,
            height=height,
            fps=fps,
        ) as streamer:

            for frame_count, color_frame, depth_frame in capture:
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Push frame to V4L2 streamer
                streamer.push_frame(frame, depth_frame, frame_count)

                # Collect any available 3D keypoints
                new_results = streamer.get_3d_keypoints(capture)
                keypoints_3d_data.extend(new_results)

                # Preview
                if show_preview:
                    # Show frame with processing info
                    display_frame = frame.copy()
                    cv2.putText(
                        display_frame,
                        f"Frame: {frame_count} | Processed: {len(keypoints_3d_data)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("V4L2 Streaming", display_frame)

                # Write raw video (only if recording)
                if out_raw:
                    out_raw.write(frame)

                if show_preview and cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Interruption utilisateur.")
                    break

                # Status update every 30 frames
                if frame_count % 30 == 0:
                    logger.info(
                        f"Frame {frame_count}: "
                        f"{len(keypoints_3d_data)} keypoints collectes"
                    )

            # Wait for remaining results
            logger.info(f"Attente des resultats restants ({collect_delay}s)...")
            time.sleep(collect_delay)
            final_results = streamer.get_3d_keypoints(capture)
            keypoints_3d_data.extend(final_results)

    # Cleanup
    if out_raw:
        out_raw.release()
    cv2.destroyAllWindows()

    # Sort by frame number
    keypoints_3d_data.sort(key=lambda x: x["frame"])

    # Save keypoints
    with open(keypoints_json_path, 'w') as f:
        json.dump(keypoints_3d_data, f, indent=4)

    logger.info(f"{len(keypoints_3d_data)} frames avec keypoints 3D.")
    logger.info(f"Donnees 3D: {keypoints_json_path}")

    return {
        "raw_video": str(output_video_path) if output_video_path else None,
        "keypoints_json": str(keypoints_json_path),
    }
