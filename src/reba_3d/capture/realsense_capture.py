"""
RealSense capture and OpenPose integration.

Processes RealSense .bag files to extract 3D skeleton keypoints.
"""

import os
import cv2
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

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
    """Lazy initialization of OpenPose."""
    global _op
    if _op is None:
        os.environ["PYTHONPATH"] += os.pathsep + openpose_path + "/build/python"
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

        print(f"⏱️ Durée du .bag: {self._duration:.2f} secondes")
        print(f"🎯 Nombre de frames estimées: {self._expected_frames}")

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
                print(f"Fin du fichier .bag: {e}")
                break


class OpenPoseDetector:
    """
    OpenPose skeleton detector wrapper.

    Example:
        >>> detector = OpenPoseDetector("/path/to/openpose")
        >>> keypoints = detector.detect(frame)
    """

    def __init__(self, openpose_path: str):
        """
        Initialize OpenPose detector.

        Args:
            openpose_path: Path to OpenPose installation
        """
        self.openpose_path = openpose_path
        self.op = _init_openpose(openpose_path)

        params = {"model_folder": openpose_path + "/models/"}
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


def process_bag_file(
    bag_file: Union[str, Path],
    output_dir: Union[str, Path],
    openpose_path: str,
    show_preview: bool = True
) -> Dict[str, str]:
    """
    Process a .bag file to extract 3D keypoints.

    Args:
        bag_file: Path to input .bag file
        output_dir: Directory for output files
        openpose_path: Path to OpenPose installation
        show_preview: Whether to show preview window (default: True)

    Returns:
        Dictionary with output file paths:
        {
            "raw_video": path,
            "openpose_video": path,
            "keypoints_json": path,
        }
    """
    import numpy as np

    bag_file = Path(bag_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    output_video_path = output_dir / "output.avi"
    output_openpose_path = output_dir / "output_openpose.avi"
    keypoints_json_path = output_dir / "keypoints_3d.json"

    # Clean up existing files
    for path in [output_video_path, output_openpose_path, keypoints_json_path]:
        if path.exists():
            path.unlink()
            print(f"🗑 Fichier supprimé: {path}")

    # Initialize OpenPose
    detector = OpenPoseDetector(openpose_path)

    # Video writers (initialized after first frame)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_raw = None
    out_openpose = None
    keypoints_3d_data = []

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
                print(f"Frame {frame_count}: Keypoints détectés.")

            # Preview
            if show_preview:
                cv2.imshow("OpenPose + RealSense", output_frame)

            # Initialize video writers on first frame
            if out_raw is None:
                height, width = frame.shape[:2]
                print(f"📸 FPS utilisé: {fps}")
                out_raw = cv2.VideoWriter(
                    str(output_video_path), fourcc, fps, (width, height)
                )
                out_openpose = cv2.VideoWriter(
                    str(output_openpose_path), fourcc, fps, (width, height)
                )

            out_raw.write(frame)
            out_openpose.write(output_frame)

            if show_preview and cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Interruption utilisateur.")
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

    print(f"✅ {len(keypoints_3d_data)} frames enregistrées.")
    print(f"✅ Données 3D: {keypoints_json_path}")

    # Recalibrate video FPS
    actual_frame_count = len(keypoints_3d_data)
    if actual_frame_count > 0:
        with RealSenseCapture(bag_file) as capture:
            estimated_fps = actual_frame_count / capture.duration

        print(f"🎯 FPS réel estimé: {estimated_fps:.2f}")
        _recalibrate_video(output_video_path, estimated_fps)
        _recalibrate_video(output_openpose_path, estimated_fps)

    return {
        "raw_video": str(output_video_path),
        "openpose_video": str(output_openpose_path),
        "keypoints_json": str(keypoints_json_path),
    }


def _recalibrate_video(video_path: Path, target_fps: float) -> None:
    """Recalibrate video FPS using ffmpeg."""
    tmp_path = video_path.with_suffix(".tmp.avi")

    print(f"🎞 Recalibrage avec ffmpeg: {video_path}")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter:v", f"fps={target_fps:.2f}",
        str(tmp_path)
    ], capture_output=True)

    tmp_path.replace(video_path)
    print(f"✅ Fichier recalibré: {video_path}")
