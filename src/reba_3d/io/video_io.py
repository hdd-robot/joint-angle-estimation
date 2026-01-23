"""
Video input/output utilities.

Provides helper classes for video reading and writing using OpenCV.
"""

import cv2
from pathlib import Path
from typing import Optional, Tuple, Generator, Union


class VideoReader:
    """
    Video reader with context manager support.

    Provides convenient iteration over video frames.

    Example:
        >>> with VideoReader("video.avi") as reader:
        ...     for frame_num, frame in reader:
        ...         process(frame)
    """

    def __init__(self, video_path: Union[str, Path]):
        """
        Initialize video reader.

        Args:
            video_path: Path to the video file

        Raises:
            FileNotFoundError: If the video file doesn't exist
            RuntimeError: If the video cannot be opened
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Vidéo introuvable: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """Video frames per second."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return self._frame_count

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def resolution(self) -> Tuple[int, int]:
        """Frame resolution (width, height)."""
        return self._width, self._height

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return self._frame_count / self._fps if self._fps > 0 else 0

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """
        Read the next frame.

        Returns:
            Tuple (success, frame) where success is True if frame was read
        """
        return self.cap.read()

    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame index to seek to

        Returns:
            True if seek was successful
        """
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number))

    def __iter__(self) -> Generator[Tuple[int, "cv2.Mat"], None, None]:
        """Iterate over all frames."""
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_num, frame
            frame_num += 1

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def release(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoWriter:
    """
    Video writer with context manager support.

    Example:
        >>> with VideoWriter("output.avi", fps=30, resolution=(640, 480)) as writer:
        ...     writer.write(frame)
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        resolution: Tuple[int, int],
        codec: str = "XVID"
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path for output video file
            fps: Frames per second
            resolution: (width, height) tuple
            codec: FourCC codec string (default: "XVID")
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            resolution
        )

        self._fps = fps
        self._resolution = resolution
        self._frame_count = 0

    @property
    def fps(self) -> float:
        """Video frames per second."""
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        """Frame resolution (width, height)."""
        return self._resolution

    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count

    def write(self, frame: "cv2.Mat") -> None:
        """
        Write a frame to the video.

        Args:
            frame: Frame to write (BGR format)
        """
        self.writer.write(frame)
        self._frame_count += 1

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def release(self) -> None:
        """Release video writer resources."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Get information about a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with video properties:
        {
            "fps": float,
            "frame_count": int,
            "width": int,
            "height": int,
            "duration": float,  # in seconds
        }

    Raises:
        FileNotFoundError: If the video file doesn't exist
        RuntimeError: If the video cannot be opened
    """
    with VideoReader(video_path) as reader:
        return {
            "fps": reader.fps,
            "frame_count": reader.frame_count,
            "width": reader.width,
            "height": reader.height,
            "duration": reader.duration,
        }
