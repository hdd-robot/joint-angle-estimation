"""
Video annotation with REBA risk labels.

Overlays REBA risk level information on video frames.
"""

import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from reba_3d.config.settings import VIDEO_COLORS
from reba_3d.io.json_io import read_risk_times_json
from reba_3d.io.video_io import VideoReader, VideoWriter


class VideoAnnotator:
    """
    Annotates video frames with REBA risk information.

    Attributes:
        risk_times: Dictionary mapping risk labels to time intervals
        fps: Video frames per second
    """

    def __init__(
        self,
        risk_times: Dict[str, List[Tuple[float, float]]],
        fps: float = 15.0
    ):
        """
        Initialize video annotator.

        Args:
            risk_times: Dictionary mapping risk labels to (start, end) intervals
            fps: Video frames per second (default: 15.0)
        """
        self.risk_times = risk_times
        self.fps = fps
        self._frame_to_label = self._build_frame_labels()

    def _build_frame_labels(self) -> Dict[int, str]:
        """Build mapping from frame number to risk label."""
        frame_to_label = {}

        for label, intervals in self.risk_times.items():
            for start_s, end_s in intervals:
                start_frame = int(start_s * self.fps)
                end_frame = int(end_s * self.fps)
                for frame_num in range(start_frame, end_frame):
                    frame_to_label[frame_num] = label

        return frame_to_label

    def get_label(self, frame_number: int) -> str:
        """
        Get risk label for a specific frame.

        Args:
            frame_number: Frame index

        Returns:
            Risk label string, or "invalid" if no label assigned
        """
        return self._frame_to_label.get(frame_number, "invalid")

    def get_color(self, label: str) -> Tuple[int, int, int]:
        """
        Get BGR color for a risk label.

        Args:
            label: Risk label string

        Returns:
            BGR color tuple
        """
        return VIDEO_COLORS.get(label, VIDEO_COLORS["invalid"])

    def annotate_frame(
        self,
        frame: "cv2.Mat",
        frame_number: int,
        show_time: bool = True
    ) -> "cv2.Mat":
        """
        Annotate a single frame with REBA information.

        Args:
            frame: Input frame (BGR)
            frame_number: Frame index
            show_time: Whether to show time information (default: True)

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        label = self.get_label(frame_number)
        bg_color = self.get_color(label)
        frame_time_sec = frame_number / self.fps

        # Frame info (top left)
        cv2.putText(
            annotated,
            f"Frame: {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        if show_time:
            cv2.putText(
                annotated,
                f"Time: {frame_time_sec:.2f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

        # REBA label (top center)
        cv2.rectangle(annotated, (310, 0), (710, 40), bg_color, -1)
        cv2.putText(
            annotated,
            f"REBA: {label}",
            (320, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        return annotated


def annotate_with_reba(
    input_video: Union[str, Path],
    risk_times: Union[Dict, str, Path],
    output_path: Optional[Union[str, Path]] = None,
    show_preview: bool = False
) -> str:
    """
    Annotate a video with REBA risk labels.

    Args:
        input_video: Path to input video file
        risk_times: Risk times dictionary or path to risk_times.json
        output_path: Path for output video (default: adds '_annotated' suffix)
        show_preview: Whether to show preview window (default: False)

    Returns:
        Path to annotated video file
    """
    input_video = Path(input_video)

    # Load risk times if path provided
    if isinstance(risk_times, (str, Path)):
        risk_times = read_risk_times_json(risk_times)

    # Determine output path
    if output_path is None:
        output_path = input_video.parent / f"{input_video.stem}_annotated{input_video.suffix}"
    output_path = Path(output_path)

    # Open input video
    with VideoReader(input_video) as reader:
        print(f"✅ Vidéo chargée: {input_video}")
        print(f"- FPS: {reader.fps}")
        print(f"- Nombre de frames: {reader.frame_count}")
        print(f"- Résolution: {reader.width}x{reader.height}")

        # Create annotator
        annotator = VideoAnnotator(risk_times, reader.fps)

        # Create output video
        with VideoWriter(
            output_path,
            reader.fps,
            reader.resolution
        ) as writer:
            if show_preview:
                cv2.namedWindow("REBA Annotée", cv2.WINDOW_NORMAL)

            for frame_num, frame in reader:
                annotated = annotator.annotate_frame(frame, frame_num)
                writer.write(annotated)

                if show_preview:
                    cv2.imshow("REBA Annotée", annotated)
                    delay = int(1000 / reader.fps)
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        print("⏹️ Interruption par l'utilisateur.")
                        break

            if show_preview:
                cv2.destroyAllWindows()
                cv2.waitKey(1)

    print(f"✅ Vidéo annotée sauvegardée: {output_path}")
    return str(output_path)
