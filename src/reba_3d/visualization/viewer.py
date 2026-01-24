"""
Interactive frame viewer for keypoint visualization.

Allows navigation through pertinent frames for analysis.
"""

import cv2
from pathlib import Path
from typing import Dict, List, Union, Optional

from reba_3d.core.keypoints import load_keypoints_3d
from reba_3d.core.frame_classifier import check_keypoints_dict
from reba_3d.config.settings import (
    KEYPOINT_NAMES,
    MANDATORY_KEYPOINTS_FACE,
    MANDATORY_KEYPOINTS_PROFIL_DROIT,
    MANDATORY_KEYPOINTS_PROFIL_GAUCHE,
)
from reba_3d.io.video_io import VideoReader


class FrameViewer:
    """
    Interactive frame viewer for pertinent frames.

    Allows keyboard navigation through frames that contain valid keypoints.

    Keyboard controls:
        - Left arrow: Previous frame
        - Right arrow: Next frame
        - 'q': Quit
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        keypoints_path: Union[str, Path]
    ):
        """
        Initialize frame viewer.

        Args:
            video_path: Path to the video file
            keypoints_path: Path to keypoints_3d.json
        """
        self.video_path = Path(video_path)
        self.keypoints_path = Path(keypoints_path)

        self.data_json = load_keypoints_3d(keypoints_path)
        self.pertinent_frames = self._classify_frames()
        self.all_frames = self._get_all_frames()

    def _classify_frames(self) -> Dict[str, List[int]]:
        """Classify frames by view type."""
        pertinent_frames = {
            "face": [],
            "profil_droit": [],
            "profil_gauche": []
        }

        for frame_data in self.data_json:
            frame_number = frame_data["frame"]
            keypoints_3d = frame_data["keypoints_3d"]

            if len(keypoints_3d) > 0:
                person_keypoints = {
                    KEYPOINT_NAMES[i]: kp
                    for i, kp in enumerate(keypoints_3d[0])
                    if i < len(KEYPOINT_NAMES)
                }

                if check_keypoints_dict(person_keypoints, MANDATORY_KEYPOINTS_FACE):
                    pertinent_frames["face"].append(frame_number)
                elif check_keypoints_dict(person_keypoints, MANDATORY_KEYPOINTS_PROFIL_DROIT):
                    pertinent_frames["profil_droit"].append(frame_number)
                elif check_keypoints_dict(person_keypoints, MANDATORY_KEYPOINTS_PROFIL_GAUCHE):
                    pertinent_frames["profil_gauche"].append(frame_number)

        # Sort frames
        for key in pertinent_frames:
            pertinent_frames[key] = sorted(pertinent_frames[key])

        return pertinent_frames

    def _get_all_frames(self) -> List[int]:
        """Get sorted list of all pertinent frames."""
        all_frames = set()
        for frames in self.pertinent_frames.values():
            all_frames.update(frames)
        return sorted(all_frames)

    def print_stats(self) -> None:
        """Print frame classification statistics."""
        print(f"[OK] {len(self.pertinent_frames['face'])} frames de vue de face détectées.")
        print(f"[OK] {len(self.pertinent_frames['profil_droit'])} frames de vue de profil droit détectées.")
        print(f"[OK] {len(self.pertinent_frames['profil_gauche'])} frames de vue de profil gauche détectées.")
        print(f"[OK] {len(self.all_frames)} frames pertinentes totales détectées.")

    def run(self) -> None:
        """
        Run interactive viewer.

        Opens a window displaying pertinent frames with keyboard navigation.
        """
        if not self.all_frames:
            print("[ERROR] Aucune frame pertinente détectée, arrêt du programme.")
            return

        self.print_stats()

        with VideoReader(self.video_path) as reader:
            print(f"[OK] Vidéo chargée: {self.video_path}")
            print(f"- FPS: {reader.fps}")
            print(f"- Nombre de frames: {reader.frame_count}")
            print(f"- Résolution: {reader.width}x{reader.height}")

            current_index = 0
            window_name = "Frame Pertinente"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while current_index < len(self.all_frames):
                frame_number = self.all_frames[current_index]

                # Seek to frame
                reader.seek(frame_number)
                ret, frame = reader.read()

                if not ret:
                    print(f"📋 Frame {frame_number} introuvable ou hors limite.")
                    break

                # Annotate frame
                view_type = self._get_view_type(frame_number)
                cv2.putText(
                    frame,
                    f"Frame: {frame_number}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Vue: {view_type}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"[{current_index + 1}/{len(self.all_frames)}]",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(window_name, frame)

                # Handle keyboard input
                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("📋 Interruption par l'utilisateur (touche 'q').")
                    break
                elif key == 81 or key == 2:  # Left arrow
                    current_index = max(0, current_index - 1)
                elif key == 83 or key == 3:  # Right arrow
                    current_index = min(len(self.all_frames) - 1, current_index + 1)

            cv2.destroyAllWindows()

    def _get_view_type(self, frame_number: int) -> str:
        """Get view type for a frame."""
        if frame_number in self.pertinent_frames["face"]:
            return "face"
        elif frame_number in self.pertinent_frames["profil_droit"]:
            return "profil_droit"
        elif frame_number in self.pertinent_frames["profil_gauche"]:
            return "profil_gauche"
        return "inconnu"


def view_keypoints(
    video_path: Union[str, Path],
    keypoints_path: Union[str, Path]
) -> None:
    """
    Launch interactive frame viewer.

    Args:
        video_path: Path to the video file
        keypoints_path: Path to keypoints_3d.json
    """
    viewer = FrameViewer(video_path, keypoints_path)
    viewer.run()
