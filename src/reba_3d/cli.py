"""
Command-line interface for reba_3d package.

Provides commands for capture, analysis, annotation, and visualization.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from reba_3d.config.settings import (
    OPENPOSE_PATH,
    OUTPUT_DIR,
    OPENPOSE_MODE,
    V4L2_DEVICE,
    V4L2_OPENPOSE_JSON_DIR,  # Default: /dev/shm/openpose_json (RAM)
    RECORDING_ENABLED,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="reba3d",
        description="3D Ergonomic Posture Analysis System using REBA methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  reba3d gui                                    # Launch interactive GUI
  reba3d capture --bag recording.bag --output ./output
  reba3d analyze --input keypoints_3d.json --output risk_times.json
  reba3d annotate --video output_openpose.avi --risks risk_times.json
  reba3d view --video output_openpose.avi --keypoints keypoints_3d.json
  reba3d pipeline --bag recording.bag --output ./output
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands"
    )

    # GUI command
    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch interactive GUI application",
        description="Start the REBA 3D graphical user interface"
    )
    gui_parser.add_argument(
        "--bag-dir",
        help="Directory containing .bag files"
    )
    gui_parser.add_argument(
        "--bag",
        help="Default .bag file to load"
    )

    # Capture command
    capture_parser = subparsers.add_parser(
        "capture",
        help="Extract 3D skeleton from RealSense .bag file",
        description="Process a RealSense .bag file with OpenPose to extract 3D keypoints"
    )
    capture_parser.add_argument(
        "--bag", "-b",
        required=True,
        help="Path to input .bag file"
    )
    capture_parser.add_argument(
        "--output", "-o",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    capture_parser.add_argument(
        "--openpose", "-p",
        default=OPENPOSE_PATH,
        help=f"Path to OpenPose installation for local mode (default: {OPENPOSE_PATH})"
    )
    capture_parser.add_argument(
        "--openpose-mode", "-m",
        choices=["local", "v4l2"],
        default=OPENPOSE_MODE,
        help=f"OpenPose execution mode (default: {OPENPOSE_MODE})"
    )
    capture_parser.add_argument(
        "--v4l2-device",
        default=V4L2_DEVICE,
        help=f"V4L2 loopback device for 'v4l2' mode (default: {V4L2_DEVICE})"
    )
    capture_parser.add_argument(
        "--v4l2-json-dir",
        default=V4L2_OPENPOSE_JSON_DIR,
        help=f"OpenPose JSON output directory for 'v4l2' mode (default: {V4L2_OPENPOSE_JSON_DIR})"
    )
    capture_parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window"
    )
    capture_parser.add_argument(
        "--save-video",
        action="store_true",
        default=None,
        help="Enable video recording (default: from config)"
    )
    capture_parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Disable video recording"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Calculate REBA scores from keypoints",
        description="Analyze 3D keypoints and calculate REBA risk scores"
    )
    analyze_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to keypoints_3d.json"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        default="risk_times.json",
        help="Path for output risk_times.json (default: risk_times.json)"
    )
    analyze_parser.add_argument(
        "--load-malus",
        type=int,
        default=1,
        help="Load/force malus score (default: 1)"
    )
    analyze_parser.add_argument(
        "--coupling-malus",
        type=int,
        default=1,
        help="Coupling quality malus score (default: 1)"
    )
    analyze_parser.add_argument(
        "--activity-malus",
        type=int,
        default=1,
        help="Activity malus score (default: 1)"
    )

    # Annotate command
    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Annotate video with REBA risk labels",
        description="Overlay REBA risk level information on video"
    )
    annotate_parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to input video"
    )
    annotate_parser.add_argument(
        "--risks", "-r",
        required=True,
        help="Path to risk_times.json"
    )
    annotate_parser.add_argument(
        "--output", "-o",
        help="Path for output video (default: adds '_annotated' suffix)"
    )
    annotate_parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview window while processing"
    )

    # View command
    view_parser = subparsers.add_parser(
        "view",
        help="Interactive keypoint frame viewer",
        description="Browse pertinent frames with keyboard navigation"
    )
    view_parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to video file"
    )
    view_parser.add_argument(
        "--keypoints", "-k",
        required=True,
        help="Path to keypoints_3d.json"
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run complete pipeline (capture → analyze → annotate)",
        description="Process a .bag file through the complete REBA analysis pipeline"
    )
    pipeline_parser.add_argument(
        "--bag", "-b",
        required=True,
        help="Path to input .bag file"
    )
    pipeline_parser.add_argument(
        "--output", "-o",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    pipeline_parser.add_argument(
        "--openpose", "-p",
        default=OPENPOSE_PATH,
        help=f"Path to OpenPose installation for local mode (default: {OPENPOSE_PATH})"
    )
    pipeline_parser.add_argument(
        "--openpose-mode", "-m",
        choices=["local", "v4l2"],
        default=OPENPOSE_MODE,
        help=f"OpenPose execution mode (default: {OPENPOSE_MODE})"
    )
    pipeline_parser.add_argument(
        "--v4l2-device",
        default=V4L2_DEVICE,
        help=f"V4L2 loopback device for 'v4l2' mode (default: {V4L2_DEVICE})"
    )
    pipeline_parser.add_argument(
        "--v4l2-json-dir",
        default=V4L2_OPENPOSE_JSON_DIR,
        help=f"OpenPose JSON output directory for 'v4l2' mode (default: {V4L2_OPENPOSE_JSON_DIR})"
    )
    pipeline_parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview windows"
    )
    pipeline_parser.add_argument(
        "--save-video",
        action="store_true",
        default=None,
        help="Enable video recording (default: from config)"
    )
    pipeline_parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Disable video recording"
    )

    return parser


def cmd_gui(args) -> int:
    """Execute GUI command."""
    from reba_3d.gui.app import REBAApp
    from reba_3d.utils.logger import setup_logging

    setup_logging()

    try:
        app = REBAApp(
            bag_directory=args.bag_dir,
            default_bag_name=args.bag
        )
        app.run()
        return 0
    except Exception as e:
        print(f"[ERROR] Erreur GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_capture(args) -> int:
    """Execute capture command."""
    # Resolve save_video from CLI args
    save_video = None
    if args.no_save_video:
        save_video = False
    elif args.save_video:
        save_video = True
    # Otherwise None = use config default

    try:
        if args.openpose_mode == "v4l2":
            # Mode V4L2 streaming temps réel
            from reba_3d.capture.realsense_capture import process_bag_file_realtime

            result = process_bag_file_realtime(
                bag_file=args.bag,
                output_dir=args.output,
                v4l2_device=args.v4l2_device,
                json_dir=args.v4l2_json_dir,
                show_preview=not args.no_preview,
                save_video=save_video,
            )
            print(f"\n[OK] Capture temps réel terminée:")
            if result.get('raw_video'):
                print(f"  - Vidéo brute: {result['raw_video']}")
            print(f"  - Keypoints JSON: {result['keypoints_json']}")

        else:
            # Mode local (pyopenpose)
            from reba_3d.capture.realsense_capture import process_bag_file

            result = process_bag_file(
                bag_file=args.bag,
                output_dir=args.output,
                openpose_path=args.openpose,
                openpose_mode=args.openpose_mode,
                show_preview=not args.no_preview,
                save_video=save_video,
            )
            print(f"\n[OK] Capture terminée:")
            if result.get('raw_video'):
                print(f"  - Vidéo brute: {result['raw_video']}")
            if result.get('openpose_video'):
                print(f"  - Vidéo OpenPose: {result['openpose_video']}")
            print(f"  - Keypoints JSON: {result['keypoints_json']}")

        return 0
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_analyze(args) -> int:
    """Execute analyze command."""
    from reba_3d.reba.risk_assessment import assess_video

    try:
        results = assess_video(
            keypoints_path=args.input,
            output_path=args.output,
            load_malus=args.load_malus,
            coupling_malus=args.coupling_malus,
            activity_malus=args.activity_malus
        )
        print(f"\n[OK] Analyse terminée: {results['num_windows']} fenêtres analysées")
        return 0
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        return 1


def cmd_annotate(args) -> int:
    """Execute annotate command."""
    from reba_3d.visualization.annotator import annotate_with_reba

    try:
        output_path = annotate_with_reba(
            input_video=args.video,
            risk_times=args.risks,
            output_path=args.output,
            show_preview=args.preview
        )
        print(f"\n[OK] Annotation terminée: {output_path}")
        return 0
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        return 1


def cmd_view(args) -> int:
    """Execute view command."""
    from reba_3d.visualization.viewer import view_keypoints

    try:
        view_keypoints(
            video_path=args.video,
            keypoints_path=args.keypoints
        )
        return 0
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        return 1


def cmd_pipeline(args) -> int:
    """Execute complete pipeline."""
    from reba_3d.reba.risk_assessment import assess_video
    from reba_3d.visualization.annotator import annotate_with_reba

    # Resolve save_video from CLI args
    save_video = None
    if args.no_save_video:
        save_video = False
    elif args.save_video:
        save_video = True
    # Otherwise None = use config default

    output_dir = Path(args.output)

    print("=" * 60)
    print("ÉTAPE 1/3: Capture des keypoints 3D")
    print(f"Mode: {args.openpose_mode}")
    print("=" * 60)

    try:
        if args.openpose_mode == "v4l2":
            # Mode V4L2 streaming temps réel
            from reba_3d.capture.realsense_capture import process_bag_file_realtime

            capture_result = process_bag_file_realtime(
                bag_file=args.bag,
                output_dir=output_dir,
                v4l2_device=args.v4l2_device,
                json_dir=args.v4l2_json_dir,
                show_preview=not args.no_preview,
                save_video=save_video,
            )
        else:
            # Mode local (pyopenpose)
            from reba_3d.capture.realsense_capture import process_bag_file

            capture_result = process_bag_file(
                bag_file=args.bag,
                output_dir=output_dir,
                openpose_path=args.openpose,
                openpose_mode=args.openpose_mode,
                show_preview=not args.no_preview,
                save_video=save_video,
            )
    except Exception as e:
        print(f"[ERROR] Erreur lors de la capture: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("ÉTAPE 2/3: Analyse REBA")
    print("=" * 60)

    risk_times_path = output_dir / "risk_times.json"
    try:
        assess_video(
            keypoints_path=capture_result["keypoints_json"],
            output_path=str(risk_times_path)
        )
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'analyse: {e}")
        return 1

    # Step 3: Annotation (only if video was recorded)
    openpose_video = capture_result.get("openpose_video")
    if openpose_video:
        print("\n" + "=" * 60)
        print("ÉTAPE 3/3: Annotation vidéo")
        print("=" * 60)

        try:
            annotate_with_reba(
                input_video=openpose_video,
                risk_times=str(risk_times_path),
                show_preview=not args.no_preview
            )
        except Exception as e:
            print(f"[ERROR] Erreur lors de l'annotation: {e}")
            return 1
    else:
        print("\n" + "=" * 60)
        print("ÉTAPE 3/3: Annotation vidéo (ignorée - enregistrement désactivé)")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("[OK] Pipeline complet terminé!")
    print("=" * 60)
    print(f"\nFichiers générés dans {output_dir}:")
    if capture_result.get('raw_video'):
        print(f"  - output.avi (vidéo brute)")
    if capture_result.get('openpose_video'):
        print(f"  - output_openpose.avi (vidéo OpenPose)")
        print(f"  - output_openpose_annotated.avi (vidéo annotée REBA)")
    print(f"  - keypoints_3d.json (données 3D)")
    print(f"  - risk_times.json (intervalles de risque)")

    return 0


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        argv: Command line arguments (default: sys.argv)

    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    command_handlers = {
        "gui": cmd_gui,
        "capture": cmd_capture,
        "analyze": cmd_analyze,
        "annotate": cmd_annotate,
        "view": cmd_view,
        "pipeline": cmd_pipeline,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
