#!/usr/bin/env python3
"""
RealSense Acquisition Test - RGB and Depth

This script verifies that the RealSense camera is working correctly
by capturing RGB and depth images.

Usage:
    python test_realsense.py
"""

import sys
import numpy as np

# Logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("test_realsense")


def test_realsense():
    """Test the RealSense camera."""

    logger.info("=" * 50)
    logger.info("  Intel RealSense Camera Test")
    logger.info("=" * 50)

    # Test import pyrealsense2
    logger.info("[1/5] Import pyrealsense2...")
    try:
        import pyrealsense2 as rs
        logger.info(f"      pyrealsense2 version: {rs.__version__}")
    except ImportError as e:
        logger.error(f"      Error: {e}")
        logger.error("      Install with: pip install pyrealsense2")
        return False

    # Test import OpenCV
    logger.info("[2/5] Import OpenCV...")
    try:
        import cv2
        logger.info(f"      OpenCV version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"      Error: {e}")
        return False

    # Detect connected cameras
    logger.info("[3/5] Detecting cameras...")
    try:
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            logger.warning("      No RealSense camera detected!")
            logger.warning("      Check that the camera is connected.")
            return False

        for i, dev in enumerate(devices):
            logger.info(f"      Camera {i+1}: {dev.get_info(rs.camera_info.name)}")
            logger.info(f"         Serial: {dev.get_info(rs.camera_info.serial_number)}")
            logger.info(f"         Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
    except Exception as e:
        logger.error(f"      Detection error: {e}")
        return False

    # Configure and start the pipeline
    logger.info("[4/5] Configuring pipeline...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable RGB and Depth streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start the pipeline
        profile = pipeline.start(config)
        logger.info("      Pipeline started")

        # Get intrinsics
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        logger.info(f"      Depth scale: {depth_scale:.6f} m/unit")

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

        color_intrinsics = color_profile.get_intrinsics()
        depth_intrinsics = depth_profile.get_intrinsics()

        logger.info(f"      RGB: {color_intrinsics.width}x{color_intrinsics.height}")
        logger.info(f"      Depth: {depth_intrinsics.width}x{depth_intrinsics.height}")

    except Exception as e:
        logger.error(f"      Configuration error: {e}")
        return False

    # Capture test
    logger.info("[5/5] Capture test (10 frames)...")
    try:
        align = rs.align(rs.stream.color)
        colorizer = rs.colorizer()

        frames_captured = 0
        depth_values = []

        for i in range(10):
            # Wait for frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                logger.warning(f"      Frame {i+1}: Missing frames")
                continue

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Depth statistics
            depth_m = depth_image * depth_scale
            valid_depth = depth_m[depth_m > 0]

            if len(valid_depth) > 0:
                mean_depth = np.mean(valid_depth)
                depth_values.append(mean_depth)

            frames_captured += 1

        logger.info(f"      {frames_captured}/10 frames captured")

        if depth_values:
            logger.info(f"      Mean depth: {np.mean(depth_values):.3f} m")
            logger.info(f"      Range: {np.min(depth_values):.3f} - {np.max(depth_values):.3f} m")

    except Exception as e:
        logger.error(f"      Capture error: {e}")
        pipeline.stop()
        return False

    # Stop the pipeline
    pipeline.stop()
    logger.info("=" * 50)
    logger.info("  Test successful - RealSense camera functional")
    logger.info("=" * 50)

    # Offer live preview
    response = input("Do you want to display a live preview? (y/n): ")

    if response.lower() in ['o', 'oui', 'y', 'yes']:
        show_live_preview()

    return True


def show_live_preview():
    """Display a live RGB + Depth preview."""
    import pyrealsense2 as rs
    import cv2

    logger.info("Live preview - Press 'q' to quit")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Images
            color_image = np.asanyarray(color_frame.get_data())
            depth_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # Center depth
            depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            depth_data = np.asanyarray(depth_frame.get_data())

            h, w = depth_data.shape
            center_depth = depth_data[h//2, w//2] * depth_scale

            # Display center depth on RGB image
            cv2.circle(color_image, (w//2, h//2), 5, (0, 255, 0), -1)
            cv2.putText(
                color_image,
                f"Depth: {center_depth:.3f} m",
                (w//2 + 10, h//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Combine images side by side
            combined = np.hstack((color_image, depth_colorized))

            cv2.imshow("RealSense - RGB | Depth (colorized)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def test_3d_point():
    """Test 2D -> 3D deprojection."""
    import pyrealsense2 as rs
    import cv2

    logger.info("2D -> 3D deprojection test")
    logger.info("Click on the image to get 3D coordinates")
    logger.info("Press 'q' to quit")

    # Variables for the callback
    click_point = [None]
    point_3d = [None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point[0] = (x, y)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    cv2.namedWindow("3D Point Test")
    cv2.setMouseCallback("3D Point Test", mouse_callback)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # If a point was clicked
            if click_point[0] is not None:
                x, y = click_point[0]
                depth = depth_frame.get_distance(x, y)

                if depth > 0:
                    point_3d[0] = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsics, [x, y], depth
                    )
                    logger.info(f"Pixel ({x}, {y}) -> 3D: X={point_3d[0][0]:.3f}m, Y={point_3d[0][1]:.3f}m, Z={point_3d[0][2]:.3f}m")
                else:
                    logger.warning(f"Pixel ({x}, {y}) -> No valid depth")
                    point_3d[0] = None

                click_point[0] = None

            # Display last 3D point
            if point_3d[0] is not None:
                cv2.putText(
                    color_image,
                    f"3D: ({point_3d[0][0]:.3f}, {point_3d[0][1]:.3f}, {point_3d[0][2]:.3f}) m",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                color_image,
                "Click to measure - 'q' to quit",
                (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow("3D Point Test", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    success = test_realsense()

    if success:
        response = input("Do you want to test 3D deprojection? (y/n): ")
        if response.lower() in ['o', 'oui', 'y', 'yes']:
            test_3d_point()

    sys.exit(0 if success else 1)
