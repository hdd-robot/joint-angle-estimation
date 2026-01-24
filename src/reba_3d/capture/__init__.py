"""RealSense capture and OpenPose integration modules."""

from reba_3d.capture.realsense_capture import (
    RealSenseCapture,
    OpenPoseDetector,
    create_openpose_detector,
    process_bag_file,
    process_bag_file_realtime,
)

from reba_3d.capture.v4l2_stream import (
    V4L2Writer,
    V4L2OpenPoseStreamer,
    DepthBuffer,
    OpenPoseJSONReader,
    check_v4l2_loopback,
)

__all__ = [
    # RealSense
    "RealSenseCapture",
    # OpenPose detectors
    "OpenPoseDetector",
    "create_openpose_detector",
    # Processing functions
    "process_bag_file",
    "process_bag_file_realtime",
    # V4L2 streaming
    "V4L2Writer",
    "V4L2OpenPoseStreamer",
    "DepthBuffer",
    "OpenPoseJSONReader",
    "check_v4l2_loopback",
]
