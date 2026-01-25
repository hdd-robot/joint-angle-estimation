import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

# Activer explicitement le flux couleur
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

try:
    # Intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    print(f"Width: {intrinsics.width}")
    print(f"Height: {intrinsics.height}")
    print(f"fx: {intrinsics.fx}")
    print(f"fy: {intrinsics.fy}")
    print(f"cx (ppx): {intrinsics.ppx}")
    print(f"cy (ppy): {intrinsics.ppy}")
    print(f"Distortion: {intrinsics.coeffs}")

    # Depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    print(f"Depth scale: {depth_sensor.get_depth_scale()}")

finally:
    pipeline.stop()
