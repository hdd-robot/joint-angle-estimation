#!/usr/bin/env python3
"""
Test d'acquisition RealSense - RGB et Profondeur

Ce script vérifie que la caméra RealSense fonctionne correctement
en capturant des images RGB et de profondeur.

Usage:
    python test_realsense.py
"""

import sys
import numpy as np

# Configuration du logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("test_realsense")


def test_realsense():
    """Test de la caméra RealSense."""

    logger.info("=" * 50)
    logger.info("  Test Caméra Intel RealSense")
    logger.info("=" * 50)

    # Test import pyrealsense2
    logger.info("[1/5] Import pyrealsense2...")
    try:
        import pyrealsense2 as rs
        logger.info(f"      pyrealsense2 version: {rs.__version__}")
    except ImportError as e:
        logger.error(f"      Erreur: {e}")
        logger.error("      Installez avec: pip install pyrealsense2")
        return False

    # Test import OpenCV
    logger.info("[2/5] Import OpenCV...")
    try:
        import cv2
        logger.info(f"      OpenCV version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"      Erreur: {e}")
        return False

    # Détecter les caméras connectées
    logger.info("[3/5] Détection des caméras...")
    try:
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            logger.warning("      Aucune caméra RealSense détectée!")
            logger.warning("      Vérifiez que la caméra est branchée.")
            return False

        for i, dev in enumerate(devices):
            logger.info(f"      Caméra {i+1}: {dev.get_info(rs.camera_info.name)}")
            logger.info(f"         Série: {dev.get_info(rs.camera_info.serial_number)}")
            logger.info(f"         Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
    except Exception as e:
        logger.error(f"      Erreur détection: {e}")
        return False

    # Configuration et démarrage du pipeline
    logger.info("[4/5] Configuration du pipeline...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()

        # Activer les flux RGB et Depth
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Démarrer le pipeline
        profile = pipeline.start(config)
        logger.info("      Pipeline démarré")

        # Obtenir les intrinsèques
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
        logger.error(f"      Erreur configuration: {e}")
        return False

    # Test de capture
    logger.info("[5/5] Test de capture (10 frames)...")
    try:
        align = rs.align(rs.stream.color)
        colorizer = rs.colorizer()

        frames_captured = 0
        depth_values = []

        for i in range(10):
            # Attendre les frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                logger.warning(f"      Frame {i+1}: Frames manquantes")
                continue

            # Convertir en numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Statistiques de profondeur
            depth_m = depth_image * depth_scale
            valid_depth = depth_m[depth_m > 0]

            if len(valid_depth) > 0:
                mean_depth = np.mean(valid_depth)
                depth_values.append(mean_depth)

            frames_captured += 1

        logger.info(f"      {frames_captured}/10 frames capturées")

        if depth_values:
            logger.info(f"      Profondeur moyenne: {np.mean(depth_values):.3f} m")
            logger.info(f"      Plage: {np.min(depth_values):.3f} - {np.max(depth_values):.3f} m")

    except Exception as e:
        logger.error(f"      Erreur capture: {e}")
        pipeline.stop()
        return False

    # Arrêter le pipeline
    pipeline.stop()
    logger.info("=" * 50)
    logger.info("  Test réussi - Caméra RealSense fonctionnelle")
    logger.info("=" * 50)

    # Proposer un affichage live
    response = input("Voulez-vous afficher un aperçu live? (o/n): ")

    if response.lower() in ['o', 'oui', 'y', 'yes']:
        show_live_preview()

    return True


def show_live_preview():
    """Affiche un aperçu live RGB + Depth."""
    import pyrealsense2 as rs
    import cv2

    logger.info("Aperçu live - Appuyez sur 'q' pour quitter")

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

            # Profondeur au centre
            depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            depth_data = np.asanyarray(depth_frame.get_data())

            h, w = depth_data.shape
            center_depth = depth_data[h//2, w//2] * depth_scale

            # Afficher la profondeur au centre sur l'image RGB
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

            # Combiner les images côte à côte
            combined = np.hstack((color_image, depth_colorized))

            cv2.imshow("RealSense - RGB | Depth (colorized)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def test_3d_point():
    """Test de déprojection 2D -> 3D."""
    import pyrealsense2 as rs
    import cv2

    logger.info("Test de déprojection 2D -> 3D")
    logger.info("Cliquez sur l'image pour obtenir les coordonnées 3D")
    logger.info("Appuyez sur 'q' pour quitter")

    # Variables pour le callback
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

            # Si un point a été cliqué
            if click_point[0] is not None:
                x, y = click_point[0]
                depth = depth_frame.get_distance(x, y)

                if depth > 0:
                    point_3d[0] = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsics, [x, y], depth
                    )
                    logger.info(f"Pixel ({x}, {y}) -> 3D: X={point_3d[0][0]:.3f}m, Y={point_3d[0][1]:.3f}m, Z={point_3d[0][2]:.3f}m")
                else:
                    logger.warning(f"Pixel ({x}, {y}) -> Pas de profondeur valide")
                    point_3d[0] = None

                click_point[0] = None

            # Afficher le dernier point 3D
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
                "Cliquez pour mesurer - 'q' pour quitter",
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
        response = input("Voulez-vous tester la déprojection 3D? (o/n): ")
        if response.lower() in ['o', 'oui', 'y', 'yes']:
            test_3d_point()

    sys.exit(0 if success else 1)
