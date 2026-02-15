#!/usr/bin/env python3
"""
Script de test pour vérifier les modifications des calculs d'angles.
"""

import numpy as np
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from reba_3d.core.geometry import (
    orthonormalize_frame,
    extract_nautical_angles,
    extract_nautical_angles_2d,
)
from reba_3d.reba.angles import (
    compute_neck_angles,
    compute_neck_angles_2d,
    compute_torso_angles,
    compute_torso_angles_2d,
    compute_shoulder_angles_right,
    compute_shoulder_angles_right_2d,
    compute_shoulder_angles_left,
    compute_shoulder_angles_left_2d,
)


def test_orthonormalize_frame():
    """Test de la fonction orthonormalize_frame avec cohérence du signe."""
    print("\n=== Test orthonormalize_frame ===")

    # Test 1: Vecteurs simples
    v1 = np.array([0, 0, 1])  # Z vers le haut
    v2 = np.array([1, 0, 0])  # X vers la droite

    result = orthonormalize_frame(v1, v2)
    if result is not None:
        X, Y, Z = result
        print(f"✓ Test 1 réussi:")
        print(f"  Z = {Z}")
        print(f"  X = {X}")
        print(f"  Y = {Y}")
        print(f"  X·Y = {np.dot(X, Y):.10f} (devrait être ~0)")
        print(f"  Y·Z = {np.dot(Y, Z):.10f} (devrait être ~0)")
        print(f"  Z·X = {np.dot(Z, X):.10f} (devrait être ~0)")

        # Vérifier la cohérence du signe
        print(f"  X·v2 = {np.dot(X, v2):.6f} (devrait être > 0)")
    else:
        print("✗ Test 1 échoué: result is None")

    # Test 2: Vecteurs qui nécessitent la correction de signe
    v1 = np.array([1, 1, 0])
    v2 = np.array([-1, 1, 0])

    result = orthonormalize_frame(v1, v2)
    if result is not None:
        X, Y, Z = result
        print(f"\n✓ Test 2 réussi:")
        print(f"  X·v2 = {np.dot(X, v2):.6f} (devrait être > 0 après correction)")
    else:
        print("\n✗ Test 2 échoué: result is None")


def test_extract_nautical_angles():
    """Test de la fonction extract_nautical_angles avec correction gamma."""
    print("\n=== Test extract_nautical_angles ===")

    # Repère cible (rotation autour de Z de 45°)
    angle_rot = np.radians(45)
    X_t = np.array([np.cos(angle_rot), np.sin(angle_rot), 0])
    Y_t = np.array([-np.sin(angle_rot), np.cos(angle_rot), 0])
    Z_t = np.array([0, 0, 1])

    # Repère de base (identité)
    X_b = np.array([1, 0, 0])
    Y_b = np.array([0, 1, 0])
    Z_b = np.array([0, 0, 1])

    result = extract_nautical_angles(X_t, Y_t, Z_t, X_b, Y_b, Z_b)
    if result is not None:
        alpha, beta, gamma = result
        print(f"✓ Rotation de 45° autour de Z:")
        print(f"  alpha = {alpha:.2f}° (devrait être ~0)")
        print(f"  beta  = {beta:.2f}° (devrait être ~0)")
        print(f"  gamma = {gamma:.2f}° (devrait être ~45)")
    else:
        print("✗ Test échoué: result is None")


def test_extract_nautical_angles_2d():
    """Test de la nouvelle fonction extract_nautical_angles_2d."""
    print("\n=== Test extract_nautical_angles_2d ===")

    # Vecteur de base
    X_b = np.array([1, 0, 0])

    # Test 1: Rotation de 45° dans le plan XY
    angle = np.radians(45)
    X_t = np.array([np.cos(angle), np.sin(angle), 0])

    result = extract_nautical_angles_2d(X_t, X_b)
    if result is not None:
        alpha, beta, gamma = result
        print(f"✓ Test 1 - Rotation de 45°:")
        print(f"  alpha = {alpha:.2f}° (devrait être 0)")
        print(f"  beta  = {beta:.2f}° (devrait être 0)")
        print(f"  gamma = {gamma:.2f}° (devrait être ~45)")
    else:
        print("✗ Test 1 échoué: result is None")

    # Test 2: Rotation de -30°
    angle = np.radians(-30)
    X_t = np.array([np.cos(angle), np.sin(angle), 0])

    result = extract_nautical_angles_2d(X_t, X_b)
    if result is not None:
        alpha, beta, gamma = result
        print(f"\n✓ Test 2 - Rotation de -30°:")
        print(f"  gamma = {gamma:.2f}° (devrait être ~-30)")
    else:
        print("\n✗ Test 2 échoué: result is None")


def test_compute_angles_with_use_3d():
    """Test des fonctions compute_* avec le paramètre use_3d."""
    print("\n=== Test fonctions compute_* avec use_3d ===")

    # Créer des positions de test
    positions = {
        "Nose": np.array([0.0, 0.0, 1.7]),
        "Neck": np.array([0.0, 0.0, 1.5]),
        "REye": np.array([0.05, 0.0, 1.72]),
        "LEye": np.array([-0.05, 0.0, 1.72]),
        "MidHip": np.array([0.0, 0.0, 1.0]),
        "RShoulder": np.array([0.2, 0.0, 1.45]),
        "LShoulder": np.array([-0.2, 0.0, 1.45]),
        "RElbow": np.array([0.3, 0.0, 1.2]),
        "LElbow": np.array([-0.3, 0.0, 1.2]),
    }

    # Test compute_neck_angles en mode 3D
    neck_3d = compute_neck_angles(positions)
    if neck_3d is not None:
        print(f"✓ compute_neck_angles (3D): alpha={neck_3d[0]:.2f}°, beta={neck_3d[1]:.2f}°, gamma={neck_3d[2]:.2f}°")
    else:
        print("✗ compute_neck_angles (3D) échoué")

    # Test compute_neck_angles_2d en mode 2D
    neck_2d = compute_neck_angles_2d(positions)
    if neck_2d is not None:
        print(f"✓ compute_neck_angles_2d (2D): alpha={neck_2d[0]:.2f}°, beta={neck_2d[1]:.2f}°, gamma={neck_2d[2]:.2f}°")
        print(f"  (alpha et beta devraient être 0 en mode 2D)")
    else:
        print("✗ compute_neck_angles_2d (2D) échoué")

    # Test compute_torso_angles en mode 3D
    torso_3d = compute_torso_angles(positions)
    if torso_3d is not None:
        print(f"\n✓ compute_torso_angles (3D): alpha={torso_3d[0]:.2f}°, beta={torso_3d[1]:.2f}°, gamma={torso_3d[2]:.2f}°")
    else:
        print("\n✗ compute_torso_angles (3D) échoué")

    # Test compute_torso_angles_2d en mode 2D
    torso_2d = compute_torso_angles_2d(positions)
    if torso_2d is not None:
        print(f"✓ compute_torso_angles_2d (2D): alpha={torso_2d[0]:.2f}°, beta={torso_2d[1]:.2f}°, gamma={torso_2d[2]:.2f}°")
        print(f"  (alpha et beta devraient être 0 en mode 2D)")
    else:
        print("✗ compute_torso_angles_2d (2D) échoué")

    # Test compute_shoulder_angles_right
    shoulder_3d = compute_shoulder_angles_right(positions)
    if shoulder_3d is not None:
        print(f"\n✓ compute_shoulder_angles_right (3D): alpha={shoulder_3d[0]:.2f}°, beta={shoulder_3d[1]:.2f}°, gamma={shoulder_3d[2]:.2f}°")
    else:
        print("\n✗ compute_shoulder_angles_right (3D) échoué")

    shoulder_2d = compute_shoulder_angles_right_2d(positions)
    if shoulder_2d is not None:
        print(f"✓ compute_shoulder_angles_right_2d (2D): alpha={shoulder_2d[0]:.2f}°, beta={shoulder_2d[1]:.2f}°, gamma={shoulder_2d[2]:.2f}°")
    else:
        print("✗ compute_shoulder_angles_right_2d (2D) échoué")


def main():
    """Exécute tous les tests."""
    print("="*60)
    print("Test des modifications du calcul d'angles")
    print("="*60)

    try:
        test_orthonormalize_frame()
        test_extract_nautical_angles()
        test_extract_nautical_angles_2d()
        test_compute_angles_with_use_3d()

        print("\n" + "="*60)
        print("✓ Tous les tests sont terminés")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
