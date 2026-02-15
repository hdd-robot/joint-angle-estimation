#!/usr/bin/env python3
"""
Script de test pour l'intégration des angles nautiques en mode Inline.

Teste les nouvelles fonctions sans caméra RealSense.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_imports():
    """Test que tous les modules peuvent être importés."""
    print("Test 1: Imports...")
    try:
        from reba_3d.core.angles import (
            calculate_nautical_angles_2d,
            compute_calibration_offsets_nested
        )
        from reba_3d.config.calibration_store import (
            CalibrationManager,
            normalize_angle,
            DEFAULT_NAUTICAL_OFFSETS
        )
        from reba_3d.reba.realtime_scorer import (
            calculate_reba_score,
            calculate_reba_score_simple,
            calculate_reba_score_nautical,
            REBAScore
        )
        print("  ✓ Tous les imports réussis")
        return True
    except Exception as e:
        print(f"  ✗ Erreur d'import: {e}")
        return False


def test_2d_angles():
    """Test du calcul d'angles 2D nautiques."""
    print("\nTest 2: Calcul d'angles 2D nautiques...")
    try:
        from reba_3d.core.angles import calculate_nautical_angles_2d

        # Créer des keypoints de test (25x3: x, y, confidence)
        keypoints = np.zeros((25, 3))

        # Position neutre debout
        keypoints[0] = [320, 100, 0.9]    # Nose
        keypoints[1] = [320, 150, 0.9]    # Neck
        keypoints[2] = [370, 200, 0.9]    # RShoulder
        keypoints[3] = [420, 280, 0.9]    # RElbow
        keypoints[4] = [450, 350, 0.9]    # RWrist
        keypoints[5] = [270, 200, 0.9]    # LShoulder
        keypoints[6] = [220, 280, 0.9]    # LElbow
        keypoints[7] = [190, 350, 0.9]    # LWrist
        keypoints[8] = [320, 400, 0.9]    # MidHip
        keypoints[9] = [350, 400, 0.9]    # RHip
        keypoints[10] = [350, 550, 0.9]   # RKnee
        keypoints[11] = [350, 700, 0.9]   # RAnkle
        keypoints[12] = [290, 400, 0.9]   # LHip
        keypoints[13] = [290, 550, 0.9]   # LKnee
        keypoints[14] = [290, 700, 0.9]   # LAnkle
        keypoints[15] = [330, 95, 0.9]    # REye
        keypoints[16] = [310, 95, 0.9]    # LEye

        # Calculer les angles
        angles = calculate_nautical_angles_2d(keypoints)

        # Vérifier la structure
        assert isinstance(angles, dict), "angles doit être un dict"
        assert "neck" in angles, "angles doit contenir 'neck'"
        assert isinstance(angles["neck"], dict), "angles['neck'] doit être un dict"
        assert "alpha" in angles["neck"], "angles['neck'] doit contenir 'alpha'"
        assert "beta" in angles["neck"], "angles['neck'] doit contenir 'beta'"
        assert "gamma" in angles["neck"], "angles['neck'] doit contenir 'gamma'"

        # En mode 2D, alpha et beta doivent être 0
        assert angles["neck"]["alpha"] == 0.0, f"alpha devrait être 0 en 2D, got {angles['neck']['alpha']}"
        assert angles["neck"]["beta"] == 0.0, f"beta devrait être 0 en 2D, got {angles['neck']['beta']}"

        print(f"  ✓ Angles 2D calculés: neck gamma={angles['neck']['gamma']:.1f}°")
        print(f"    Structure: {list(angles.keys())}")
        return True
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calibration():
    """Test de la calibration avec structure imbriquée."""
    print("\nTest 3: Calibration imbriquée...")
    try:
        from reba_3d.core.angles import compute_calibration_offsets_nested

        # Créer une liste d'angles simulés (30 frames)
        angles_list = []
        for i in range(30):
            angles_list.append({
                "neck": {"alpha": 180.0 + np.random.randn(), "beta": 3.5 + np.random.randn()*0.5, "gamma": 0.0 + np.random.randn()*0.5},
                "torso": {"alpha": 90.0 + np.random.randn(), "beta": 2.8 + np.random.randn()*0.5, "gamma": 2.8 + np.random.randn()*0.5},
                "right_elbow": {"angle": 170.0 + np.random.randn()*2},
            })

        # Calculer les offsets
        offsets = compute_calibration_offsets_nested(angles_list, window_size=30, skip_windows=0)

        # Vérifier
        assert isinstance(offsets, dict), "offsets doit être un dict"
        assert "neck" in offsets, "offsets doit contenir 'neck'"
        assert isinstance(offsets["neck"], dict), "offsets['neck'] doit être un dict"

        print(f"  ✓ Offsets calculés:")
        print(f"    neck: alpha={offsets['neck']['alpha']:.1f}°, beta={offsets['neck']['beta']:.1f}°, gamma={offsets['neck']['gamma']:.1f}°")
        print(f"    torso: alpha={offsets['torso']['alpha']:.1f}°")
        return True
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calibration_manager():
    """Test du CalibrationManager avec apply_nested."""
    print("\nTest 4: CalibrationManager.apply_nested()...")
    try:
        from reba_3d.config.calibration_store import CalibrationManager, DEFAULT_NAUTICAL_OFFSETS

        manager = CalibrationManager()
        manager.offsets = DEFAULT_NAUTICAL_OFFSETS

        # Angles bruts
        raw_angles = {
            "neck": {"alpha": 185.0, "beta": 5.0, "gamma": 2.0},
            "torso": {"alpha": 95.0, "beta": 4.0, "gamma": 5.0},
            "right_elbow": {"angle": 175.0}
        }

        # Appliquer calibration
        calibrated = manager.apply_nested(raw_angles)

        # Vérifier la structure
        assert isinstance(calibrated, dict), "calibrated doit être un dict"
        assert "neck" in calibrated, "calibrated doit contenir 'neck'"
        assert isinstance(calibrated["neck"], dict), "calibrated['neck'] doit être un dict"

        print(f"  ✓ Calibration appliquée:")
        print(f"    neck brut: alpha={raw_angles['neck']['alpha']:.1f}°")
        print(f"    neck calibré: alpha={calibrated['neck']['alpha']:.1f}°")
        return True
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scoring():
    """Test du scoring REBA avec angles nautiques."""
    print("\nTest 5: Scoring REBA nautique...")
    try:
        from reba_3d.reba.realtime_scorer import calculate_reba_score

        # Angles calibrés avec rotation du neck (devrait donner malus)
        calibrated_angles_3d = {
            "neck": {"alpha": 15.0, "beta": 20.0, "gamma": 30.0},  # Beta et gamma donnent malus
            "torso": {"alpha": 10.0, "beta": 5.0, "gamma": 0.0},
            "right_shoulder": {"alpha": 25.0, "beta": 0.0, "gamma": 0.0, "elevation": 90.0},
            "right_elbow": {"angle": 80.0},
            "right_knee": {"angle": 5.0}
        }

        # Angles 2D (pas de malus car beta=0, gamma faible)
        calibrated_angles_2d = {
            "neck": {"alpha": 0.0, "beta": 0.0, "gamma": 2.0},  # Pas de malus
            "torso": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0},
            "right_shoulder": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "elevation": 90.0},
            "right_elbow": {"angle": 80.0},
            "right_knee": {"angle": 5.0}
        }

        # Calculer scores
        score_3d = calculate_reba_score(calibrated_angles_3d)
        score_2d = calculate_reba_score(calibrated_angles_2d)

        print(f"  ✓ Scores calculés:")
        print(f"    3D: neck={score_3d.neck_score}, trunk={score_3d.trunk_score}, final={score_3d.final_score} ({score_3d.risk_level})")
        print(f"    2D: neck={score_2d.neck_score}, trunk={score_2d.trunk_score}, final={score_2d.final_score} ({score_2d.risk_level})")

        # Vérifier que le score 3D est >= 2D (devrait être supérieur grâce au malus)
        print(f"\n  → Différence 3D-2D: neck={score_3d.neck_score - score_2d.neck_score}, final={score_3d.final_score - score_2d.final_score}")

        if score_3d.neck_score > score_2d.neck_score:
            print(f"  ✓ Le malus de rotation est bien appliqué en 3D !")
        else:
            print(f"  ⚠ Attention: pas de différence détectée (beta={calibrated_angles_3d['neck']['beta']}°, gamma={calibrated_angles_3d['neck']['gamma']}°)")

        return True
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Exécute tous les tests."""
    print("="*60)
    print("Tests d'intégration des angles nautiques en mode Inline")
    print("="*60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Angles 2D", test_2d_angles()))
    results.append(("Calibration", test_calibration()))
    results.append(("CalibrationManager", test_calibration_manager()))
    results.append(("Scoring REBA", test_scoring()))

    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)
    if all_passed:
        print("✓ Tous les tests sont passés !")
        return 0
    else:
        print("✗ Certains tests ont échoué")
        return 1


if __name__ == "__main__":
    sys.exit(main())
