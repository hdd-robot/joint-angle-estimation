#!/usr/bin/env python3
"""
Script de test pour valider l'intégration de la calibration robust.

Ce script teste:
1. La conversion de format des données
2. Le calcul des offsets robusts
3. La compatibilité avec la méthode legacy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

def test_conversion_format():
    """Test de la conversion de format des données de calibration."""
    print("\n=== Test 1: Conversion de format ===")

    from reba_3d.core.angles import convert_calibration_data_to_robust_format

    # Données de test: 5 frames
    frames = [
        {
            "neck": {"alpha": 180.0, "beta": 3.5, "gamma": 0.0},
            "torso": {"alpha": 90.0, "beta": 2.8, "gamma": 2.8},
            "right_elbow": {"angle": 170.0}
        },
        {
            "neck": {"alpha": 180.2, "beta": 3.2, "gamma": 0.1},
            "torso": {"alpha": 90.1, "beta": 2.9, "gamma": 2.7},
            "right_elbow": {"angle": 170.5}
        },
        {
            "neck": {"alpha": 179.8, "beta": 3.6, "gamma": -0.1},
            "torso": {"alpha": 89.9, "beta": 2.7, "gamma": 2.9},
            "right_elbow": {"angle": 169.8}
        },
        {
            "neck": {"alpha": 180.1, "beta": 3.4, "gamma": 0.0},
            "torso": {"alpha": 90.0, "beta": 2.8, "gamma": 2.8},
            "right_elbow": {"angle": 170.2}
        },
        {
            "neck": {"alpha": 179.9, "beta": 3.5, "gamma": 0.0},
            "torso": {"alpha": 90.0, "beta": 2.8, "gamma": 2.8},
            "right_elbow": {"angle": 170.0}
        }
    ]

    # Conversion
    result = convert_calibration_data_to_robust_format(frames)

    # Vérifications
    assert "neck" in result, "Segment 'neck' manquant"
    assert "torso" in result, "Segment 'torso' manquant"
    assert "right_elbow" in result, "Segment 'right_elbow' manquant"

    assert result["neck"]["alpha"] == [180.0, 180.2, 179.8, 180.1, 179.9], \
        "Conversion alpha_neck incorrecte"

    assert len(result["neck"]["alpha"]) == 5, "Nombre de frames incorrect"

    print("✓ Conversion de format réussie")
    print(f"  - Segments: {list(result.keys())}")
    print(f"  - Exemple neck.alpha: {result['neck']['alpha']}")

    return True


def test_calibration_robust():
    """Test du calcul des offsets avec calibration robust."""
    print("\n=== Test 2: Calibration robust (MAD) ===")

    from reba_3d.core.angles import compute_calibration_offsets_robust

    # Créer 90 frames de données stables avec quelques outliers
    frames = []

    # 60 frames normales
    for i in range(60):
        frames.append({
            "neck": {
                "alpha": 180.0 + np.random.normal(0, 0.3),
                "beta": 3.5 + np.random.normal(0, 0.2),
                "gamma": 0.0 + np.random.normal(0, 0.1)
            },
            "torso": {
                "alpha": 90.0 + np.random.normal(0, 0.3),
                "beta": 2.8 + np.random.normal(0, 0.2),
                "gamma": 2.8 + np.random.normal(0, 0.2)
            },
            "right_elbow": {"angle": 170.0 + np.random.normal(0, 0.5)}
        })

    # 5 frames avec outliers (pour tester le filtrage MAD)
    for i in range(5):
        frames.append({
            "neck": {"alpha": 250.0, "beta": 20.0, "gamma": 50.0},  # Outliers évidents
            "torso": {"alpha": 150.0, "beta": 30.0, "gamma": 30.0},
            "right_elbow": {"angle": 90.0}
        })

    # 25 frames normales supplémentaires
    for i in range(25):
        frames.append({
            "neck": {
                "alpha": 180.0 + np.random.normal(0, 0.3),
                "beta": 3.5 + np.random.normal(0, 0.2),
                "gamma": 0.0 + np.random.normal(0, 0.1)
            },
            "torso": {
                "alpha": 90.0 + np.random.normal(0, 0.3),
                "beta": 2.8 + np.random.normal(0, 0.2),
                "gamma": 2.8 + np.random.normal(0, 0.2)
            },
            "right_elbow": {"angle": 170.0 + np.random.normal(0, 0.5)}
        })

    # Calcul des offsets robusts
    offsets = compute_calibration_offsets_robust(
        frames,
        n_neutre=60,
        k_mad=3.5
    )

    # Vérifications
    assert "neck" in offsets, "Segment 'neck' manquant dans les offsets"
    assert "torso" in offsets, "Segment 'torso' manquant"
    assert "right_elbow" in offsets, "Segment 'right_elbow' manquant"

    # Les offsets doivent être proches des valeurs attendues (malgré les outliers)
    assert abs(offsets["neck"]["alpha"] - 180.0) < 2.0, \
        f"Offset neck.alpha incorrect: {offsets['neck']['alpha']} (attendu ~180)"

    assert abs(offsets["neck"]["beta"] - 3.5) < 1.0, \
        f"Offset neck.beta incorrect: {offsets['neck']['beta']} (attendu ~3.5)"

    assert abs(offsets["torso"]["alpha"] - 90.0) < 2.0, \
        f"Offset torso.alpha incorrect: {offsets['torso']['alpha']} (attendu ~90)"

    print("✓ Calibration robust réussie")
    print(f"  - neck.alpha: {offsets['neck']['alpha']:.2f}° (attendu ~180°)")
    print(f"  - neck.beta: {offsets['neck']['beta']:.2f}° (attendu ~3.5°)")
    print(f"  - torso.alpha: {offsets['torso']['alpha']:.2f}° (attendu ~90°)")
    print(f"  - right_elbow.angle: {offsets['right_elbow']['angle']:.2f}° (attendu ~170°)")

    return True


def test_calibration_legacy():
    """Test de la méthode legacy pour comparaison."""
    print("\n=== Test 3: Calibration legacy (moyenne simple) ===")

    from reba_3d.core.angles import compute_calibration_offsets_nested

    # Données propres (sans outliers)
    frames = []
    for i in range(90):
        frames.append({
            "neck": {"alpha": 180.0, "beta": 3.5, "gamma": 0.0},
            "torso": {"alpha": 90.0, "beta": 2.8, "gamma": 2.8}
        })

    # Calcul avec méthode legacy
    offsets = compute_calibration_offsets_nested(
        frames,
        window_size=30,
        skip_windows=1
    )

    # Vérifications
    assert "neck" in offsets, "Segment 'neck' manquant"
    assert abs(offsets["neck"]["alpha"] - 180.0) < 0.1, "Offset incorrect"

    print("✓ Calibration legacy réussie")
    print(f"  - neck.alpha: {offsets['neck']['alpha']:.2f}°")
    print(f"  - neck.beta: {offsets['neck']['beta']:.2f}°")

    return True


def test_frames_insuffisantes():
    """Test de la gestion d'erreur avec frames insuffisantes."""
    print("\n=== Test 4: Gestion erreur (frames insuffisantes) ===")

    from reba_3d.core.angles import compute_calibration_offsets_robust

    # Seulement 20 frames (< 60 requis)
    frames = [{"neck": {"alpha": 180.0}} for _ in range(20)]

    try:
        offsets = compute_calibration_offsets_robust(frames, n_neutre=60)
        print("✗ Erreur: L'exception n'a pas été levée!")
        return False
    except ValueError as e:
        print(f"✓ Exception correctement levée: {str(e)[:60]}...")
        return True


def test_format_output():
    """Test que le format de sortie est compatible avec CalibrationManager."""
    print("\n=== Test 5: Format de sortie (compatibilité) ===")

    from reba_3d.core.angles import compute_calibration_offsets_robust

    frames = []
    for _ in range(60):
        frames.append({
            "neck": {"alpha": 180.0, "beta": 3.5, "gamma": 0.0},
            "right_shoulder": {"alpha": 0.0, "beta": 9.6, "gamma": 0.0, "elevation": 94.0},
            "right_elbow": {"angle": 170.5},
            "right_knee": {"angle": 178.0}
        })

    offsets = compute_calibration_offsets_robust(frames, n_neutre=60)

    # Vérifier la structure
    assert isinstance(offsets, dict), "Offsets doit être un dict"

    for segment, angles in offsets.items():
        assert isinstance(angles, dict), f"Angles de {segment} doit être un dict"
        for angle_name, value in angles.items():
            assert isinstance(value, (int, float)), \
                f"Valeur {segment}.{angle_name} doit être numérique"

    print("✓ Format de sortie valide")
    print(f"  - Structure: {list(offsets.keys())}")
    print(f"  - Exemple: neck = {offsets['neck']}")

    return True


def main():
    """Exécute tous les tests."""
    print("=" * 70)
    print("Tests d'intégration - Calibration robust MAD")
    print("=" * 70)

    tests = [
        ("Conversion de format", test_conversion_format),
        ("Calibration robust", test_calibration_robust),
        ("Calibration legacy", test_calibration_legacy),
        ("Gestion erreurs", test_frames_insuffisantes),
        ("Format de sortie", test_format_output),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' échoué avec exception:")
            print(f"  {type(e).__name__}: {e}")
            results.append((name, False))

    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")

    print(f"\nRésultat: {passed}/{total} tests réussis")

    if passed == total:
        print("\n🎉 Tous les tests sont passés avec succès!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
