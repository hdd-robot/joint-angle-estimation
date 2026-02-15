"""
Calibration data for REBA angle calculations.

These values are used to offset measured angles to match a neutral standing position.
Data collected from calibration sessions with subjects in neutral poses.
"""

import numpy as np
from typing import Dict, List

# =============================================================================
# Neck calibration data
# =============================================================================

# Neck alpha angles - rotation around the main axis
ALPHA_NECK_CALI = [
    180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0,
    180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0
]

# Neck beta angles - flexion/extension
BETA_NECK_CALI = [
    4.508621709363438, 3.3930767171568235, 2.6814812643374673,
    4.456043124150283, 4.229149735226317, 3.5163121314974686,
    3.491473303239572, -8.161519008450918, 2.934576258863965,
    2.8542392964123264, 2.85052881879462, 3.529710169445216,
    4.072840527185665, 2.757050167264582
]

# Neck gamma angles - lateral rotation
GAMMA_NECK_CALI = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

# =============================================================================
# Torso calibration data
# =============================================================================

# Torso alpha angles - forward/backward inclination
ALPHA_TORSO_CALI = [
    90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0,
    90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0
]

# Torso beta angles - lateral rotation
BETA_TORSO_CALI = [
    2.846859891358757, 2.8808991309186203, 2.725109530518876,
    2.9526196305301218, 3.090760529430995, 2.8921685892574343,
    3.1738402885060792, 2.8438847867157757, 2.88019828603389,
    2.822287265716175, 2.763553166859605, 2.780685261495781,
    3.1038146758322425, 2.645724772250394
]

# Torso gamma angles - torsion
GAMMA_TORSO_CALI = [
    2.8468598913587577, 2.88089913091862, 2.7251095305188757,
    2.9526196305301213, 3.090760529430995, 2.8921685892574334,
    3.17384028850608, 2.8438847867157757, 2.88019828603389,
    2.8222872657161746, 2.763553166859605, 2.7806852614957807,
    3.103814675832243, 2.6457247722503943
]

# =============================================================================
# Right shoulder calibration data
# =============================================================================

# Right shoulder alpha angles
ALPHA_SHOULDER_CALI = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

# Right shoulder beta angles - flexion
BETA_SHOULDER_CALI = [
    10.998188079117796, 9.475642345191273, 9.625618585559465,
    8.526897770250807, 8.981957410601247, 8.48291238365517,
    10.202254827624845, 10.658278696151012, 11.092725181516155,
    10.5803798790983, 10.564418075356508, 8.500040653184849,
    8.676933256032067, 8.512812686436547
]

# Right shoulder gamma angles
GAMMA_SHOULDER_CALI = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

# Right shoulder elevation
ELEVATION_SHOULDER_R_CALI = [
    93.58379161558777, 95.0252697491975, 94.83105701621447,
    94.62498497839249, 94.02168920696008, 95.39609968738289,
    93.11199382672633, 87.82792407380136, 93.38634818661961,
    94.99020245990599, 94.59618497062412, 94.68584724239996,
    93.3043490771712, 95.94803957310249
]

# =============================================================================
# Right elbow calibration data
# =============================================================================

ANGLES_ELBOW_CALI = [
    167.73217163197967, 171.6137828982442, 171.32689856354887,
    173.43265398522735, 172.19747661357246, 173.03735452940197,
    169.45590113837332, 168.17705391244837, 167.09029672453505,
    169.16595599156284, 168.60936503067222, 173.58825534657,
    172.82861995504527, 173.98197923866994
]

# =============================================================================
# Right knee calibration data
# =============================================================================

ANGLES_KNEE_CALI = [
    177.62803692855246, 177.67513469953343, 177.7248403970671,
    178.48560859398125, 178.07966161918165, 178.32789838709408,
    178.18831337841982, 177.54387402790337, 178.15135206656046,
    177.91797236069712, 177.79869193035304, 178.33280326704391,
    178.26412834651688, 178.65529045039176
]

# =============================================================================
# Left shoulder calibration data
# =============================================================================

ALPHA_SHOULDER_L_CALI = [
    21.199278899198077, 24.79397906524404, 20.42657967863544,
    19.706006771245857, 21.07757135507793, 20.446157116198314,
    21.456843997711534, 21.159700465089998
]

BETA_SHOULDER_L_CALI = [
    -15.366265277178563, -15.724823761402192, -15.00120307192697,
    -14.404190560942007, -17.710943163697944, -14.972029451433269,
    -14.93924119967565, -14.96892383268446
]

GAMMA_SHOULDER_L_CALI = [
    174.18730138449277, 173.37921773997786, 174.53955618199305,
    174.72961384093688, 173.80656067193235, 174.5131787641376,
    174.216069228456, 174.33040322157063
]

ELEVATION_SHOULDER_L_CALI = [
    89.59152639780024, 88.14577286179517, 89.32086021928497,
    90.34092638432426, 89.23336772463217, 89.62460079227684,
    89.82393506890116, 89.71327320150371
]

# =============================================================================
# Left elbow calibration data
# =============================================================================

ANGLES_ELBOW_L_CALI = [
    164.10463514107644, 165.4562573558675, 162.26132407222337,
    165.47384940355752, 165.54097453142907, 167.65229393191018,
    162.39571362893295, 164.24626908595351, 164.10357375546872,
    164.32677635269687, 162.89547690147654, 165.68848665297836,
    164.67458932500426, 167.02622544093293
]

# =============================================================================
# Left knee (Genou gauche) calibration data
# =============================================================================

ANGLES_KNEE_L_CALI = [
    176.9519524466862, 177.87676391083656, 178.45578463345163,
    177.5225892658498, 176.46403663933225, 177.15476372074497,
    176.49900033114434, 174.84029381751006, 177.09711256525384,
    176.69591507564513, 177.92995972707283, 177.79453614891156,
    176.8218842958174, 176.93379279375205
]

# Elevation de l'épaule gauche (ancienne)
ELEVATION_SHOULDER_L_OLD_CALI = [
    87.88522213339326, 87.42421795645876, 86.27017673447027,
    87.98187772414201, 87.80053398096014, 87.87879636317786,
    87.76955429263893, 88.07306072189229, 87.92783483368353,
    87.10398395360802, 87.30325985697951, 87.08696741236572,
    88.4973730301612, 87.80048882763538
]


def mean_without_zeros(liste: List[float], skip_n: int = 1) -> float:
    """
    Calculate mean excluding NaN values and zeros.

    Args:
        liste: List of values to average
        skip_n: Number of initial elements to skip (default: 1)

    Returns:
        Mean value, or 0.0 if all values are filtered out
    """
    filtered_list = liste[skip_n:]
    filtered_values = [val for val in filtered_list if val != 0.0 and not np.isnan(val)]

    if filtered_values:
        return float(np.mean(filtered_values))
    return 0.0


def get_calibration_offsets() -> Dict[str, Dict[str, float]]:
    """
    Calculate all calibration offsets from stored calibration data.

    Returns:
        Dictionary containing offset values for each body segment
    """
    offsets = {
        "neck": {
            "alpha": mean_without_zeros(ALPHA_NECK_CALI),
            "beta": mean_without_zeros(BETA_NECK_CALI),
            "gamma": mean_without_zeros(GAMMA_NECK_CALI),
        },
        "torso": {
            "alpha": mean_without_zeros(ALPHA_TORSO_CALI),
            "beta": mean_without_zeros(BETA_TORSO_CALI),
            "gamma": mean_without_zeros(GAMMA_TORSO_CALI),
        },
        "right_shoulder": {
            "alpha": mean_without_zeros(ALPHA_SHOULDER_CALI),
            "beta": mean_without_zeros(BETA_SHOULDER_CALI),
            "gamma": mean_without_zeros(GAMMA_SHOULDER_CALI),
            "elevation": mean_without_zeros(ELEVATION_SHOULDER_R_CALI),
        },
        "right_elbow": {
            "angle": mean_without_zeros(ANGLES_ELBOW_CALI),
        },
        "right_knee": {
            "angle": mean_without_zeros(ANGLES_KNEE_CALI),
        },
        "left_shoulder": {
            "alpha": mean_without_zeros(ALPHA_SHOULDER_L_CALI),
            "beta": mean_without_zeros(BETA_SHOULDER_L_CALI),
            "gamma": mean_without_zeros(GAMMA_SHOULDER_L_CALI),
            "elevation": mean_without_zeros(ELEVATION_SHOULDER_L_CALI),
        },
        "left_elbow": {
            "angle": mean_without_zeros(ANGLES_ELBOW_L_CALI),
        },
        "left_knee": {
            "angle": mean_without_zeros(ANGLES_KNEE_L_CALI),
        },
    }
    return offsets


# Pre-computed offsets for faster access
CALIBRATION_OFFSETS = get_calibration_offsets()
