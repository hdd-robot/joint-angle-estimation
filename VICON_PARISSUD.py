import json
import numpy as np
import pandas as pd

# ============================================================
# HARD-CODED SYNCHRO PARAMS (user inputs)
# ============================================================
SYNC_ENABLED = True

#VUE DE FACE "trial_RGBD0002.json"
# Intervalle RealSense (en secondes) correspondant au plateau coude (ex: 1 seconde)
RS_T0 = 38.697
RS_T1 = 39.697

# Plage de frames QTM correspondant au même plateau
QTM_F0 = 1308
QTM_F1 = 1408

#VUE DE PROFIL "trial_RGBD0003.json"
# RS_T0 = 36.027
# RS_T1 = 37.027
# QTM_F0 = 1500
# QTM_F1 = 1600

json_path = "trial_RGBD0002.json"
out_csv = "trial_RGBD0002_angles_sync.csv"



# QTM acquisition settings (hardcoded)
QTM_FREQ_HZ = 100.0
QTM_DURATION_S = 40.0
QTM_START_FRAME = 1  # souvent 1; mets la vraie valeur si besoin

# ============================================================
# 0) Helpers for QTM JSON variants
# ============================================================
def get_part_start_end(part):
    if "Range" in part and isinstance(part["Range"], dict):
        return int(part["Range"]["Start"]), int(part["Range"]["End"])
    if "Start" in part and "End" in part:
        return int(part["Start"]), int(part["End"])
    raise KeyError(f"Start/End not found in Part. Keys: {list(part.keys())}")

# ============================================================
# 1) Load QTM JSON -> DataFrame (frame x marker coords)
# ============================================================
def load_qtm_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    tb = data["Timebase"]
    freq = float(tb["Frequency"])
    start = int(tb["Range"]["Start"])
    end = int(tb["Range"]["End"])

    frames = np.arange(start, end + 1)
    n = len(frames)

    df = pd.DataFrame({"_Frame": frames, "time_s": (frames - start) / freq})

    for m in data["Markers"]:
        name = m["Name"]
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        z = np.full(n, np.nan)
        r = np.full(n, np.nan)

        for part in m.get("Parts", []):
            ps, pe = get_part_start_end(part)
            vals = np.asarray(part["Values"], dtype=float)

            ps_clip = max(ps, start)
            pe_clip = min(pe, end)
            if pe_clip < ps_clip:
                continue

            i0 = ps_clip - start
            i1 = pe_clip - start + 1

            v0 = ps_clip - ps
            v1 = v0 + (i1 - i0)

            x[i0:i1] = vals[v0:v1, 0]
            y[i0:i1] = vals[v0:v1, 1]
            z[i0:i1] = vals[v0:v1, 2]
            if vals.shape[1] >= 4:
                r[i0:i1] = vals[v0:v1, 3]

        df[f"{name}_X"] = x
        df[f"{name}_Y"] = y
        df[f"{name}_Z"] = z
        df[f"{name}_res"] = r

    return df

def get_marker(df, name):
    cols = [f"{name}_X", f"{name}_Y", f"{name}_Z"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Marker {name} missing columns: {missing}")
    return df[cols].to_numpy()

# ============================================================
# 2) Vector utilities (row-wise)
# ============================================================
def row_norm(v, eps=1e-9):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (n + eps)

def orthonormalize_frame(v1, v2, eps=1e-9):
    Z = row_norm(v1, eps)
    dot = np.sum(v2 * Z, axis=1, keepdims=True)
    v2p = v2 - dot * Z
    X = row_norm(v2p, eps)

    flip = (np.sum(X * v2, axis=1) < 0)
    X[flip] *= -1.0

    Y = np.cross(Z, X)
    Y = row_norm(Y, eps)
    return X, Y, Z

def rotation_from_frames(Xt, Yt, Zt, Xb, Yb, Zb):
    r00 = np.sum(Xt*Xb, axis=1); r01 = np.sum(Xt*Yb, axis=1); r02 = np.sum(Xt*Zb, axis=1)
    r10 = np.sum(Yt*Xb, axis=1); r11 = np.sum(Yt*Yb, axis=1); r12 = np.sum(Yt*Zb, axis=1)
    r20 = np.sum(Zt*Xb, axis=1); r21 = np.sum(Zt*Yb, axis=1); r22 = np.sum(Zt*Zb, axis=1)

    R = np.stack([
        np.stack([r00, r01, r02], axis=1),
        np.stack([r10, r11, r12], axis=1),
        np.stack([r20, r21, r22], axis=1),
    ], axis=1)
    return R

def extract_zyx_nautical(R, eps=1e-6):
    r20 = R[:, 2, 0]
    beta = -np.arcsin(np.clip(r20, -1.0, 1.0))
    cosb = np.cos(beta)

    gimbal = np.abs(cosb) < eps
    cosb_safe = np.where(gimbal, np.nan, cosb)

    alpha = np.arctan2(R[:, 2, 1] / cosb_safe, R[:, 2, 2] / cosb_safe)
    gamma = np.arctan2(R[:, 1, 0] / cosb_safe, R[:, 0, 0] / cosb_safe)

    return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)

# ============================================================
# 3) Angles computation
# ============================================================
def compute_angles_qtm(df, vertebra_low_name="TV7"):
    RAH = get_marker(df, "RAH"); LAH = get_marker(df, "LAH")
    RPH = get_marker(df, "RPH"); LPH = get_marker(df, "LPH")

    RCAJ = get_marker(df, "RCAJ"); LCAJ = get_marker(df, "LCAJ")
    SJN  = get_marker(df, "SJN");  SXS  = get_marker(df, "SXS")
    CV7  = get_marker(df, "CV7")
    VLOW = get_marker(df, vertebra_low_name)

    RHLE = get_marker(df, "RHLE"); RHME = get_marker(df, "RHME")
    RRSP = get_marker(df, "RRSP"); RUSP = get_marker(df, "RUSP")

    A = 0.5*(RAH + LAH)
    P = 0.5*(RPH + LPH)

    E = 0.5*(RHLE + RHME)
    W = 0.5*(RRSP + RUSP)
    S = RCAJ

    T_high = 0.5*(SJN + CV7)
    T_low  = 0.5*(SXS + VLOW)

    v1_torso = T_high - T_low
    v2_torso = RCAJ - LCAJ
    Xt, Yt, Zt = orthonormalize_frame(v1_torso, v2_torso)

    n = len(df)
    Xg = np.tile(np.array([[1.0, 0.0, 0.0]]), (n, 1))
    Yg = np.tile(np.array([[0.0, 1.0, 0.0]]), (n, 1))
    Zg = np.tile(np.array([[0.0, 0.0, 1.0]]), (n, 1))

    R_torso = rotation_from_frames(Xt, Yt, Zt, Xg, Yg, Zg)
    torso_alpha, torso_beta, torso_gamma = extract_zyx_nautical(R_torso)

    v1_head = P - A
    v2_head = RAH - LAH
    Xh, Yh, Zh = orthonormalize_frame(v1_head, v2_head)

    R_neck = rotation_from_frames(Xh, Yh, Zh, Xt, Yt, Zt)
    neck_alpha, neck_beta, neck_gamma = extract_zyx_nautical(R_neck)

    v1_arm = S - E
    v2_arm = RHLE - RHME
    Xa, Ya, Za = orthonormalize_frame(v1_arm, v2_arm)

    R_sh = rotation_from_frames(Xa, Ya, Za, Xt, Yt, Zt)
    sh_alpha, sh_beta, sh_gamma = extract_zyx_nautical(R_sh)

    v_forearm  = W - E
    v_upperarm = S - E
    u1 = row_norm(v_forearm)
    u2 = row_norm(v_upperarm)
    cosang = np.clip(np.sum(u1*u2, axis=1), -1.0, 1.0)
    elbow_deg = np.degrees(np.arccos(cosang))

    out = df.copy()
    out["torso_alpha_deg"] = torso_alpha
    out["torso_beta_deg"] = torso_beta
    out["torso_gamma_deg"] = torso_gamma

    out["neck_alpha_deg"] = np.abs(neck_alpha)
    out["neck_beta_deg"]  = neck_beta
    out["neck_gamma_deg"] = neck_gamma

    out["shoulderR_alpha_deg"] = sh_alpha
    out["shoulderR_beta_deg"]  = sh_beta
    out["shoulderR_gamma_deg"] = sh_gamma

    out["elbowR_deg"] = elbow_deg
    return out

# ============================================================
# 4) Synchronisation (QTM frame -> RealSense time)
# ============================================================
def estimate_sync_params(rs_t0, rs_t1, qtm_f0, qtm_f1, qtm_freq_hz, qtm_start_frame):
    """
    Fit: t_rs = a * t_qtm + b using two points (start & end of plateau).
    """
    t_qtm0 = (qtm_f0 - qtm_start_frame) / qtm_freq_hz
    t_qtm1 = (qtm_f1 - qtm_start_frame) / qtm_freq_hz

    dt_rs = rs_t1 - rs_t0
    dt_qtm = t_qtm1 - t_qtm0

    if abs(dt_qtm) < 1e-9:
        raise ValueError("dt_qtm is zero (check qtm_f0/qtm_f1).")

    a = dt_rs / dt_qtm
    b = rs_t0 - a * t_qtm0
    return a, b

# ============================================================
# 5) Run
# ============================================================
df = load_qtm_json(json_path)

angles_df = compute_angles_qtm(df, vertebra_low_name="TV7")

# (Optionnel mais utile) écrase avec les hardcodes si tu veux forcer
# sinon tu peux prendre ceux du JSON:
if SYNC_ENABLED:
    # Tu peux aussi faire: QTM_START_FRAME = int(angles_df["_Frame"].min())
    a, b = estimate_sync_params(
        rs_t0=RS_T0, rs_t1=RS_T1,
        qtm_f0=QTM_F0, qtm_f1=QTM_F1,
        qtm_freq_hz=QTM_FREQ_HZ,
        qtm_start_frame=QTM_START_FRAME
    )

    # temps QTM (selon hardcode)
    angles_df["time_qtm_s"] = (angles_df["_Frame"] - QTM_START_FRAME) / QTM_FREQ_HZ

    # temps RealSense estimé correspondant à chaque frame QTM
    angles_df["time_rs_s"] = a * angles_df["time_qtm_s"] + b

    print(f"✅ Sync fitted: t_rs = a*t_qtm + b  with a={a:.9f}, b={b:.6f}s")

# --- aperçu
print(angles_df[["_Frame",
                 "torso_alpha_deg","torso_beta_deg","torso_gamma_deg",
                 "neck_alpha_deg","neck_beta_deg","neck_gamma_deg",
                 "shoulderR_alpha_deg","shoulderR_beta_deg","shoulderR_gamma_deg",
                 "elbowR_deg"]].head(10))

# ============================================================
# 6) Save to CSV (with synced time)
# ============================================================
cols_to_save = ["_Frame"]

if SYNC_ENABLED:
    cols_to_save += ["time_qtm_s", "time_rs_s"]

cols_to_save += [
    "torso_alpha_deg","torso_beta_deg","torso_gamma_deg",
    "neck_alpha_deg","neck_beta_deg","neck_gamma_deg",
    "shoulderR_alpha_deg","shoulderR_beta_deg","shoulderR_gamma_deg",
    "elbowR_deg"
]

angles_df[cols_to_save].to_csv(out_csv, index=False)
print(f"✅ CSV sauvegardé : {out_csv}")
