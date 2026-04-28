import os as _os

import pandas as pd
import matplotlib
matplotlib.use("Agg")

# ── Global font sizes — référence pour une figure de REF_W pouces de large ─
REF_W     = 8    # largeur de référence (pouces) pour le scaling
FS_TITLE  = 24   # titres
FS_LABEL  = 20   # axes x/y
FS_TICK   = 18   # graduations
FS_ANNOT  = 18   # chiffres dans les cases
FS_LEGEND = 18   # légendes
FS_TEXT   = 18   # texte inline (barres, etc.)

def fs(base, fig_w):
    """Scale base fontsize proportionally to the actual figure width."""
    return max(8, int(round(base * fig_w / REF_W)))
# ──────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.size":        FS_TICK,
    "axes.titlesize":   FS_TITLE,
    "axes.labelsize":   FS_LABEL,
    "xtick.labelsize":  FS_TICK,
    "ytick.labelsize":  FS_TICK,
    "legend.fontsize":  FS_LEGEND,
})
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

_HERE = _os.path.dirname(_os.path.abspath(__file__))
OUTPUT_DIR = _os.path.join(_HERE, "output")
_os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv(_os.path.join(_HERE, "ULTIMATE_ALL_DATA_EXPERIMENT.csv"))

# Forcer les colonnes numériques (si jamais "invalid" ou texte traîne)
for c in ["REBA_2D", "REBA_3D", "Error"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # invalid -> NaN

# ---------------------------
# Créer les labels REBA (alignés avec CODE 1)
# ---------------------------
risk_inverse = {
    0: "low/negligible risk",
    1: "medium risk",
    2: "high risk",
    3: "very high risk"
}

def reba_to_label(x):
    if pd.isna(x):
        return "invalid"
    try:
        return risk_inverse.get(int(x), "invalid")
    except Exception:
        return "invalid"

if "REBA_2D" in df.columns:
    df["REBA_2D_label"] = df["REBA_2D"].apply(reba_to_label)
else:
    df["REBA_2D_label"] = "invalid"

if "REBA_3D" in df.columns:
    df["REBA_3D_label"] = df["REBA_3D"].apply(reba_to_label)
else:
    df["REBA_3D_label"] = "invalid"

# ---------------------------
# Colormap Error (NaN -> gris)
# ---------------------------
vmin, vmax = -3, 3

def scale(val):
    return (val - vmin) / (vmax - vmin)

colors = [
    (scale(-3), "#08306B"),  # dark blue
    (scale(-2), "#2171B5"),  # medium blue
    (scale(-1), "#6BAED6"),  # light blue
    (scale(0),  "#41AB5D"),  # green
    (scale(1),  "#FC9272"),  # light red
    (scale(2),  "#DE2D26"),  # medium red
    (scale(3),  "#A50F15")   # dark red
]
custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
custom_cmap.set_bad(color="lightgrey")  # NaN en gris

# ---------------------------
# HEATMAP Error (robuste à doublons)
# ---------------------------
if {"Person_ID", "Window_ID", "Error"}.issubset(df.columns):
    # pivot_table pour éviter crash si doublons Person_ID/Window_ID
    heatmap_data = df.pivot_table(
        index="Person_ID",
        columns="Window_ID",
        values="Error",
        aggfunc="mean"
    )

    mask = heatmap_data.isna()
    # Annotations en entier (pas de décimale)
    annot = heatmap_data.applymap(lambda v: str(int(v)) if pd.notna(v) else "")

    n_rows, n_cols = heatmap_data.shape
    fig_w = max(18, 0.55 * n_cols)
    fig_h = max(8, 0.7 * n_rows)

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        heatmap_data,
        cmap=custom_cmap,
        center=0,
        vmin=vmin, vmax=vmax,
        mask=mask,
        annot=annot,
        fmt="",
        annot_kws={"size": fs(FS_ANNOT, fig_w / 2), "weight": "bold"},
        linewidths=1,
        linecolor="white",
        cbar=False,
    )
    plt.title("Difference REBA 2D vs 3D by Person and Window", fontsize=fs(FS_TITLE, fig_w / 2), weight="bold", pad=12)
    plt.xlabel("Window", fontsize=fs(FS_LABEL, fig_w / 2))
    plt.ylabel("Person ID", fontsize=fs(FS_LABEL, fig_w / 2))
    plt.xticks(fontsize=fs(FS_TICK, fig_w / 2))
    plt.yticks(fontsize=fs(FS_TICK, fig_w / 2), rotation=0)
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUT_DIR, "heatmap_error.png"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("[WARN] Heatmap: colonnes Person_ID / Window_ID / Error manquantes.")

# ---------------------------
# HISTOGRAM Error (ignore NaN)
# ---------------------------
if "Error" in df.columns:
    error_vals = df["Error"].dropna()
    unique_vals = sorted(error_vals.unique())
    counts = [int((error_vals == v).sum()) for v in unique_vals]
    labels = [str(int(v)) for v in unique_vals]

    _fw = 7
    plt.figure(figsize=(_fw, 7))
    bars = plt.bar(labels, counts, width=0.6, color="#2171B5", edgecolor="black", linewidth=1)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 str(count), ha="center", va="bottom", fontsize=fs(FS_TEXT, _fw), weight="bold")
    plt.title("Global Histogram of Errors (REBA 2D - 3D)", fontsize=fs(FS_TITLE, _fw), weight="bold", pad=12)
    plt.xlabel("Error", fontsize=fs(FS_LABEL, _fw))
    plt.ylabel("Frequency", fontsize=fs(FS_LABEL, _fw))
    plt.xticks(fontsize=fs(FS_TICK, _fw))
    plt.yticks(fontsize=fs(FS_TICK, _fw))
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUT_DIR, "histogram_error.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------
# CONFUSION MATRIX (sans invalid)
# ---------------------------
risk_order_valid = ["low/negligible risk", "medium risk", "high risk", "very high risk"]

df_valid = df[(df["REBA_2D_label"] != "invalid") & (df["REBA_3D_label"] != "invalid")].copy()

df_valid["REBA_2D_label"] = pd.Categorical(df_valid["REBA_2D_label"], categories=risk_order_valid, ordered=True)
df_valid["REBA_3D_label"] = pd.Categorical(df_valid["REBA_3D_label"], categories=risk_order_valid, ordered=True)

confusion = pd.crosstab(df_valid["REBA_2D_label"], df_valid["REBA_3D_label"], dropna=False)
confusion = confusion.reindex(index=risk_order_valid, columns=risk_order_valid, fill_value=0)

print(confusion)

_fw = 8
plt.figure(figsize=(_fw, _fw))
ax = sns.heatmap(
    confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    annot_kws={"size": fs(FS_ANNOT + 4, _fw), "weight": "bold"},
    linewidths=2,
    linecolor="black",
    square=True,
    cbar=False,
)
for t in ax.texts:
    t.set_rotation(45)
ax.axhline(0, color="black", lw=3)
ax.axhline(confusion.shape[0], color="black", lw=3)
ax.axvline(0, color="black", lw=3)
ax.axvline(confusion.shape[1], color="black", lw=3)
plt.title("Confusion Matrix: REBA 2D vs 3D", fontsize=fs(FS_TITLE, _fw), weight="bold", pad=12)
plt.xlabel("REBA 3D (Reference)", fontsize=fs(FS_LABEL, _fw))
plt.ylabel("REBA 2D", fontsize=fs(FS_LABEL, _fw))
plt.xticks(fontsize=fs(FS_TICK, _fw), rotation=30, ha="right")
plt.yticks(fontsize=fs(FS_TICK, _fw), rotation=0)
plt.tight_layout()
plt.savefig(_os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"Lignes gardées pour la confusion matrix: {len(df_valid)} / {len(df)}")

# ---------------------------
# Compter les "invalid" en 2D vs 3D
# ---------------------------
n_invalid_2d = (df["REBA_2D_label"] == "invalid").sum()
n_invalid_3d = (df["REBA_3D_label"] == "invalid").sum()
n_total = len(df)

print(f"Invalid 2D : {n_invalid_2d} / {n_total} ({100*n_invalid_2d/n_total:.1f}%)")
print(f"Invalid 3D : {n_invalid_3d} / {n_total} ({100*n_invalid_3d/n_total:.1f}%)")
print(f"Diff invalid (2D - 3D) : {n_invalid_2d - n_invalid_3d}")

## ============================================================
# Selection colonnes + filtrage colonnes uniquement 0 (NaN ignorés)
# + garder LEFT et pas RIGHT (symétrie)
# ============================================================

def is_right_angle_col(col: str) -> bool:
    """
    Retourne True si la colonne correspond à un angle côté droit
    selon la convention de ton CODE 1:
      - Right_* = droit
      - Elbow = coude droit (Left_elbow = gauche)
      - Knee  = genou droit (Left_knee = gauche)
    """
    base = re.sub(r"_(2D|3D)$", "", col)
    return base.startswith("Right_") or base in {"Elbow", "Knee"}

# 1) Colonnes angles (fins par _2D/_3D, hors Tables/REBA/Error/labels)
#    + filtrage: on retire les angles RIGHT
angles_cols_initial = [
    c for c in df.columns
    if re.search(r"_(2D|3D)$", c)
    and not c.startswith("Table_")
    and not c.startswith("REBA")
    and c != "Error"
    and not c.endswith("_label")
    and not is_right_angle_col(c)  # <-- ICI: on enlève le côté droit
]

# 2) Colonnes pour corrélations (angles + tables + REBA + Error si tu veux)
wanted = [
    "Table_A_2D", "Table_B_2D", "Table_C_2D",
    "Table_A_3D", "Table_B_3D", "Table_C_3D",
    "REBA_2D", "REBA_3D", "Error"
]
cols_present = angles_cols_initial + [c for c in wanted if c in df.columns]

# 3) Sélection + conversion numérique
df_selected = df[cols_present].apply(pd.to_numeric, errors="coerce")

# (Optionnel mais safe) Supprimer colonnes entièrement NaN
df_selected = df_selected.loc[:, df_selected.notna().any(axis=0)]

# 4) Supprimer les colonnes contenant uniquement des 0 (en ignorant les NaN)
zero_cols = [
    c for c in df_selected.columns
    if not df_selected[c].dropna().empty and (df_selected[c].dropna() == 0).all()
]
df_selected = df_selected.drop(columns=zero_cols)

print(f"Colonnes supprimées (uniquement 0): {len(zero_cols)}")
print(zero_cols)

# IMPORTANT: reconstruire les listes à partir de df_selected (sinon KeyError)
cols_present = df_selected.columns.tolist()

angles_cols = [
    c for c in cols_present
    if re.search(r"_(2D|3D)$", c)
    and not c.startswith("Table_")
    and not c.startswith("REBA")
    and c != "Error"
    and not c.endswith("_label")
]
angles_2d_cols = [c for c in angles_cols if c.endswith("_2D")]
angles_3d_cols = [c for c in angles_cols if c.endswith("_3D")]


# ============================================================
# PCA (robuste)
# ============================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cumulative_variance(df_subset: pd.DataFrame) -> np.ndarray:
    X_std = StandardScaler().fit_transform(df_subset.values)
    pca = PCA().fit(X_std)
    return np.cumsum(pca.explained_variance_ratio_)


def run_pca(df_subset: pd.DataFrame, title: str, plot_cumvar: bool = True):
    X = df_subset.values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)

    # --- OPTIONAL cumulative variance plot (disabled for your use-case) ---
    if plot_cumvar:
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(cum_var_ratio) + 1), cum_var_ratio, marker="o")
        plt.xlabel("Nombre de composantes principales")
        plt.ylabel("Variance expliquée cumulée")
        plt.title(f"PCA — {title}\nVariance expliquée cumulée")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(_os.path.join(OUTPUT_DIR, f"pca_cumvar_{title.replace(' ', '_').lower()}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Contributions PC1-3 (si possible)
    n_pc = min(3, pca.components_.shape[0])
    for pc_idx in range(n_pc):
        loadings = pd.Series(pca.components_[pc_idx], index=df_subset.columns)
        contrib = (loadings ** 2)
        contrib_percent = 100 * contrib / contrib.sum()
        print(f"\nTop variables PCA{pc_idx+1} — {title} (%):\n")
        print(contrib_percent.sort_values(ascending=False).head(10))

    return pca, X_pca


def plot_biplot_individuals(X_pca, pca, title, color_labels):
    if X_pca.shape[1] < 2:
        print(f"[WARN] {title}: pas assez de composantes pour un scatter PC1/PC2.")
        return
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color_labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="REBA")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title(f"Biplot — {title} — Individuals colored by REBA")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUT_DIR, f"biplot_{title.replace(' ', '_').lower()}.png"), dpi=300, bbox_inches="tight")
    plt.close()


# Color palette by body segment for PCA arrows
_SEGMENT_COLORS = {
    "Neck": "#E41A1C",
    "Torso": "#377EB8",
    "Left_elbow": "#4DAF4A",
    "Left_knee": "#984EA3",
    "Left_shoulder": "#FF7F00",
    "Table": "#A65628",
    "REBA": "#F781BF",
}

def _segment_color(var_name):
    """Return a color based on the body segment of the variable."""
    base = re.sub(r"_(2D|3D)$", "", var_name)
    for prefix, color in _SEGMENT_COLORS.items():
        if base.startswith(prefix):
            return color
    return "#333333"

def _short_label(var_name):
    """Strip _2D/_3D suffix for cleaner labels."""
    return re.sub(r"_(2D|3D)$", "", var_name)

def _wrap_label(label, max_len=15):
    """Wrap label to two lines at the nearest _ if longer than max_len."""
    if len(label) <= max_len:
        return label
    idx = label.rfind("_", 0, max_len)
    if idx == -1:
        idx = label.find("_")
    if idx == -1:
        return label
    return label[:idx] + "\n" + label[idx + 1:]

LOADING_NORM_THRESH = 0.35  # only show arrows with norm >= this

def plot_correlation_plane(pca, df_subset, pc_x, pc_y, title):
    if pca.components_.shape[0] <= max(pc_x, pc_y):
        print(f"[WARN] {title}: pas assez de composantes pour PC{pc_x+1}/PC{pc_y+1}.")
        return

    ARROW_SCALE = 1.1
    fig, ax = plt.subplots(figsize=(16, 16))
    circle = plt.Circle((0, 0), ARROW_SCALE, facecolor="none", edgecolor="black", linestyle="--", lw=1.5)
    ax.add_artist(circle)

    loadings = pca.components_.T[:, [pc_x, pc_y]]

    # First pass: collect strong arrows info
    strong = []
    for i, var in enumerate(df_subset.columns):
        lx, ly = loadings[i, 0], loadings[i, 1]
        norm = np.sqrt(lx**2 + ly**2)
        if norm < LOADING_NORM_THRESH:
            ax.annotate("", xy=(lx * ARROW_SCALE, ly * ARROW_SCALE), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.15",
                                        color="#AAAAAA", lw=2.5))
        else:
            arrow_angle = np.degrees(np.arctan2(ly, lx))
            strong.append({"var": var, "lx": lx, "ly": ly, "norm": norm, "arrow_angle": arrow_angle})

    # Detect overlapping labels and nudge — multi-pass
    NUDGE_DEG = 10.0
    OVERLAP_THRESH = 22.0  # degrees apart to consider overlapping
    nudges = [0.0] * len(strong)
    for _pass in range(4):
        changed = False
        for i in range(len(strong)):
            for j in range(i + 1, len(strong)):
                diff = abs((strong[i]["arrow_angle"] + nudges[i]) - (strong[j]["arrow_angle"] + nudges[j]))
                if diff > 180:
                    diff = 360 - diff
                if diff < OVERLAP_THRESH:
                    if strong[i]["ly"] >= strong[j]["ly"]:
                        nudges[i] += NUDGE_DEG
                        nudges[j] -= NUDGE_DEG
                    else:
                        nudges[i] -= NUDGE_DEG
                        nudges[j] += NUDGE_DEG
                    changed = True
        if not changed:
            break

    # Second pass: draw arrows and labels
    for idx, s in enumerate(strong):
        lx, ly, var = s["lx"], s["ly"], s["var"]
        color = _segment_color(var)
        label = _wrap_label(_short_label(var))

        ax.annotate("", xy=(lx * ARROW_SCALE, ly * ARROW_SCALE), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.18",
                                    color=color, lw=3))

        # Label angle = arrow angle + nudge, then flip if upside down
        angle_deg = s["arrow_angle"] + nudges[idx]
        text_angle = angle_deg
        if text_angle > 90:
            text_angle -= 180
        elif text_angle < -90:
            text_angle += 180

        # Place label at nudged position outside arrow tip
        rad = np.radians(angle_deg)
        r = s["norm"] * 1.15
        tx, ty = r * np.cos(rad), r * np.sin(rad)
        ha = "left" if np.cos(rad) >= 0 else "right"

        ax.text(tx, ty, label, color=color, fontsize=36, weight="bold",
                ha=ha, va="center", rotation=text_angle, rotation_mode="anchor")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"Correlation Plane — {title} — PC{pc_x+1} vs PC{pc_y+1}", fontsize=37, weight="bold", pad=15)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}% var)", fontsize=36)
    ax.set_ylabel(f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}% var)", fontsize=36)
    ax.tick_params(labelsize=28)
    ax.set_aspect("equal")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUT_DIR, f"correlation_plane_{title.replace(' ', '_').lower()}_pc{pc_x+1}_pc{pc_y+1}.png"), dpi=300, bbox_inches="tight")
    plt.close()


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_biplot_individuals_3d(X_pca, pca, title, color_labels):
    if X_pca.shape[1] < 3:
        print(f"[WARN] {title}: pas assez de composantes pour un scatter 3D (PC1/PC2/PC3).")
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=color_labels, cmap="viridis", alpha=0.7)
    fig.colorbar(p, label="REBA")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    ax.set_title(f"Biplot 3D — {title} — Individuals colored by REBA")
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUT_DIR, f"biplot_3d_{title.replace(' ', '_').lower()}.png"), dpi=300, bbox_inches="tight")
    plt.close()


# Masques REBA valides (pour colorer, et filtrer les lignes)
mask_reba_2d = df_selected["REBA_2D"].notna() if "REBA_2D" in df_selected.columns else pd.Series(False, index=df_selected.index)
mask_reba_3d = df_selected["REBA_3D"].notna() if "REBA_3D" in df_selected.columns else pd.Series(False, index=df_selected.index)

# PCA input : lignes REBA valides + dropna sur angles
df_pca_clean_2d = df_selected.loc[mask_reba_2d, angles_2d_cols].dropna() if angles_2d_cols else pd.DataFrame()
df_pca_clean_3d = df_selected.loc[mask_reba_3d, angles_3d_cols].dropna() if angles_3d_cols else pd.DataFrame()

color_2d = df_selected.loc[df_pca_clean_2d.index, "REBA_2D"].astype(float).values if (not df_pca_clean_2d.empty and "REBA_2D" in df_selected.columns) else None
color_3d = df_selected.loc[df_pca_clean_3d.index, "REBA_3D"].astype(float).values if (not df_pca_clean_3d.empty and "REBA_3D" in df_selected.columns) else None


# Variance cumulée comparée (seulement si assez de data)
def enough_for_pca(df_sub: pd.DataFrame) -> bool:
    return (df_sub is not None) and (not df_sub.empty) and (df_sub.shape[0] >= 2) and (df_sub.shape[1] >= 2)


if enough_for_pca(df_pca_clean_2d) and enough_for_pca(df_pca_clean_3d):
    var_cum_2d = cumulative_variance(df_pca_clean_2d)
    var_cum_3d = cumulative_variance(df_pca_clean_3d)

    _fw = 12
    fig, ax = plt.subplots(figsize=(_fw, 9))
    ax.plot(range(1, len(var_cum_2d) + 1), var_cum_2d * 100,
            marker="o", markersize=10, lw=2.5, color="#2171B5", label="2D Angles")
    ax.plot(range(1, len(var_cum_3d) + 1), var_cum_3d * 100,
            marker="s", markersize=10, lw=2.5, color="#E41A1C", label="3D Angles")
    ax.axhline(80, color="grey", ls="--", lw=1.2)
    ax.text(0.6, 81, "80%", fontsize=fs(FS_TEXT, _fw), color="grey", weight="bold")
    ax.set_xlabel("Number of Principal Components", fontsize=fs(FS_LABEL, _fw))
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=fs(FS_LABEL, _fw))
    ax.set_title("Cumulative Explained Variance\n2D Angles vs 3D Angles", fontsize=fs(FS_TITLE, _fw), weight="bold", pad=12)
    ax.legend(fontsize=fs(FS_LEGEND, _fw), frameon=True, fancybox=True, shadow=True)
    ax.tick_params(labelsize=fs(FS_TICK, _fw))
    ax.set_xticks(range(1, max(len(var_cum_2d), len(var_cum_3d)) + 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUT_DIR, "pca_cumvar_comparison_2d_vs_3d.png"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("[WARN] Variance cumulée: pas assez de données pour comparer 2D et 3D (>=2 lignes et >=2 features).")


# === PCA 2D ===
if enough_for_pca(df_pca_clean_2d):
    print("\n=== PCA sur angles 2D (REBA valide uniquement) ===\n")
    # Disable per-dataset cumulative variance plot (already shown in comparison plot)
    pca_2d, X_pca_2d = run_pca(df_pca_clean_2d, "Angles 2D", plot_cumvar=False)

    plot_correlation_plane(pca_2d, df_pca_clean_2d, pc_x=0, pc_y=1, title="Angles 2D")
    plot_correlation_plane(pca_2d, df_pca_clean_2d, pc_x=0, pc_y=2, title="Angles 2D")
else:
    print("\n[WARN] PCA 2D: pas assez de données/colonnes présentes (>=2 lignes et >=2 features).\n")


# === PCA 3D ===
if enough_for_pca(df_pca_clean_3d):
    print("\n=== PCA sur angles 3D (REBA valide uniquement) ===\n")
    # Disable per-dataset cumulative variance plot (already shown in comparison plot)
    pca_3d, X_pca_3d = run_pca(df_pca_clean_3d, "Angles 3D", plot_cumvar=False)

    plot_correlation_plane(pca_3d, df_pca_clean_3d, pc_x=0, pc_y=1, title="Angles 3D")
    plot_correlation_plane(pca_3d, df_pca_clean_3d, pc_x=0, pc_y=2, title="Angles 3D")
else:
    print("\n[WARN] PCA 3D: pas assez de données/colonnes présentes (>=2 lignes et >=2 features).\n")

# ============================================================
# Concatenated PCA correlation planes (1 row x 4 columns)
# ============================================================
from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont

_pca_panels = [
    "correlation_plane_angles_2d_pc1_pc2.png",
    "correlation_plane_angles_2d_pc1_pc3.png",
    "correlation_plane_angles_3d_pc1_pc2.png",
    "correlation_plane_angles_3d_pc1_pc3.png",
]
_pca_paths = [_os.path.join(OUTPUT_DIR, f) for f in _pca_panels]

if all(_os.path.exists(p) for p in _pca_paths):
    imgs = [_PILImage.open(p) for p in _pca_paths]
    # Resize all to same height (use the smallest)
    min_h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * min_h / im.height), min_h), _PILImage.LANCZOS) for im in imgs]
    total_w = sum(im.width for im in imgs)
    concat = _PILImage.new("RGB", (total_w, min_h), "white")
    x_offset = 0
    for im in imgs:
        concat.paste(im, (x_offset, 0))
        x_offset += im.width
    concat.save(_os.path.join(OUTPUT_DIR, "correlation_planes_all.png"), dpi=(300, 300))
    print(f"[OK] Concatenated PCA planes -> correlation_planes_all.png ({total_w}x{min_h})")
else:
    print("[WARN] Some PCA plane images missing, skipping concatenation.")

# ============================================================
# Correlation matrix + Bootstrap IC 95%
# Display only LOWER triangle, only |r| >= THRESH_R, compact (NO SORT)
# Annotations: 1 significant figure (e.g., 0.47->0.5, 0.33->0.3)
# ============================================================

THRESH_R = 0.3  # keep only correlations with abs(r) >= THRESH_R

df_corr = df_selected.copy()
cols_present = df_corr.columns.tolist()

if len(cols_present) < 2:
    print("[WARN] Corr/Bootstrap: moins de 2 colonnes après filtrage -> skip.")
    raise SystemExit

# ---------------------------
# Compute correlation matrix
# ---------------------------
correlation_matrix = df_corr.corr()

# ---------------------------
# Bootstrap IC 95%
# ---------------------------
n_bootstrap = 1000
rng = np.random.default_rng(seed=42)

p = len(cols_present)
bootstrap_corrs = np.full((p, p, n_bootstrap), np.nan)

for i in range(n_bootstrap):
    df_sample = df_corr.sample(
        n=len(df_corr),
        replace=True,
        random_state=int(rng.integers(0, 1_000_000))
    )
    cmat = df_sample.corr().values
    if cmat.shape == (p, p):
        bootstrap_corrs[:, :, i] = cmat

lower_bound = np.nanpercentile(bootstrap_corrs, 2.5, axis=2)
upper_bound = np.nanpercentile(bootstrap_corrs, 97.5, axis=2)

# ---------------------------
# Summary table (robust + |r| >= THRESH_R)
# ---------------------------
summary_rows = []
for i in range(p):
    for j in range(i + 1, p):
        r = correlation_matrix.values[i, j]
        lb = lower_bound[i, j]
        ub = upper_bound[i, j]

        ic_contains_zero = (lb <= 0.0 <= ub) if (not np.isnan(lb) and not np.isnan(ub)) else True

        summary_rows.append({
            "Var1": cols_present[i],
            "Var2": cols_present[j],
            "Correlation": r,
            "CI_lower": lb,
            "CI_upper": ub,
            "IC_contains_zero": ic_contains_zero,
            "Abs_corr": abs(r) if not np.isnan(r) else np.nan
        })

df_summary = pd.DataFrame(summary_rows)

df_significant = df_summary[df_summary["IC_contains_zero"] == False].copy()
df_significant = df_significant[df_significant["Abs_corr"] >= THRESH_R].sort_values(by="Abs_corr", ascending=False)

n_pairs = p * (p - 1) // 2
print(f"\n=== Corrélations robustes (IC exclut 0) ET |r| >= {THRESH_R} — total: {len(df_significant)} / {n_pairs} ===\n")
if not df_significant.empty:
    print(df_significant[["Var1", "Var2", "Correlation", "CI_lower", "CI_upper"]].to_string(index=False))
else:
    print("(aucune)")

# ============================================================
# Plot helpers: LOWER triangle only, threshold + compact (NO SORT)
# ============================================================

def round_sig(x, sig=1):
    """Round to `sig` significant figures (handles NaN and 0)."""
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return float(np.round(x, sig - int(np.floor(np.log10(abs(x)))) - 1))

def compact_threshold(mat: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    1) set |r| < thresh to NaN
    2) drop all-NaN rows/cols (ignoring diagonal)
    3) KEEP ORIGINAL ORDER (no sorting)
    """
    m = mat.copy()

    # threshold
    m[m.abs() < thresh] = np.nan

    # drop empty rows/cols (excluding diagonal effect)
    m_for_keep = m.copy()
    np.fill_diagonal(m_for_keep.values, np.nan)
    keep = m_for_keep.notna().any(axis=1)

    m = m.loc[keep, keep]
    return m

def plot_lower_triangle_heatmap(mat: pd.DataFrame, title: str, out_png: str, sig_figs: int = 1):
    """
    LOWER triangle only (diagonal hidden), NaNs masked, annotations blank on masked.
    Annotations are rounded to `sig_figs` significant figures.
    """
    if mat.shape[0] == 0:
        print(f"[INFO] {title}: rien à afficher (matrice vide après filtrage).")
        return

    # mask: hide NaNs + UPPER triangle + diagonal
    mask_nan = mat.isna().values
    mask_upper = np.triu(np.ones(mat.shape, dtype=bool), k=1)  # includes diagonal -> hidden
    mask = mask_nan | mask_upper

    # annotations: 1 significant figure, blank on masked
    annot = mat.applymap(lambda v: round_sig(v, sig=sig_figs)).astype(object)
    annot.values[mask] = ""

    # dynamic size
    n = mat.shape[0]
    fig_size = max(10, 0.6 * n)

    plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(
        mat,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": fs(FS_ANNOT, fig_size / 2)},
        cbar=False,
    )
    for t in ax.texts:
        t.set_rotation(45)
    plt.title(title, fontsize=fs(FS_TITLE, fig_size / 2), weight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=fs(FS_TICK, fig_size / 2))
    plt.yticks(rotation=0, fontsize=fs(FS_TICK, fig_size / 2))
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUT_DIR, out_png), dpi=300, bbox_inches="tight")
    plt.close()



# ---------------------------
# Robust correlation matrix (IC excludes 0): LOWER triangle, compact (NO SORT), |r| >= THRESH_R
# ---------------------------
robust_corr_matrix = correlation_matrix.copy()

for i in range(p):
    for j in range(p):
        lb = lower_bound[i, j]
        ub = upper_bound[i, j]
        ic_contains_zero = (lb <= 0.0 <= ub) if (not np.isnan(lb) and not np.isnan(ub)) else True
        if ic_contains_zero:
            robust_corr_matrix.values[i, j] = np.nan

robust_to_plot = compact_threshold(robust_corr_matrix, THRESH_R)
plot_lower_triangle_heatmap(
    robust_to_plot,
    title=f"Robust Correlations (IC excludes 0) — lower triangle — only |r| >= {THRESH_R}",
    out_png="correlation_heatmap_only_robust_abs_ge_thresh_lower_triangle.png",
    sig_figs=1
)

# ============================================================
# SPLIT correlation into 3 sub-matrices (2D, 3D, cross 2D vs 3D)
# ============================================================

def split_heatmap(corr_mat, row_cols, col_cols, title, out_png, lower_triangle=False, show_cbar=False):
    """Plot a rectangular sub-matrix of correlations with readable fonts."""
    row_cols = [c for c in row_cols if c in corr_mat.columns]
    col_cols = [c for c in col_cols if c in corr_mat.columns]
    if not row_cols or not col_cols:
        print(f"[WARN] {title}: no columns available, skipping.")
        return

    sub = corr_mat.loc[row_cols, col_cols].copy()

    # mask NaN + optional lower triangle
    mask = sub.isna().values
    if lower_triangle:
        mask_upper = np.triu(np.ones(sub.shape, dtype=bool), k=1)
        mask = mask | mask_upper

    annot = sub.round(2).astype(object)
    annot.values[mask] = ""

    n_rows, n_cols = sub.shape
    fig_w = max(10, 0.9 * n_cols + 2)
    fig_h = max(7,  0.7 * n_rows + 2)

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        sub,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        annot_kws={"size": fs(FS_ANNOT, fig_w / 1.4)},
        linewidths=1,
        linecolor="white",
        square=False,
        cbar=show_cbar,
        cbar_kws={"label": "Pearson r", "shrink": 0.8} if show_cbar else {},
    )
    for t in plt.gca().texts:
        t.set_rotation(45)
    plt.title(title, fontsize=fs(FS_TITLE, fig_w / 1.4), weight="bold", pad=12)
    plt.xticks(fontsize=fs(FS_TICK, fig_w / 1.4), rotation=45, ha="right")
    plt.yticks(fontsize=fs(FS_TICK, fig_w / 1.4), rotation=0)
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUT_DIR, out_png), dpi=300, bbox_inches="tight")
    plt.close()


# Identify 2D vs 3D column groups
robust_cols = robust_to_plot.columns.tolist()
cols_2d_only = [c for c in robust_cols if c.endswith("_2D")]
cols_3d_only = [c for c in robust_cols if c.endswith("_3D")]

# 1) Intra-2D correlations (lower triangle)
split_heatmap(
    robust_to_plot, cols_2d_only, cols_2d_only,
    "Robust Correlations — 2D Angles",
    "correlation_split_2d.png",
    lower_triangle=True,
)

# 2) Intra-3D correlations (lower triangle)
split_heatmap(
    robust_to_plot, cols_3d_only, cols_3d_only,
    "Robust Correlations — 3D Angles",
    "correlation_split_3d.png",
    lower_triangle=True,
)

# 3) Cross 2D vs 3D correlations
split_heatmap(
    robust_to_plot, cols_2d_only, cols_3d_only,
    "Robust Correlations — 2D vs 3D (cross)",
    "correlation_split_2d_vs_3d.png",
    show_cbar=True,
)


# ============================================================
# Concatenated correlation split matrices (1 row x 3 columns)
# ============================================================
_corr_split_panels = [
    "correlation_split_2d.png",
    "correlation_split_3d.png",
    "correlation_split_2d_vs_3d.png",
]
_corr_split_paths = [_os.path.join(OUTPUT_DIR, f) for f in _corr_split_panels]

if all(_os.path.exists(p) for p in _corr_split_paths):
    imgs = [_PILImage.open(p) for p in _corr_split_paths]
    min_h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * min_h / im.height), min_h), _PILImage.LANCZOS) for im in imgs]
    total_w = sum(im.width for im in imgs)
    label_h = 240  # pixels for (a), (b), (c) labels
    concat = _PILImage.new("RGB", (total_w, min_h + label_h), "white")
    x_offset = 0
    for im in imgs:
        concat.paste(im, (x_offset, 0))
        x_offset += im.width
    # Draw (a), (b), (c) labels with PIL ImageDraw
    draw = _PILDraw.Draw(concat)
    try:
        font = _PILFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 124)
    except Exception:
        font = _PILFont.load_default()
    x_offset = 0
    for im, lbl in zip(imgs, ["(a)", "(b)", "(c)"]):
        bbox = draw.textbbox((0, 0), lbl, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx = x_offset + im.width // 2 - tw // 2
        cy = min_h + label_h // 2 - th // 2
        draw.text((cx, cy), lbl, fill="black", font=font)
        x_offset += im.width
    concat.save(_os.path.join(OUTPUT_DIR, "correlation_splits_all.png"), dpi=(300, 300))
    print(f"[OK] Concatenated correlation splits -> correlation_splits_all.png ({total_w}x{min_h + label_h})")
else:
    print("[WARN] Some correlation split images missing, skipping concatenation.")

# ============================================================
# DOT PLOT of robust correlations
# ============================================================

if not df_significant.empty:
    df_dot = df_significant.head(40).copy()  # top 40 strongest
    df_dot["pair"] = df_dot["Var1"] + "  vs  " + df_dot["Var2"]
    df_dot = df_dot.sort_values("Correlation")

    n_pairs_dot = len(df_dot)
    fig_h = max(8, 0.4 * n_pairs_dot + 1.5)

    _fw = 10
    fig, ax = plt.subplots(figsize=(_fw, fig_h))

    colors = df_dot["Correlation"].apply(lambda r: "#2171B5" if r > 0 else "#DE2D26")
    sizes = df_dot["Abs_corr"] * 300

    ax.scatter(df_dot["Correlation"], range(n_pairs_dot), s=sizes, c=colors, alpha=0.8, edgecolors="black", linewidths=0.5)

    # CI bars
    for i, (_, row) in enumerate(df_dot.iterrows()):
        ax.plot([row["CI_lower"], row["CI_upper"]], [i, i], color="grey", lw=1, zorder=0)

    ax.set_yticks(range(n_pairs_dot))
    ax.set_yticklabels(df_dot["pair"], fontsize=fs(FS_TICK, _fw))
    ax.set_xlabel("Correlation (r)", fontsize=fs(FS_LABEL, _fw))
    ax.set_title("Top Robust Correlations (IC 95% excludes 0, |r| >= 0.3)", fontsize=fs(FS_TITLE, _fw), weight="bold", pad=12)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(_os.path.join(OUTPUT_DIR, "correlation_dotplot.png"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("[WARN] Dot plot: aucune corrélation significative à afficher.")

