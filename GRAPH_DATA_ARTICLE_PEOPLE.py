import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv("ULTIMATE_ALL_DATA_EXPERIMENT.csv")

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
    plt.figure(figsize=(16, 8))

    # pivot_table pour éviter crash si doublons Person_ID/Window_ID
    heatmap_data = df.pivot_table(
        index="Person_ID",
        columns="Window_ID",
        values="Error",
        aggfunc="mean"
    )

    mask = heatmap_data.isna()
    annot = heatmap_data.round(1).astype(object)
    annot[mask] = ""

    sns.heatmap(
        heatmap_data,
        cmap=custom_cmap,
        center=0,
        vmin=vmin, vmax=vmax,
        mask=mask,
        annot=annot,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": "Error (REBA_2D - REBA_3D)"}
    )
    plt.title("Difference REBA 2D vs 3D by Person and Window")
    plt.xlabel("Window")
    plt.ylabel("Person ID")
    plt.tight_layout()
else:
    print("[WARN] Heatmap: colonnes Person_ID / Window_ID / Error manquantes.")

# ---------------------------
# HISTOGRAM Error (ignore NaN)
# ---------------------------
if "Error" in df.columns:
    plt.figure(figsize=(8, 4))
    df["Error"].dropna().hist(bins=30)
    plt.title("Global Histogram of Errors (REBA 2D - 3D)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(False)

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

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: REBA 2D vs 3D (valid only)")
plt.xlabel("REBA 3D (Reference)")
plt.ylabel("REBA 2D")
plt.tight_layout()
plt.show()

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
        plt.show()

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
    plt.show()


def plot_correlation_plane(pca, df_subset, pc_x, pc_y, title):
    if pca.components_.shape[0] <= max(pc_x, pc_y):
        print(f"[WARN] {title}: pas assez de composantes pour PC{pc_x+1}/PC{pc_y+1}.")
        return

    plt.figure(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="black", linestyle="--")
    plt.gca().add_artist(circle)

    loadings = pca.components_.T[:, [pc_x, pc_y]]
    for i, var in enumerate(df_subset.columns):
        plt.arrow(
            0, 0, loadings[i, 0], loadings[i, 1],
            color="darkblue", alpha=0.7, head_width=0.02, head_length=0.02
        )
        plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, var, color="darkred", fontsize=10)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel(f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}% var)")
    plt.ylabel(f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}% var)")
    plt.title(f"Correlation Plane — {title} — PC{pc_x+1} vs PC{pc_y+1}")
    plt.axhline(0, color="grey", lw=1)
    plt.axvline(0, color="grey", lw=1)
    plt.grid(False)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


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
    plt.show()


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

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(var_cum_2d) + 1), var_cum_2d * 100, marker="o", label="Angles 2D")
    plt.plot(range(1, len(var_cum_3d) + 1), var_cum_3d * 100, marker="s", label="Angles 3D")
    plt.axhline(70, color="grey", ls="--", lw=0.8)
    plt.axhline(90, color="grey", ls="--", lw=0.8)
    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Variance expliquée cumulée (%)")
    plt.title("Comparaison de la variance expliquée cumulée\nAngles 2D vs Angles 3D")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
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
# Correlation matrix + Bootstrap IC 95%
# Display only LOWER triangle, only |r| >= THRESH_R, compact (NO SORT)
# Annotations: 1 significant figure (e.g., 0.47->0.5, 0.33->0.3)
# ============================================================

THRESH_R = 0.3  # keep only correlations with abs(r) >= THRESH_R

df_corr = df_selected.copy()
cols_present = df_corr.columns.tolist()

if len(cols_present) < 2:
    print("[WARN] Corr/Bootstrap: moins de 2 colonnes après filtrage -> skip.")
    plt.show()
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
    sns.heatmap(
        mat,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "r"}
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()



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



