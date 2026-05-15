"""
shap_ehr_analysis.py

SHAP analysis of the EHR encoder from the contrastive learning model.

Pipeline:
    1. Load frozen EHR embeddings (ehr_0..ehr_127 + severity)
    2. Load raw EHR feature matrix (6 normalized clinical features)
    3. Train a logistic regression linear probe on the embeddings
       to confirm embedding quality
    4. Train a second logistic regression directly on raw features
       — this is what SHAP explains
    5. Run KernelExplainer SHAP on raw features -> severity predictions
    6. Save plots and tables

Why explain raw features (not embeddings):
    The 128-d embedding dimensions have no clinical meaning.
    Explaining raw features answers the clinically useful question:
    "which of PSA max, age, LOS, etc. drive severity predictions?"

Outputs (written to --out directory):
    shap_summary_bar.png          mean |SHAP| per feature, stacked by class
    shap_beeswarm_low.png         beeswarm for Low severity class
    shap_beeswarm_moderate.png    beeswarm for Moderate severity class
    shap_beeswarm_high.png        beeswarm for High severity class
    shap_heatmap.png              mean SHAP heatmap: features x classes
    shap_feature_importance.csv   ranked feature importance table
    shap_values.csv               raw per-sample SHAP values

Usage:
    python shap_ehr_analysis.py \
        --ehr_matrix  path/to/ehr_feature_matrix_balanced.csv \
        --ehr_labels  path/to/ehr_severity_balanced.csv \
        --embeddings  path/to/ehr_embeddings.csv \
        --out         path/to/output_dir
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import shap
except ImportError:
    sys.exit("ERROR: shap not installed. Run: pip install shap")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_NAMES  = {0: "Low", 1: "Moderate", 2: "High"}
SEVERITY_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#F44336"}

# Display names for the 6 EHR features (order must match feature matrix columns)
FEATURE_DISPLAY_NAMES = [
    "PSA max (ng/mL)",
    "PSA order count",
    "Procedure count",
    "Medication count",
    "Admission LOS (days)",
    "Age (years)",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(path: str):
    """
    Load EHR embeddings CSV.

    Expects columns:  ehr_0, ehr_1, ..., ehr_127, severity
    Returns:
        X_emb  : np.ndarray [N, 128]
        y_emb  : np.ndarray [N]  (int severity labels)
    """
    print(f"Loading EHR embeddings from: {path}")
    df = pd.read_csv(path)

    emb_cols = sorted(
        [c for c in df.columns if c.startswith("ehr_")],
        key=lambda c: int(c.split("_")[1])
    )
    if not emb_cols:
        sys.exit(
            "ERROR: No columns starting with 'ehr_' found in embeddings file.\n"
            f"  Columns present: {list(df.columns)}"
        )

    label_col = _find_col(df, ["severity", "severity_int", "label"], "severity label")

    X_emb = df[emb_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    y_emb = df[label_col].astype(int).values

    print(f"  Embeddings shape : {X_emb.shape}")
    print(f"  Label column     : '{label_col}'")
    print(f"  Class counts     : {dict(zip(*np.unique(y_emb, return_counts=True)))}")
    print()
    return X_emb, y_emb


def load_raw_features(matrix_path: str, labels_path: str):
    """
    Load raw normalized EHR feature matrix and severity labels.

    Returns:
        X_raw         : np.ndarray [N, 6]
        y             : np.ndarray [N]
        feature_names : list[str]  (display names, length 6)
    """
    print(f"Loading EHR feature matrix from: {matrix_path}")
    matrix_df = pd.read_csv(matrix_path)
    matrix_df.columns = matrix_df.columns.str.strip().str.lower()

    feature_cols = [c for c in matrix_df.columns if c != "subject_id"]
    X_raw = (
        matrix_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values
        .astype(np.float32)
    )

    # Map display names; fall back to raw column names if count differs
    if len(feature_cols) == len(FEATURE_DISPLAY_NAMES):
        feature_names = FEATURE_DISPLAY_NAMES
    else:
        print(
            f"  WARNING: expected {len(FEATURE_DISPLAY_NAMES)} feature columns, "
            f"found {len(feature_cols)}. Using raw column names."
        )
        feature_names = feature_cols

    print(f"  Feature matrix shape : {X_raw.shape}")
    print(f"  Features             : {feature_names}")

    print(f"Loading EHR severity labels from: {labels_path}")
    labels_df = pd.read_csv(labels_path)
    label_col = _find_col(labels_df, ["severity_int", "severity", "label"], "severity label")
    y = labels_df[label_col].astype(int).values

    print(f"  Label column  : '{label_col}'")
    print(f"  Class counts  : {dict(zip(*np.unique(y, return_counts=True)))}")
    print()
    return X_raw, y, feature_names


def _find_col(df: pd.DataFrame, candidates: list, description: str) -> str:
    """Return first candidate column name present in df, or exit with error."""
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(
        f"ERROR: Could not find {description} column.\n"
        f"  Tried: {candidates}\n"
        f"  Columns present: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_embedding_probe(X_emb: np.ndarray, y: np.ndarray) -> float:
    """
    Train logistic regression on EHR embeddings.
    Used only to report embedding quality; not used for SHAP.

    Returns macro F1 on held-out test set.
    """
    print("Training linear probe on EHR embeddings (quality check)...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_emb, y, test_size=0.2, stratify=y, random_state=42
    )
    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    probe.fit(X_tr, y_tr)
    f1 = f1_score(y_te, probe.predict(X_te), average="macro")
    print(f"  Embedding probe macro F1 : {f1:.3f}")
    print()
    return f1


def train_raw_classifier(X_raw: np.ndarray, y: np.ndarray):
    """
    Train logistic regression directly on raw normalized features.
    SHAP explains this model — the mapping from 6 clinical features
    to predicted severity class.

    Returns:
        clf    : fitted sklearn Pipeline (StandardScaler + LogisticRegression)
        X_te   : test-set raw features
        y_te   : test-set true labels
    """
    print("Training logistic regression on raw EHR features (for SHAP)...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    f1 = f1_score(y_te, y_pred, average="macro")
    print(f"  Raw feature classifier macro F1 : {f1:.3f}")
    print()
    print(classification_report(y_te, y_pred, target_names=["Low", "Moderate", "High"]))
    return clf, X_te, y_te


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap(
    clf: Pipeline,
    X_raw: np.ndarray,
    y: np.ndarray,
    n_background: int = 100,
    n_explain: int = 500,
    nsamples: int = 200,
):
    """
    Run KernelExplainer SHAP on the raw-feature classifier.

    Returns:
        shap_values : list of 3 np.ndarray, each [n_explain, n_features]
        X_explain   : np.ndarray [n_explain, n_features]  (scaled)
        y_explain   : np.ndarray [n_explain]
    """
    print("Running SHAP KernelExplainer...")

    scaler = clf.named_steps["scaler"]
    lr     = clf.named_steps["lr"]
    X_scaled = scaler.transform(X_raw)

    # Background: k-means summary of a random subset
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_scaled), min(n_background, len(X_scaled)), replace=False)
    X_bg   = shap.kmeans(X_scaled[bg_idx], min(10, len(bg_idx)))

    explainer = shap.KernelExplainer(lr.predict_proba, X_bg)

    # Explain a representative subset
    exp_idx   = rng.choice(len(X_scaled), min(n_explain, len(X_scaled)), replace=False)
    X_explain = X_scaled[exp_idx]
    y_explain = y[exp_idx]

    print(f"  Background samples : {n_background} (k-means compressed to 10)")
    print(f"  Explained samples  : {len(X_explain)}")
    print(f"  SHAP nsamples      : {nsamples}")

    raw_shap = explainer.shap_values(X_explain, nsamples=nsamples)
    shap_values = _parse_shap_output(raw_shap, len(X_explain), X_explain.shape[1])

    print(
        f"  SHAP output shape  : {len(shap_values)} classes x "
        f"{shap_values[0].shape[0]} samples x "
        f"{shap_values[0].shape[1]} features  ✓"
    )
    print()
    return shap_values, X_explain, y_explain


def _parse_shap_output(raw_shap, n_samples: int, n_features: int):
    """
    Normalize SHAP output to a list of 3 arrays shaped [n_samples, n_features].
    Handles the various shapes different SHAP versions return.
    """
    n_classes = 3

    if isinstance(raw_shap, list):
        return [np.array(sv) for sv in raw_shap[:n_classes]]

    if isinstance(raw_shap, np.ndarray):
        if raw_shap.ndim == 3:
            if raw_shap.shape[2] == n_classes:
                # [N, F, C]
                return [raw_shap[:, :, i] for i in range(n_classes)]
            if raw_shap.shape[0] == n_classes:
                # [C, N, F]
                return [raw_shap[i] for i in range(n_classes)]
        if raw_shap.ndim == 2:
            # Collapsed — broadcast across features
            return [
                raw_shap[:, [i]] * np.ones((n_samples, n_features))
                for i in range(n_classes)
            ]

    sys.exit(f"ERROR: Unrecognised SHAP output type/shape: {type(raw_shap)}, "
             f"{getattr(raw_shap, 'shape', 'N/A')}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_summary_bar(shap_values, feature_names: list, out_dir: str):
    """Grouped horizontal bar chart: mean |SHAP| per feature per class."""
    print("  Plotting summary bar chart...")

    mean_abs = np.array([np.abs(sv).mean(axis=0) for sv in shap_values])  # [3, F]
    n_feats  = len(feature_names)
    x        = np.arange(n_feats)
    width    = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(3):
        ax.barh(
            x + (i - 1) * width,
            mean_abs[i],
            height=width,
            color=SEVERITY_COLORS[i],
            alpha=0.85,
            label=SEVERITY_NAMES[i],
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_yticks(x)
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        "EHR Feature Importance — Mean |SHAP| by Severity Class",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Severity", fontsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    _style_ax(ax)
    plt.tight_layout()

    _save(fig, out_dir, "shap_summary_bar.png")


def plot_beeswarm(shap_values, X_explain: np.ndarray, feature_names: list, out_dir: str):
    """Beeswarm plot (one per severity class)."""
    print("  Plotting beeswarm plots...")

    rng = np.random.default_rng(0)
    cmap = plt.cm.RdYlBu_r

    for cls_idx, cls_name in SEVERITY_NAMES.items():
        sv    = shap_values[cls_idx]              # [N, F]
        order = np.argsort(np.abs(sv).mean(axis=0))  # ascending — bottom = least important

        fig, ax = plt.subplots(figsize=(9, 5))

        for plot_pos, feat_i in enumerate(order):
            shap_col = sv[:, feat_i]
            feat_col = X_explain[:, feat_i]

            norm     = mcolors.Normalize(vmin=feat_col.min(), vmax=feat_col.max())
            colors_i = cmap(norm(feat_col))
            jitter   = rng.uniform(-0.25, 0.25, size=len(shap_col))

            ax.scatter(
                shap_col,
                np.full(len(shap_col), plot_pos) + jitter,
                c=colors_i,
                alpha=0.45,
                s=14,
                linewidths=0,
            )

        ax.set_yticks(range(n_feats := len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in order], fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
        ax.set_title(
            f"SHAP Beeswarm — {cls_name} Severity\n"
            "Color: blue = low feature value, red = high feature value",
            fontsize=12, fontweight="bold",
        )
        _style_ax(ax)

        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Feature value (normalized)", fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["Low", "Mid", "High"])

        plt.tight_layout()
        _save(fig, out_dir, f"shap_beeswarm_{cls_name.lower()}.png")


def plot_heatmap(shap_values, feature_names: list, out_dir: str):
    """Mean SHAP heatmap: rows = severity classes, cols = features."""
    print("  Plotting SHAP heatmap...")

    mean_shap = np.array([sv.mean(axis=0) for sv in shap_values])  # [3, F]
    vmax = np.abs(mean_shap).max()

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(
        mean_shap,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels([SEVERITY_NAMES[i] for i in range(3)], fontsize=10)

    for i in range(3):
        for j in range(len(feature_names)):
            val = mean_shap[i, j]
            ax.text(
                j, i,
                f"{val:.3f}",
                ha="center", va="center",
                fontsize=8,
                color="white" if abs(val) > vmax * 0.5 else "black",
            )

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Mean SHAP value")
    ax.set_title(
        "Mean SHAP Values — EHR Features by Severity Class\n"
        "Positive = pushes toward this class   Negative = pushes away",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out_dir, "shap_heatmap.png")


def _save(fig, out_dir: str, filename: str):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {path}")


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def save_tables(
    shap_values,
    X_explain: np.ndarray,
    y_explain: np.ndarray,
    feature_names: list,
    out_dir: str,
):
    print("  Saving SHAP tables...")

    # --- Feature importance ---
    overall = np.array([np.abs(sv).mean(axis=0) for sv in shap_values]).mean(axis=0)

    importance_df = pd.DataFrame({
        "rank":            range(1, len(feature_names) + 1),
        "feature":         feature_names,
        "mean_abs_shap":   overall,
        **{
            f"mean_abs_shap_{SEVERITY_NAMES[i].lower()}": np.abs(sv).mean(axis=0)
            for i, sv in enumerate(shap_values)
        },
    })
    importance_df = (
        importance_df
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["rank"] = range(1, len(importance_df) + 1)

    imp_path = os.path.join(out_dir, "shap_feature_importance.csv")
    importance_df.to_csv(imp_path, index=False)
    print(f"    Feature importance -> {imp_path}")

    # Print to console
    print()
    header = f"  {'Rank':<5} {'Feature':<26} {'Overall':>9} {'Low':>9} {'Moderate':>10} {'High':>9}"
    print(header)
    print("  " + "-" * 70)
    for _, row in importance_df.iterrows():
        print(
            f"  {int(row['rank']):<5} {row['feature']:<26} "
            f"{row['mean_abs_shap']:>9.4f} "
            f"{row['mean_abs_shap_low']:>9.4f} "
            f"{row['mean_abs_shap_moderate']:>10.4f} "
            f"{row['mean_abs_shap_high']:>9.4f}"
        )
    print()

    # --- Raw SHAP values per sample ---
    rows = []
    for cls_idx, sv in enumerate(shap_values):
        df_cls = pd.DataFrame(sv, columns=feature_names)
        df_cls.insert(0, "severity_true", y_explain)
        df_cls.insert(1, "explained_class", SEVERITY_NAMES[cls_idx])
        rows.append(df_cls)

    raw_path = os.path.join(out_dir, "shap_values.csv")
    pd.concat(rows, ignore_index=True).to_csv(raw_path, index=False)
    print(f"    Raw SHAP values  -> {raw_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(out_dir: str):
    expected = [
        "shap_summary_bar.png",
        "shap_beeswarm_low.png",
        "shap_beeswarm_moderate.png",
        "shap_beeswarm_high.png",
        "shap_heatmap.png",
        "shap_feature_importance.csv",
        "shap_values.csv",
    ]
    print("=" * 60)
    print("  SHAP Analysis Complete")
    print("=" * 60)
    for fname in expected:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            kb = os.path.getsize(fpath) / 1024
            print(f"  ✓  {fname:<42} ({kb:,.0f} KB)")
        else:
            print(f"  ✗  {fname:<42} (NOT FOUND)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ehr_matrix_path: str, ehr_labels_path: str, embeddings_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("=" * 60)
    print("  SHAP Analysis — EHR Encoder")
    print("=" * 60)
    print()

    # 1. Load data
    X_emb, y_emb                  = load_embeddings(embeddings_path)
    X_raw, y_raw, feature_names   = load_raw_features(ehr_matrix_path, ehr_labels_path)

    # Align lengths (embeddings and raw features may differ if sampling differs)
    n = min(len(X_emb), len(X_raw))
    if len(X_emb) != len(X_raw):
        print(
            f"WARNING: embedding rows ({len(X_emb)}) != raw feature rows ({len(X_raw)}). "
            f"Truncating to {n}."
        )
    X_emb  = X_emb[:n]
    y_emb  = y_emb[:n]
    X_raw  = X_raw[:n]
    y_raw  = y_raw[:n]

    # 2. Embedding quality check
    train_embedding_probe(X_emb, y_emb)

    # 3. Raw feature classifier (SHAP explains this)
    clf, X_te, y_te = train_raw_classifier(X_raw, y_raw)

    # 4. SHAP
    shap_values, X_explain, y_explain = compute_shap(clf, X_raw, y_raw)

    # 5. Plots
    print("Generating plots...")
    plot_summary_bar(shap_values, feature_names, out_dir)
    plot_beeswarm(shap_values, X_explain, feature_names, out_dir)
    plot_heatmap(shap_values, feature_names, out_dir)

    # 6. Tables
    print("Saving tables...")
    save_tables(shap_values, X_explain, y_explain, feature_names, out_dir)

    print_summary(out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP analysis of EHR encoder from contrastive model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ehr_matrix",
        required=True,
        help="Path to balanced EHR feature matrix CSV (normalized, 6 features)",
    )
    parser.add_argument(
        "--ehr_labels",
        required=True,
        help="Path to EHR severity labels CSV (must contain severity/severity_int column)",
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to EHR embeddings CSV (columns: ehr_0..ehr_127, severity)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for plots and tables",
    )
    parser.add_argument(
        "--n_background", type=int, default=100,
        help="Number of background samples for SHAP KernelExplainer",
    )
    parser.add_argument(
        "--n_explain", type=int, default=500,
        help="Number of samples to explain with SHAP",
    )
    parser.add_argument(
        "--nsamples", type=int, default=200,
        help="SHAP nsamples per explainer call (higher = more accurate, slower)",
    )

    args = parser.parse_args()
    main(args.ehr_matrix, args.ehr_labels, args.embeddings, args.out)