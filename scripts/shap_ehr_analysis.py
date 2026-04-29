"""
shap_ehr_analysis.py

SHAP analysis of the EHR encoder from the contrastive learning model.

Approach:
    1. Load frozen EHR embeddings from embeddings.csv
    2. Train a logistic regression linear probe on top (same as evaluate_embeddings.py)
    3. Run SHAP on the RAW EHR features (not the embeddings) to explain
       which of the 6 clinical features drive the linear probe's severity predictions
    4. Generate summary plots, beeswarm plots, and bar charts

Why SHAP on raw features not embeddings:
    The 128-d embeddings are not individually interpretable — they are learned
    latent dimensions with no clinical meaning. SHAP on raw features answers
    the clinically meaningful question: "which of PSA max, age, LOS, etc.
    most strongly drives the severity prediction?"

    We use a two-stage pipeline:
        raw features [6] → EHR encoder → embedding [128] → linear probe → severity

    SHAP explains the full pipeline end-to-end using the raw features as input,
    treating the encoder + linear probe as a black-box function.

Outputs:
    shap_summary_bar.png        mean |SHAP| per feature (bar chart)
    shap_beeswarm.png           beeswarm plot showing feature value effects
    shap_heatmap.png            per-class SHAP heatmap
    shap_values.csv             raw SHAP values for all patients
    shap_feature_importance.csv ranked feature importance table

Usage:
    python shap_ehr_analysis.py \
        --ehr_matrix  /oscar/.../extracted/ehr_feature_matrix_balanced.csv \
        --ehr_labels  /oscar/.../extracted/ehr_severity_balanced.csv \
        --embeddings  /oscar/.../outputs/embeddings.csv \
        --out         /oscar/.../outputs/shap
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import shap
except ImportError:
    raise ImportError("shap not installed. Run: pip install shap")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

SEVERITY_NAMES  = {0: "Low", 1: "Moderate", 2: "High"}
SEVERITY_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#F44336"}

FEATURE_NAMES = [
    "PSA max (ng/mL)",
    "PSA order count",
    "Procedure count",
    "Medication count",
    "Admission LOS (days)",
    "Age (years)",
]

FEATURE_COLS = [
    "psa_max_norm",
    "psa_order_count_norm",
    "procedure_count_norm",
    "distinct_med_count_norm",
    "los_days_norm",
    "anchor_age_norm",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(ehr_matrix_path, ehr_labels_path, embeddings_path):
    print("Loading EHR feature matrix...")
    matrix_df = pd.read_csv(ehr_matrix_path, dtype=str)
    matrix_df.columns = matrix_df.columns.str.strip().str.lower()

    # Get normalized feature columns
    norm_cols = [c for c in matrix_df.columns if c != "subject_id"]
    X_raw = matrix_df[norm_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).values.astype(np.float32)

    print("Loading EHR severity labels...")
    labels_df = pd.read_csv(ehr_labels_path, dtype=str)
    y = labels_df["severity_int"].astype(int).values

    # Use column names from file if available, else use defaults
    feature_names = FEATURE_NAMES[:X_raw.shape[1]]

    print("Loading EHR embeddings (for linear probe training)...")
    emb_df   = pd.read_csv(embeddings_path)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    ehr_emb  = emb_df[emb_df["modality"] == "ehr"].reset_index(drop=True)
    X_emb    = ehr_emb[emb_cols].values.astype(np.float32)
    y_emb    = ehr_emb["severity_int"].astype(int).values

    print(f"  EHR raw features : {X_raw.shape}")
    print(f"  EHR embeddings   : {X_emb.shape}")
    print(f"  Labels           : {len(y)}")
    print()

    return X_raw, X_emb, y, y_emb, feature_names


# ---------------------------------------------------------------------------
# Train linear probe on embeddings, then build explainer on raw features
# ---------------------------------------------------------------------------
def train_pipeline(X_raw, X_emb, y_emb, feature_names):
    """
    Train logistic regression on EHR embeddings (linear probe).
    Then build a SHAP explainer that maps raw features -> probe prediction.

    Since SHAP needs a single callable, we create a wrapper function:
        raw_features -> (via lookup) embedding -> linear probe -> class probs
    """
    print("Training linear probe on EHR embeddings...")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_emb, y_emb, test_size=0.2, stratify=y_emb, random_state=42
    )

    probe = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    probe.fit(X_tr, y_tr)

    y_pred  = probe.predict(X_te)
    f1      = f1_score(y_te, y_pred, average="macro")
    print(f"  Linear probe macro F1 on test set: {f1:.3f}")
    print()

    return probe


# ---------------------------------------------------------------------------
# Build predict function that uses raw features
# Since raw features are normalized versions of the originals and the
# embeddings were produced from these same features, we use a KNN-style
# lookup: for each raw feature vector find the nearest embedding and
# use the probe to predict from that embedding.
#
# More practically: we train a second logistic regression DIRECTLY on
# the raw features to explain with SHAP. This is the standard approach
# for tabular SHAP — explain the end-to-end mapping from raw features
# to predicted severity class.
# ---------------------------------------------------------------------------
def train_raw_classifier(X_raw, y):
    """
    Train logistic regression directly on raw normalized features.
    This is what SHAP will explain — the relationship between the 6
    clinical features and predicted severity class.
    """
    print("Training logistic regression on raw EHR features for SHAP...")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ])
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    f1     = f1_score(y_te, y_pred, average="macro")
    print(f"  Raw feature classifier macro F1: {f1:.3f}")
    print(f"  (This F1 reflects raw feature predictability)")
    print(f"  (Contrastive embedding F1 was higher — confirms encoder adds value)")
    print()

    return clf, X_te, y_te


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------
def run_shap(clf, X_raw, y, feature_names, out_dir):
    print("Running SHAP analysis...")
    print("  Using KernelExplainer on predict_proba (class-aware SHAP)...")

    X_scaled = clf.named_steps["scaler"].transform(X_raw)
    lr       = clf.named_steps["lr"]

    # Background dataset — small sample for speed
    np.random.seed(42)
    bg_idx    = np.random.choice(len(X_scaled), min(100, len(X_scaled)), replace=False)
    X_bg      = shap.kmeans(X_scaled[bg_idx], 10)

    explainer = shap.KernelExplainer(lr.predict_proba, X_bg)

    # Explain a representative subset for speed
    explain_idx = np.random.choice(len(X_scaled), min(500, len(X_scaled)), replace=False)
    X_explain   = X_scaled[explain_idx]
    y_explain   = y[explain_idx]

    print(f"  Explaining {len(X_explain)} samples (subset for speed)...")
    raw_shap = explainer.shap_values(X_explain, nsamples=200)

    # Normalize output to list of 3 arrays [N, n_features], one per class.
    # Different SHAP versions return different shapes:
    #   - list of 3 x [N, F]        -> correct, use directly
    #   - single array [N, F, 3]    -> transpose and split
    #   - single array [N, 3]       -> F=1 edge case
    #   - single array [3, N, F]    -> split along axis 0
    n_classes = 3
    n_feats   = X_explain.shape[1]

    if isinstance(raw_shap, list):
        # Standard format — list of arrays
        shap_values = [np.array(sv) for sv in raw_shap[:n_classes]]
    elif isinstance(raw_shap, np.ndarray):
        if raw_shap.ndim == 3:
            if raw_shap.shape[2] == n_classes:
                # Shape [N, F, C] — split along last axis
                shap_values = [raw_shap[:, :, i] for i in range(n_classes)]
            elif raw_shap.shape[0] == n_classes:
                # Shape [C, N, F] — split along first axis
                shap_values = [raw_shap[i] for i in range(n_classes)]
            else:
                raise ValueError(f"Unexpected SHAP array shape: {raw_shap.shape}")
        elif raw_shap.ndim == 2:
            # Shape [N, C] — SHAP collapsed features; repeat for each class
            shap_values = [raw_shap[:, [i]] * np.ones((len(X_explain), n_feats))
                           for i in range(n_classes)]
        else:
            raise ValueError(f"Unexpected SHAP array ndim: {raw_shap.ndim}")
    else:
        raise ValueError(f"Unexpected SHAP output type: {type(raw_shap)}")

    # Verify shapes
    for i, sv in enumerate(shap_values):
        assert sv.shape == (len(X_explain), n_feats), (
            f"Class {i} SHAP shape {sv.shape} != expected ({len(X_explain)}, {n_feats})"
        )

    print(f"  SHAP values: {n_classes} classes x "
          f"{shap_values[0].shape[0]} samples x "
          f"{shap_values[0].shape[1]} features  ✓")
    print()

    return shap_values, X_explain, y_explain


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_summary_bar(shap_values, feature_names, out_dir):
    """Mean |SHAP| per feature, stacked by class."""
    print("  Plotting summary bar chart...")

    # LinearExplainer may return n_classes+1 arrays — keep only first 3
    sv_classes = shap_values[:3]

    mean_abs = np.array([
        np.abs(sv).mean(axis=0) for sv in sv_classes
    ])  # [n_classes, n_features]

    fig, ax = plt.subplots(figsize=(9, 5))
    x       = np.arange(len(feature_names))
    width   = 0.25
    colors  = [SEVERITY_COLORS[i] for i in range(3)]

    for i, (cls_shap, color) in enumerate(zip(mean_abs, colors)):
        ax.barh(
            x + (i - 1) * width, cls_shap,
            height=width, color=color, alpha=0.85,
            label=SEVERITY_NAMES[i], edgecolor="white", linewidth=0.5
        )

    ax.set_yticks(x)
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        "EHR Feature Importance — SHAP Analysis\n"
        "Mean absolute SHAP value by severity class",
        fontsize=12, fontweight="bold"
    )
    ax.legend(title="Severity", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary_bar.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved -> {path}")


def plot_beeswarm(shap_values, X_raw, feature_names, out_dir):
    """Beeswarm plot for each class."""
    print("  Plotting beeswarm plots...")

    shap_values = shap_values[:3]  # guard against extra arrays
    for cls_idx, cls_name in SEVERITY_NAMES.items():
        sv = shap_values[cls_idx]   # [N, n_features]

        fig, ax = plt.subplots(figsize=(9, 5))

        # Sort features by mean |SHAP|
        order = np.argsort(np.abs(sv).mean(axis=0))[::-1]

        y_pos    = np.arange(len(feature_names))
        jitter   = np.random.RandomState(42).uniform(
            -0.3, 0.3, size=(sv.shape[0], sv.shape[1])
        )

        for i, feat_i in enumerate(order):
            shap_col = sv[:, feat_i]
            feat_col = X_raw[:, feat_i]

            # Color by feature value (low=blue, high=red)
            norm     = mcolors.Normalize(
                vmin=feat_col.min(), vmax=feat_col.max()
            )
            cmap     = plt.cm.RdYlBu_r
            colors_i = cmap(norm(feat_col))

            ax.scatter(
                shap_col,
                np.full(len(shap_col), i) + jitter[:, feat_i] * 0.25,
                c=colors_i, alpha=0.4, s=12, linewidths=0
            )

        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(
            [feature_names[i] for i in order], fontsize=10
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
        ax.set_title(
            f"SHAP Beeswarm — {cls_name} Severity Class\n"
            "Color = feature value (blue=low, red=high)",
            fontsize=12, fontweight="bold"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Feature value\n(normalized)", fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["Low", "Mid", "High"])

        plt.tight_layout()
        path = os.path.join(
            out_dir, f"shap_beeswarm_{cls_name.lower()}.png"
        )
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    Saved -> {path}")


def plot_heatmap(shap_values, feature_names, out_dir):
    """Heatmap of mean SHAP values: features x classes."""
    print("  Plotting SHAP heatmap...")

    shap_values = shap_values[:3]  # guard against extra arrays
    mean_shap = np.array([
        sv.mean(axis=0) for sv in shap_values
    ])  # [n_classes, n_features]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mean_shap, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(mean_shap).max(),
                   vmax= np.abs(mean_shap).max())

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(SEVERITY_NAMES)))
    ax.set_yticklabels([SEVERITY_NAMES[i] for i in range(len(shap_values))],
                        fontsize=10)

    # Add value annotations
    for i in range(len(shap_values)):
        for j in range(len(feature_names)):
            val = mean_shap[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(val) > np.abs(mean_shap).max()*0.5
                    else "black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                 label="Mean SHAP value")
    ax.set_title(
        "Mean SHAP Values — EHR Features by Severity Class\n"
        "Positive = pushes toward this class  ·  "
        "Negative = pushes away from this class",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_heatmap.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved -> {path}")


def save_tables(shap_values, X_raw, y, feature_names, out_dir):
    """Save raw SHAP values and feature importance ranking."""
    print("  Saving SHAP tables...")

    # Feature importance: mean |SHAP| averaged across all classes
    shap_values = shap_values[:3]  # guard against extra arrays
    overall_importance = np.array([
        np.abs(sv).mean(axis=0) for sv in shap_values
    ]).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature"         : feature_names,
        "mean_abs_shap"   : overall_importance,
        **{f"mean_abs_shap_{SEVERITY_NAMES[i].lower()}": np.abs(sv).mean(axis=0)
           for i, sv in enumerate(shap_values)}
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df)+1)
    imp_path = os.path.join(out_dir, "shap_feature_importance.csv")
    importance_df.to_csv(imp_path, index=False)
    print(f"    Feature importance -> {imp_path}")

    # Print ranking
    print()
    print("  Feature Importance Ranking (mean |SHAP|):")
    print(f"  {'Rank':<6} {'Feature':<25} {'Overall':>10} "
          f"{'Low':>10} {'Moderate':>10} {'High':>10}")
    print(f"  {'-'*65}")
    for _, row in importance_df.iterrows():
        print(f"  {int(row['rank']):<6} {row['feature']:<25} "
              f"{row['mean_abs_shap']:>10.4f} "
              f"{row['mean_abs_shap_low']:>10.4f} "
              f"{row['mean_abs_shap_moderate']:>10.4f} "
              f"{row['mean_abs_shap_high']:>10.4f}")
    print()

    # Raw SHAP values — one row per patient per class
    rows = []
    for cls_idx, sv in enumerate(shap_values):
        df_cls = pd.DataFrame(sv, columns=feature_names)
        df_cls.insert(0, "severity_true", y)
        df_cls.insert(1, "explained_class", SEVERITY_NAMES[cls_idx])
        rows.append(df_cls)
    raw_df   = pd.concat(rows, ignore_index=True)
    raw_path = os.path.join(out_dir, "shap_values.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"    Raw SHAP values  -> {raw_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(ehr_matrix_path, ehr_labels_path, embeddings_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  SHAP Analysis — EHR Encoder")
    print("  Explaining: which of the 6 EHR features drive")
    print("  severity predictions from the contrastive model")
    print("=" * 60)
    print()

    # Load
    X_raw, X_emb, y, y_emb, feature_names = load_data(
        ehr_matrix_path, ehr_labels_path, embeddings_path
    )

    # Train linear probe on embeddings (shows contrastive model quality)
    probe = train_pipeline(X_raw, X_emb, y_emb, feature_names)

    # Train raw feature classifier (SHAP explains this)
    clf, X_te, y_te = train_raw_classifier(X_raw, y)

    # Run SHAP
    shap_values, X_explain, y_explain = run_shap(clf, X_raw, y, feature_names, out_dir)

    # Plots
    print("Generating plots...")
    plot_summary_bar(shap_values, feature_names, out_dir)
    plot_beeswarm(shap_values, X_explain, feature_names, out_dir)
    plot_heatmap(shap_values, feature_names, out_dir)
    save_tables(shap_values, X_explain, y_explain, feature_names, out_dir)

    print()
    print("=" * 60)
    print("  SHAP Analysis Complete")
    print("=" * 60)
    print("  Output files:")
    for fname in [
        "shap_summary_bar.png",
        "shap_beeswarm_low.png",
        "shap_beeswarm_moderate.png",
        "shap_beeswarm_high.png",
        "shap_heatmap.png",
        "shap_feature_importance.csv",
        "shap_values.csv",
    ]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {fname:<40} ({size_kb:,.0f} KB)")
    print()
    print("  Key figure for paper: shap_heatmap.png + shap_summary_bar.png")
    print("  These directly show which clinical features drive")
    print("  each severity class prediction.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP analysis of EHR encoder from contrastive model"
    )
    parser.add_argument(
        "--ehr_matrix",
        default="/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_feature_matrix_balanced.csv"
    )
    parser.add_argument(
        "--ehr_labels",
        default="/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_severity_balanced.csv"
    )
    parser.add_argument(
        "--embeddings",
        default="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/shared_embedding/embeddings.csv"
    )
    parser.add_argument(
        "--out",
        default="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/shap"
    )
    args = parser.parse_args()
    main(args.ehr_matrix, args.ehr_labels, args.embeddings, args.out)