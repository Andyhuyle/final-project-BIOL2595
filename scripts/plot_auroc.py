"""
plot_auroc.py

Plots One-vs-Rest ROC curves for all four models in a 2x2 grid:
    Image Only      | EHR Only
    Late Fusion     | Contrastive Retrieval

Each panel shows Low / Moderate / High curves + chance diagonal.
Moderate is highlighted in red across all panels — the clinical story.

Prerequisites
-------------
Run late_fusion_baseline.py   -> writes y_true/y_prob_img/ehr/fused.npy
Run multimodal_contrastive.py -> writes y_true/y_prob_contrastive.npy

Usage
-----
    python plot_auroc.py

Output
------
    auroc_curves.png  — 2x2 grid saved to LATE_FUSION_DIR
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ── Config ──────────────────────────────────────────────────────────────────
LATE_FUSION_DIR = "/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/late_fusion"
CONTRASTIVE_DIR = "/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/shared_embedding"
OUT_PATH        = os.path.join(LATE_FUSION_DIR, "auroc_curves.png")

CLASSES     = [0, 1, 2]
CLASS_NAMES = ["Low", "Moderate", "High"]

CLASS_COLORS = {
    "Low":      "#0D9488",   # teal
    "Moderate": "#E11D48",   # red — clinical priority
    "High":     "#1E40AF",   # navy
}
CLASS_LINESTYLES = {
    "Low":      "-",
    "Moderate": "--",
    "High":     "-.",
}

# Grid order: top-left, top-right, bottom-left, bottom-right
MODELS = [
    ("Image Only",            LATE_FUSION_DIR, "img",         None),
    ("EHR Only",              LATE_FUSION_DIR, "ehr",         None),
    ("Late Fusion",           LATE_FUSION_DIR, "fused",       None),
    ("Contrastive Retrieval", CONTRASTIVE_DIR, "contrastive",
     "Scores are similarity-derived, not softmax.\nAUROC comparison valid — ranking-based metric."),
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_arrays(directory, key):
    y_true = np.load(os.path.join(directory, f"y_true_{key}.npy"))
    y_prob = np.load(os.path.join(directory, f"y_prob_{key}.npy"))
    return y_true, y_prob


def compute_roc(y_true, y_prob):
    y_bin = label_binarize(y_true, classes=CLASSES)
    out   = {}
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        out[name]   = (fpr, tpr, auc(fpr, tpr))
    return out


def plot_panel(ax, roc_data, title, note=None):
    for cls_name, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(
            fpr, tpr,
            color     = CLASS_COLORS[cls_name],
            linestyle = CLASS_LINESTYLES[cls_name],
            linewidth = 2.4,
            label     = f"{cls_name}  (AUC = {roc_auc:.3f})",
        )

    # Chance diagonal
    ax.plot(
        [0, 1], [0, 1],
        color="#94A3B8", linestyle=":", linewidth=1.2,
        label="Chance  (AUC = 0.500)",
    )

    # Macro AUC box — bottom right of each panel
    macro_auc = np.mean([v[2] for v in roc_data.values()])
    ax.text(
        0.97, 0.05,
        f"Macro AUC = {macro_auc:.3f}",
        transform  = ax.transAxes,
        fontsize   = 10,
        color      = "#1E293B",
        ha         = "right",
        va         = "bottom",
        bbox       = dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#CBD5E1", alpha=0.92),
    )

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

    if note:
        ax.text(
            0.5, -0.16, note,
            transform  = ax.transAxes,
            fontsize   = 8.5,
            color      = "#64748B",
            style      = "italic",
            ha         = "center",
            va         = "top",
        )

    ax.legend(loc="lower right", fontsize=10, framealpha=0.92,
              bbox_to_anchor=(1.0, 0.14))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", color="#E2E8F0", linewidth=0.8)
    ax.tick_params(labelsize=10)


# ── Main ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 12))
axes_flat = axes.flatten()   # top-left, top-right, bottom-left, bottom-right

fig.suptitle(
    "One-vs-Rest ROC Curves — Prostate Cancer Severity Classification\n"
    "Low (teal)  ·  Moderate (red dashed)  ·  High (navy dash-dot)",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

for ax, (label, directory, key, note) in zip(axes_flat, MODELS):
    try:
        y_true, y_prob = load_arrays(directory, key)
    except FileNotFoundError as e:
        ax.text(
            0.5, 0.5,
            f"File not found:\n{os.path.basename(str(e))}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="#E11D48",
        )
        ax.set_title(label, fontsize=13, fontweight="bold")
        continue

    plot_panel(ax, compute_roc(y_true, y_prob), label, note=note)

# Only left-column panels need y-axis labels
axes[0, 1].set_ylabel("")
axes[1, 1].set_ylabel("")

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUT_PATH}")