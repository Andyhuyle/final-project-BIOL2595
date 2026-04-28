"""
umap_visualization.py

Generates UMAP visualizations of the shared latent space learned by the
weakly supervised contrastive model.

Produces four plots:
    1. Image embeddings colored by severity
    2. EHR embeddings colored by severity
    3. Both modalities together — colored by severity
    4. Both modalities together — colored by modality (image vs EHR)

Plot 3 and 4 together are the key figure for the paper: if the contrastive
alignment worked, severity clusters should overlap across modalities in plot 3,
and image/EHR points should be interleaved within clusters in plot 4.

Usage:
    python umap_visualization.py \
        --embeddings /oscar/.../outputs/embeddings.csv \
        --out        /oscar/.../outputs
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import umap
except ImportError:
    raise ImportError(
        "umap-learn not installed. Run: pip install umap-learn"
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEVERITY_COLORS  = {0: "#2196F3", 1: "#FF9800", 2: "#F44336"}   # blue, orange, red
SEVERITY_NAMES   = {0: "Low",     1: "Moderate", 2: "High"}
MODALITY_COLORS  = {"image": "#7B1FA2", "ehr": "#00897B"}        # purple, teal
MODALITY_MARKERS = {"image": "o",       "ehr": "^"}

UMAP_PARAMS = dict(
    n_neighbors  = 30,
    min_dist     = 0.1,
    n_components = 2,
    metric       = "cosine",   # cosine matches the contrastive loss metric
    random_state = 42,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_fig(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def make_severity_legend(ax):
    patches = [
        mpatches.Patch(color=SEVERITY_COLORS[s], label=SEVERITY_NAMES[s])
        for s in sorted(SEVERITY_NAMES)
    ]
    ax.legend(handles=patches, title="Severity", loc="upper right",
              framealpha=0.8, fontsize=9)


def make_modality_legend(ax):
    handles = [
        mpatches.Patch(color=MODALITY_COLORS[m], label=m.capitalize())
        for m in ["image", "ehr"]
    ]
    ax.legend(handles=handles, title="Modality", loc="upper right",
              framealpha=0.8, fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(embeddings_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load embeddings
    # -----------------------------------------------------------------------
    print("Loading embeddings...")
    df       = pd.read_csv(embeddings_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    img_df = df[df["modality"] == "image"].reset_index(drop=True)
    ehr_df = df[df["modality"] == "ehr"].reset_index(drop=True)

    X_img   = img_df[emb_cols].values.astype(np.float32)
    y_img   = img_df["severity_int"].values.astype(int)

    X_ehr   = ehr_df[emb_cols].values.astype(np.float32)
    y_ehr   = ehr_df["severity_int"].values.astype(int)

    X_all   = np.concatenate([X_img, X_ehr], axis=0)
    y_all   = np.concatenate([y_img, y_ehr], axis=0)
    mod_all = np.array(["image"] * len(X_img) + ["ehr"] * len(X_ehr))

    print(f"  Image embeddings : {X_img.shape}")
    print(f"  EHR embeddings   : {X_ehr.shape}")
    print(f"  Combined         : {X_all.shape}")
    print()

    # -----------------------------------------------------------------------
    # Fit UMAP — fit once on combined data so all plots share the same space
    # -----------------------------------------------------------------------
    print("Fitting UMAP on combined embeddings (this takes ~1-2 minutes)...")
    reducer    = umap.UMAP(**UMAP_PARAMS)
    emb_2d_all = reducer.fit_transform(X_all)

    emb_2d_img = emb_2d_all[:len(X_img)]
    emb_2d_ehr = emb_2d_all[len(X_img):]
    print(f"  UMAP output shape: {emb_2d_all.shape}")
    print()

    # -----------------------------------------------------------------------
    # Plot 1 — Image embeddings colored by severity
    # -----------------------------------------------------------------------
    print("Generating plots...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for sev in sorted(SEVERITY_NAMES):
        mask = (y_img == sev)
        ax.scatter(
            emb_2d_img[mask, 0], emb_2d_img[mask, 1],
            c=SEVERITY_COLORS[sev], label=SEVERITY_NAMES[sev],
            s=8, alpha=0.6, linewidths=0
        )
    make_severity_legend(ax)
    ax.set_title("UMAP — Image Embeddings by Severity\n"
                 "(PANDA histopathology, ResNet18 encoder)", fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    save_fig(fig, os.path.join(out_dir, "umap_image_severity.png"))

    # -----------------------------------------------------------------------
    # Plot 2 — EHR embeddings colored by severity
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for sev in sorted(SEVERITY_NAMES):
        mask = (y_ehr == sev)
        ax.scatter(
            emb_2d_ehr[mask, 0], emb_2d_ehr[mask, 1],
            c=SEVERITY_COLORS[sev], label=SEVERITY_NAMES[sev],
            s=8, alpha=0.6, linewidths=0
        )
    make_severity_legend(ax)
    ax.set_title("UMAP — EHR Embeddings by Severity\n"
                 "(MIMIC IV, PSA-derived labels)", fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    save_fig(fig, os.path.join(out_dir, "umap_ehr_severity.png"))

    # -----------------------------------------------------------------------
    # Plot 3 — Both modalities colored by severity
    # Key figure: clusters should overlap if alignment worked
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for sev in sorted(SEVERITY_NAMES):
        mask_img = (y_img == sev)
        mask_ehr = (y_ehr == sev)
        ax.scatter(
            emb_2d_img[mask_img, 0], emb_2d_img[mask_img, 1],
            c=SEVERITY_COLORS[sev], s=8, alpha=0.5,
            marker="o", linewidths=0
        )
        ax.scatter(
            emb_2d_ehr[mask_ehr, 0], emb_2d_ehr[mask_ehr, 1],
            c=SEVERITY_COLORS[sev], s=8, alpha=0.5,
            marker="^", linewidths=0
        )
    make_severity_legend(ax)

    # Add modality marker legend
    marker_handles = [
        plt.scatter([], [], marker="o", color="gray", s=20, label="Image"),
        plt.scatter([], [], marker="^", color="gray", s=20, label="EHR"),
    ]
    legend2 = ax.legend(handles=marker_handles, title="Modality",
                         loc="lower right", framealpha=0.8, fontsize=9)
    ax.add_artist(legend2)

    ax.set_title("UMAP — Shared Latent Space (Both Modalities) by Severity\n"
                 "Circle=Image  Triangle=EHR", fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    save_fig(fig, os.path.join(out_dir, "umap_combined_severity.png"))

    # -----------------------------------------------------------------------
    # Plot 4 — Both modalities colored by modality
    # Key figure: image and EHR should be interleaved within severity clusters
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        emb_2d_img[:, 0], emb_2d_img[:, 1],
        c=MODALITY_COLORS["image"], s=8, alpha=0.5,
        marker="o", label="Image", linewidths=0
    )
    ax.scatter(
        emb_2d_ehr[:, 0], emb_2d_ehr[:, 1],
        c=MODALITY_COLORS["ehr"], s=8, alpha=0.5,
        marker="^", label="EHR", linewidths=0
    )
    make_modality_legend(ax)
    ax.set_title("UMAP — Shared Latent Space by Modality\n"
                 "Interleaving indicates successful cross-modal alignment", fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    save_fig(fig, os.path.join(out_dir, "umap_combined_modality.png"))

    # -----------------------------------------------------------------------
    # Plot 5 — 2x2 panel figure for paper
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Shared Latent Space — Weakly Supervised Multimodal Contrastive Learning\n"
        "Prostate Cancer Severity (PANDA Images + MIMIC IV EHR)",
        fontsize=13, fontweight="bold"
    )

    panels = [
        (axes[0, 0], emb_2d_img, y_img,
         "Image Embeddings by Severity", "severity"),
        (axes[0, 1], emb_2d_ehr, y_ehr,
         "EHR Embeddings by Severity",   "severity"),
        (axes[1, 0], emb_2d_all, y_all,
         "Both Modalities by Severity",  "severity"),
        (axes[1, 1], emb_2d_all, mod_all,
         "Both Modalities by Modality",  "modality"),
    ]

    for ax, coords, labels, title, color_by in panels:
        if color_by == "severity":
            unique = sorted(np.unique(labels))
            for val in unique:
                mask = (labels == val)
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=SEVERITY_COLORS[val], s=5,
                           alpha=0.5, linewidths=0,
                           label=SEVERITY_NAMES[val])
            ax.legend(title="Severity", fontsize=8, markerscale=2)
        else:
            for mod in ["image", "ehr"]:
                mask = (labels == mod)
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=MODALITY_COLORS[mod], s=5,
                           alpha=0.5, linewidths=0,
                           label=mod.capitalize(),
                           marker="o" if mod == "image" else "^")
            ax.legend(title="Modality", fontsize=8, markerscale=2)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    panel_path = os.path.join(out_dir, "umap_panel_figure.png")
    save_fig(fig, panel_path)

    print()
    print("=" * 55)
    print("  All UMAP plots saved:")
    print("=" * 55)
    for fname in [
        "umap_image_severity.png",
        "umap_ehr_severity.png",
        "umap_combined_severity.png",
        "umap_combined_modality.png",
        "umap_panel_figure.png",
    ]:
        fpath   = os.path.join(out_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname:<40} ({size_kb:,.0f} KB)")

    print()
    print("  Key figure for paper: umap_panel_figure.png")
    print("  Key interpretation:")
    print("  - Plot 3 (combined severity): severity clusters should overlap")
    print("    across modalities if contrastive alignment worked")
    print("  - Plot 4 (combined modality): image/EHR points should be")
    print("    interleaved within clusters, not separated by modality")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UMAP visualization of contrastive learning embeddings"
    )
    parser.add_argument(
        "--embeddings",
        default="/oscar/data/class/biol1595_2595/students/hgle/outputs/embeddings.csv"
    )
    parser.add_argument(
        "--out",
        default="/oscar/data/class/biol1595_2595/students/hgle/outputs"
    )
    args = parser.parse_args()
    main(args.embeddings, args.out)