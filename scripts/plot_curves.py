"""
plot_training_curves.py

Plots training loss curves for:
    1. Contrastive model (from multimodal_model.pt checkpoint)
    2. Late fusion baseline image + EHR models (re-runs training
       with loss logging, or reads from saved log files if available)

Usage:
    python plot_training_curves.py \
        --contrastive_ckpt /oscar/.../outputs/shared_embedding/multimodal_model.pt \
        --contrastive_log  /oscar/.../logs/multimodal_pca_*.log \
        --late_fusion_log  /oscar/.../logs/late_fusion_*.log \
        --out              /oscar/.../outputs
"""

import argparse
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
C_CONTRASTIVE = "#1565C0"   # blue
C_IMG         = "#7B1FA2"   # purple
C_EHR         = "#00897B"   # teal
C_FUSION      = "#E65100"   # orange
C_CHANCE      = "#9E9E9E"   # gray


# ---------------------------------------------------------------------------
# Load loss history from checkpoint
# ---------------------------------------------------------------------------
def load_from_checkpoint(ckpt_path):
    """Load training loss history from a saved .pt checkpoint."""
    if not HAS_TORCH:
        return None
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: Checkpoint not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt.get("history", None)
    if history is None:
        print(f"  WARNING: No 'history' key in checkpoint {ckpt_path}")
    return history


# ---------------------------------------------------------------------------
# Parse loss from SLURM log file
# Matches lines like: "Epoch  1 | Avg Loss: 3.5176 | LR: 9.76e-05"
# or:                 "    Epoch  1 | Loss: 0.8432 | LR: ..."
# ---------------------------------------------------------------------------
def parse_log_file(log_path, pattern=None):
    """
    Parse epoch loss values from a SLURM log file.
    Returns list of (epoch, loss) tuples.
    """
    if not os.path.exists(log_path):
        print(f"  WARNING: Log file not found: {log_path}")
        return []

    if pattern is None:
        # Matches both "Avg Loss:" and plain "Loss:"
        pattern = r"Epoch\s+(\d+).*?Loss:\s+([\d.]+)"

    results = []
    with open(log_path, "r") as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                epoch = int(m.group(1))
                loss  = float(m.group(2))
                results.append((epoch, loss))

    return results


def find_latest_log(log_dir, prefix):
    """Find the most recently modified log matching a prefix pattern."""
    pattern = os.path.join(log_dir, f"{prefix}*.log")
    files   = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Smooth loss curve with exponential moving average
# ---------------------------------------------------------------------------
def smooth(values, alpha=0.3):
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    return smoothed


# ---------------------------------------------------------------------------
# Plot 1: Contrastive model training curve
# ---------------------------------------------------------------------------
def plot_contrastive(history, out_dir):
    if not history:
        print("  No contrastive history to plot.")
        return

    epochs = list(range(1, len(history) + 1))
    losses = list(history)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Raw loss
    ax.plot(epochs, losses, color=C_CONTRASTIVE, alpha=0.35,
            linewidth=1, label="Loss (raw)")

    # Smoothed
    if len(losses) > 3:
        ax.plot(epochs, smooth(losses, alpha=0.4),
                color=C_CONTRASTIVE, linewidth=2,
                label="Loss (smoothed)", zorder=5)

    # Annotate final value
    ax.annotate(
        f"Final: {losses[-1]:.4f}",
        xy=(epochs[-1], losses[-1]),
        xytext=(-40, 12), textcoords="offset points",
        fontsize=9, color=C_CONTRASTIVE,
        arrowprops=dict(arrowstyle="->", color=C_CONTRASTIVE, lw=0.8)
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Supervised Contrastive Loss", fontsize=11)
    ax.set_title(
        "Contrastive Model Training Curve\n"
        "ResNet18 (image) + MLP (EHR) — shared latent space",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curve_contrastive.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Plot 2: Late fusion training curves (image + EHR separately)
# ---------------------------------------------------------------------------
def plot_late_fusion(img_history, ehr_history, out_dir):
    if not img_history and not ehr_history:
        print("  No late fusion history to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Late Fusion Baseline — Training Curves\n"
        "Image and EHR models trained independently with cross-entropy loss",
        fontsize=12, fontweight="bold"
    )

    for ax, history, color, label in [
        (axes[0], img_history, C_IMG, "Image classifier (ResNet18 + CE)"),
        (axes[1], ehr_history, C_EHR, "EHR classifier (MLP + CE)"),
    ]:
        if not history:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(label, fontsize=10)
            continue

        epochs = list(range(1, len(history) + 1))
        losses = list(history)

        ax.plot(epochs, losses, color=color, alpha=0.35,
                linewidth=1, label="Loss (raw)")
        if len(losses) > 3:
            ax.plot(epochs, smooth(losses, alpha=0.4),
                    color=color, linewidth=2, label="Loss (smoothed)", zorder=5)

        ax.annotate(
            f"Final: {losses[-1]:.4f}",
            xy=(epochs[-1], losses[-1]),
            xytext=(-40, 12), textcoords="offset points",
            fontsize=9, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8)
        )

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curve_late_fusion.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Plot 3: Combined comparison — all models on one figure
# ---------------------------------------------------------------------------
def plot_combined(contrastive_history, img_history, ehr_history, out_dir):
    histories = {
        "Contrastive (shared loss)": (contrastive_history, C_CONTRASTIVE, "-"),
        "Late fusion — Image (CE)":  (img_history,          C_IMG,         "--"),
        "Late fusion — EHR (CE)":    (ehr_history,          C_EHR,         "-."),
    }

    has_data = any(h for h, _, _ in histories.values())
    if not has_data:
        print("  No data for combined plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, (history, color, ls) in histories.items():
        if not history:
            continue
        epochs = list(range(1, len(history) + 1))
        losses = list(history)
        ax.plot(epochs, losses, color=color, alpha=0.25, linewidth=1, linestyle=ls)
        if len(losses) > 3:
            ax.plot(epochs, smooth(losses, alpha=0.4),
                    color=color, linewidth=2, linestyle=ls,
                    label=f"{label}  (final={losses[-1]:.3f})", zorder=5)
        else:
            ax.plot(epochs, losses, color=color, linewidth=2, linestyle=ls,
                    label=f"{label}  (final={losses[-1]:.3f})", zorder=5)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Training Loss", fontsize=11)
    ax.set_title(
        "Training Loss Comparison — Contrastive vs Late Fusion\n"
        "Note: contrastive loss and CE loss are not directly comparable",
        fontsize=11, fontweight="bold"
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curve_combined.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(contrastive_ckpt, contrastive_log, late_fusion_log, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 55)
    print("  Training Curve Plotter")
    print("=" * 55)
    print()

    # ── Contrastive history ──
    print("Loading contrastive model history...")
    contrastive_history = None

    # Try checkpoint first (most reliable)
    if contrastive_ckpt and os.path.exists(contrastive_ckpt):
        contrastive_history = load_from_checkpoint(contrastive_ckpt)
        if contrastive_history:
            print(f"  Loaded {len(contrastive_history)} epochs from checkpoint")

    # Fall back to log file
    if contrastive_history is None and contrastive_log:
        log_path = contrastive_log
        if "*" in log_path:
            matches = sorted(glob.glob(log_path))
            log_path = matches[-1] if matches else None
        if log_path:
            parsed = parse_log_file(log_path)
            if parsed:
                contrastive_history = [loss for _, loss in sorted(parsed)]
                print(f"  Parsed {len(contrastive_history)} epochs from log file")
    print()

    # ── Late fusion history ──
    print("Loading late fusion model history...")
    img_history = None
    ehr_history = None

    if late_fusion_log:
        log_path = late_fusion_log
        if "*" in log_path:
            matches = sorted(glob.glob(log_path))
            log_path = matches[-1] if matches else None

        if log_path and os.path.exists(log_path):
            # Image model epochs appear before EHR in the log
            # Split by looking for the "Step 1" and "Step 2" markers
            with open(log_path, "r") as f:
                content = f.read()

            # Split log into image and EHR sections
            img_section = ""
            ehr_section = ""

            if "Step 1/3" in content and "Step 2/3" in content:
                parts = content.split("Step 2/3")
                img_section = parts[0]
                ehr_section = parts[1] if len(parts) > 1 else ""
            else:
                img_section = content   # fallback: treat all as image

            img_parsed = parse_log_file.__wrapped__(img_section) if hasattr(
                parse_log_file, "__wrapped__") else []

            # Parse directly from string sections
            def parse_section(text):
                results = []
                for line in text.split("\n"):
                    m = re.search(r"Epoch\s+(\d+).*?Loss:\s+([\d.]+)", line)
                    if m:
                        results.append((int(m.group(1)), float(m.group(2))))
                return results

            img_parsed = parse_section(img_section)
            ehr_parsed = parse_section(ehr_section)

            if img_parsed:
                img_history = [loss for _, loss in sorted(img_parsed)]
                print(f"  Image model: {len(img_history)} epochs from log")
            if ehr_parsed:
                ehr_history = [loss for _, loss in sorted(ehr_parsed)]
                print(f"  EHR model  : {len(ehr_history)} epochs from log")
    print()

    # ── Generate plots ──
    print("Generating plots...")

    plot_contrastive(contrastive_history, out_dir)
    plot_late_fusion(img_history, ehr_history, out_dir)
    plot_combined(contrastive_history, img_history, ehr_history, out_dir)

    print()
    print("=" * 55)
    print("  Output files:")
    for fname in [
        "training_curve_contrastive.png",
        "training_curve_late_fusion.png",
        "training_curve_combined.png",
    ]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {fname:<45} ({size_kb:,.0f} KB)")
    print()
    print("  Key figure for paper: training_curve_combined.png")
    print("  Note in caption: contrastive loss and CE loss use different")
    print("  scales — compare trends, not absolute values.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training curves for contrastive and late fusion models"
    )
    parser.add_argument(
        "--contrastive_ckpt",
        default="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/contrastive_model/contrastive_model.pt",
        help="Path to contrastive model .pt checkpoint (contains history list)"
    )
    parser.add_argument(
        "--contrastive_log",
        default="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/multimodal_models/logs/contrastive_model_20260514_174707.log",
        help="Path (or glob) to contrastive model SLURM log file"
    )
    parser.add_argument(
        "--late_fusion_log",
        default="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/multimodal_models/logs/late_fusion_20260514_174819.log",
        help="Path (or glob) to late fusion SLURM log file"
    )
    parser.add_argument(
        "--out",
        default="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs",
        help="Output directory for plots"
    )
    args = parser.parse_args()
    main(args.contrastive_ckpt, args.contrastive_log,
         args.late_fusion_log, args.out)