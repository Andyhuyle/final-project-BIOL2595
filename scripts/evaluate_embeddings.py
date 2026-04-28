"""
evaluate_embeddings.py

Evaluates the quality of learned embeddings from multimodal_contrastive.py
using a linear probe classifier.

This is the standard evaluation protocol for contrastive learning:
    1. Load frozen embeddings (no model retraining)
    2. Train a simple logistic regression on top
    3. Report F1, precision, recall, confusion matrix, AUROC

Inputs:
    embeddings.csv   output of multimodal_contrastive.py

Usage:
    python evaluate_embeddings.py \
        --embeddings /oscar/data/class/biol1595_2595/students/hgle/outputs/embeddings.csv \
        --out        /oscar/data/class/biol1595_2595/students/hgle/outputs
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed on Oscar
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize

SEVERITY_NAMES = {0: "low", 1: "moderate", 2: "high"}

# ---------------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------------
def load_embeddings(path):
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    img_df = df[df["modality"] == "image"].reset_index(drop=True)
    ehr_df = df[df["modality"] == "ehr"].reset_index(drop=True)

    X_img = img_df[emb_cols].values.astype(np.float32)
    y_img = img_df["severity_int"].values.astype(int)

    X_ehr = ehr_df[emb_cols].values.astype(np.float32)
    y_ehr = ehr_df["severity_int"].values.astype(int)

    return X_img, y_img, X_ehr, y_ehr


# ---------------------------------------------------------------------------
# Linear probe evaluation with cross-validation
# ---------------------------------------------------------------------------
def evaluate_modality(X, y, modality_name, out_dir):
    print(f"\n{'='*55}")
    print(f"  Linear Probe Evaluation — {modality_name}")
    print(f"{'='*55}")
    print(f"  Samples   : {len(X):,}")
    print(f"  Features  : {X.shape[1]}")
    print(f"  Classes   : {np.unique(y).tolist()}")
    print()

    # 5-fold stratified cross-validation
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf     = LogisticRegression(max_iter=1000, random_state=42, C=1.0)

    all_y_true, all_y_pred, all_y_prob = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        fold_f1 = f1_score(y_test, y_pred, average="macro")
        print(f"  Fold {fold+1} macro F1: {fold_f1:.3f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    classes = sorted(np.unique(all_y_true))
    labels  = [SEVERITY_NAMES[c] for c in classes]

    print()
    print("  Overall Metrics (macro-averaged):")
    print(f"  {'Metric':<20} {'Value':>8}")
    print(f"  {'-'*30}")

    macro_f1  = f1_score(all_y_true, all_y_pred, average="macro")
    macro_pre = precision_score(all_y_true, all_y_pred, average="macro")
    macro_rec = recall_score(all_y_true, all_y_pred, average="macro")

    # AUROC — one-vs-rest for multiclass
    y_bin    = label_binarize(all_y_true, classes=classes)
    macro_auc = roc_auc_score(y_bin, all_y_prob, average="macro",
                               multi_class="ovr")

    for name, val in [
        ("Macro F1",        macro_f1),
        ("Macro Precision", macro_pre),
        ("Macro Recall",    macro_rec),
        ("Macro AUROC",     macro_auc),
    ]:
        print(f"  {name:<20} {val:>8.3f}")

    print()
    print("  Per-Class Report:")
    print(classification_report(all_y_true, all_y_pred,
                                 target_names=labels, digits=3))

    # -----------------------------------------------------------------------
    # Confusion matrix plot
    # -----------------------------------------------------------------------
    cm   = confusion_matrix(all_y_true, all_y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {modality_name}\n"
                 f"Macro F1={macro_f1:.3f}  AUROC={macro_auc:.3f}")
    plt.tight_layout()

    cm_path = os.path.join(out_dir, f"confusion_matrix_{modality_name.lower().replace(' ', '_')}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {cm_path}")

    return {
        "modality"  : modality_name,
        "macro_f1"  : macro_f1,
        "macro_pre" : macro_pre,
        "macro_rec" : macro_rec,
        "macro_auc" : macro_auc,
    }


# ---------------------------------------------------------------------------
# Cross-modal retrieval metrics
# ---------------------------------------------------------------------------
def cross_modal_retrieval_metrics(X_img, y_img, X_ehr, y_ehr, out_dir):
    print(f"\n{'='*55}")
    print("  Cross-Modal Retrieval Metrics")
    print(f"{'='*55}")

    # Cosine similarity: each image vs all EHR embeddings
    X_img_n = X_img / (np.linalg.norm(X_img, axis=1, keepdims=True) + 1e-8)
    X_ehr_n = X_ehr / (np.linalg.norm(X_ehr, axis=1, keepdims=True) + 1e-8)

    sim      = X_img_n @ X_ehr_n.T       # [N_img, N_ehr]
    top1_idx = sim.argmax(axis=1)
    top1_sev = y_ehr[top1_idx]
    correct  = (top1_sev == y_img)

    overall_acc = correct.mean()
    print(f"  Overall top-1 accuracy : {overall_acc*100:.1f}%  (chance=33.3%)")
    print()
    print(f"  {'Class':<12} {'Accuracy':>10} {'N':>8}")
    print(f"  {'-'*32}")

    for cls, name in SEVERITY_NAMES.items():
        mask = (y_img == cls)
        if mask.sum() > 0:
            acc = correct[mask].mean()
            print(f"  {name:<12} {acc*100:>9.1f}%  {mask.sum():>7}")

    # Per-class F1 on retrieval predictions
    f1_retrieval = f1_score(y_img, top1_sev, average="macro")
    print(f"\n  Retrieval macro F1 : {f1_retrieval:.3f}")

    return overall_acc, f1_retrieval


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(embeddings_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading embeddings...")
    X_img, y_img, X_ehr, y_ehr = load_embeddings(embeddings_path)
    print(f"  Image embeddings : {X_img.shape}")
    print(f"  EHR embeddings   : {X_ehr.shape}")

    # Linear probe on image embeddings
    img_results = evaluate_modality(X_img, y_img, "Image Encoder", out_dir)

    # Linear probe on EHR embeddings
    ehr_results = evaluate_modality(X_ehr, y_ehr, "EHR Encoder", out_dir)

    # Cross-modal retrieval
    ret_acc, ret_f1 = cross_modal_retrieval_metrics(
        X_img, y_img, X_ehr, y_ehr, out_dir
    )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("  Final Results Summary")
    print(f"{'='*55}")
    print(f"  {'Metric':<35} {'Value':>8}")
    print(f"  {'-'*45}")
    print(f"  {'Image encoder macro F1':<35} {img_results['macro_f1']:>8.3f}")
    print(f"  {'Image encoder AUROC':<35} {img_results['macro_auc']:>8.3f}")
    print(f"  {'EHR encoder macro F1':<35} {ehr_results['macro_f1']:>8.3f}")
    print(f"  {'EHR encoder AUROC':<35} {ehr_results['macro_auc']:>8.3f}")
    print(f"  {'Cross-modal retrieval accuracy':<35} {ret_acc*100:>7.1f}%")
    print(f"  {'Cross-modal retrieval macro F1':<35} {ret_f1:>8.3f}")
    print(f"  {'Chance baseline (3 classes)':<35} {'33.3%':>8}")
    print()

    # Save summary CSV
    summary = pd.DataFrame([
        {"metric": "image_macro_f1",          "value": img_results["macro_f1"]},
        {"metric": "image_macro_precision",    "value": img_results["macro_pre"]},
        {"metric": "image_macro_recall",       "value": img_results["macro_rec"]},
        {"metric": "image_macro_auroc",        "value": img_results["macro_auc"]},
        {"metric": "ehr_macro_f1",             "value": ehr_results["macro_f1"]},
        {"metric": "ehr_macro_precision",      "value": ehr_results["macro_pre"]},
        {"metric": "ehr_macro_recall",         "value": ehr_results["macro_rec"]},
        {"metric": "ehr_macro_auroc",          "value": ehr_results["macro_auc"]},
        {"metric": "retrieval_accuracy",       "value": float(ret_acc)},
        {"metric": "retrieval_macro_f1",       "value": ret_f1},
        {"metric": "chance_baseline",          "value": 1/3},
    ])
    summary_path = os.path.join(out_dir, "results_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Results saved -> {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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