"""
evaluate.py — Classification metrics, confusion matrix, error analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)


def compute_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = ["Positive", "Negative", "Neutral"]
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision_weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "n_samples": len(y_true),
    }


def print_classification_report(y_true, y_pred, model_name="Model"):
    print(f"\n{'='*55}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, labels=None, output_dir="visualizations/output"):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def error_analysis(df, y_true_col="sentiment", y_pred_col="predicted_sentiment", text_col="text_original", n=10):
    """Show misclassified examples for review."""
    errors = df[df[y_true_col] != df[y_pred_col]].copy()
    print(f"\nMisclassified: {len(errors):,} / {len(df):,} ({len(errors)/len(df)*100:.1f}%)")
    if len(errors) > 0:
        sample = errors[[text_col, y_true_col, y_pred_col]].head(n)
        print(sample.to_string(index=False))
    return errors
