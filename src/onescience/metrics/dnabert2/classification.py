"""Metrics used by DNABERT-2 GUE and custom classification tasks."""

import numpy as np
from sklearn import metrics


def classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calculate the metric set used by the official fine-tuning script."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    valid = labels != -100
    predictions = predictions[valid]
    labels = labels[valid]
    if labels.size == 0:
        raise ValueError("No valid labels were provided")
    return {
        "accuracy": float(metrics.accuracy_score(labels, predictions)),
        "f1": float(metrics.f1_score(labels, predictions, average="macro", zero_division=0)),
        "matthews_correlation": float(metrics.matthews_corrcoef(labels, predictions)),
        "precision": float(
            metrics.precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall": float(metrics.recall_score(labels, predictions, average="macro", zero_division=0)),
    }
