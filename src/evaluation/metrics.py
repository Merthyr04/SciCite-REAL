"""Evaluation metrics used in the paper.

Primary metric: Macro-F1 over the 3 classes (robust to the class imbalance in
SciCite, where ``background`` dominates). We also report Accuracy to align with
the original SciCite work and enable direct comparison.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def micro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def confusion_matrix_df(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: List[str],
):
    """Confusion matrix as a DataFrame with label axes for report/plotting."""
    import pandas as pd

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    return pd.DataFrame(cm, index=label_names, columns=label_names)


def classification_report_dict(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: List[str],
):
    """Per-class precision/recall/F1 as a dict (JSON-serialisable)."""
    report = classification_report(
        y_true, y_pred, labels=list(range(len(label_names))), output_dict=True,
        target_names=label_names, zero_division=0,
    )
    return report