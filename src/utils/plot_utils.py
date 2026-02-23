"""
Publication-quality plotting utilities for CIC-IDS2017 DDoS detection thesis.

This module provides a consistent, robust and highly configurable set of
visualisation functions for:
  • Confusion matrices (raw & normalised)
  • ROC curves (single & multi-model)
  • Precision-Recall curves (single & multi-model)
  • Feature importance bar plots
  • Model comparison bar plots (F1-score)
  • Error scatter plots (FP/FN over time)

All functions follow the same strict standards as data_utils.py and eval_utils.py:
  • Full type hints
  • Comprehensive docstrings with examples
  • Input validation and informative error messages
  • Structured logging with verbose control
  • Unified style constants
  • Automatic figure saving and closing
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Sequence, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,  # <-- добавь эту строку
)

# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)

# ============================================================================
# STYLE CONSTANTS (centralised — change once, affects all plots)
# ============================================================================

PLOT_STYLES = {
    # Figure sizes
    "figsize_single": (6.0, 5.0),
    "figsize_comparison": (7.5, 6.0),
    "figsize_importance": (8.0, 6.5),
    "figsize_bar": (7.0, 5.0),
    "figsize_error": (10.0, 3.0),

    # Colours (colorblind-friendly palette + distinct hues)
    "palette": [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # lime
        "#17becf",  # cyan
    ],
    "roc_color": "#1f77b4",
    "pr_color": "#d62728",
    "fp_color": "#e74c3c",      # red for False Positives
    "fn_color": "#3498db",       # blue for False Negatives
    "random_line": "k--",
    "grid_alpha": 0.3,
    "cmap_cm": "Blues",
    "cmap_fp": "Reds",
    "cmap_fn": "Blues",

    # Fonts & text
    "font_size": 11,
    "title_size": 13,
    "label_size": 12,
    "legend_size": 10,
    "tick_size": 10,
    "annotation_size": 9,

    # Resolution
    "screen_dpi": 120,
    "save_dpi": 300,
}

CLASS_NAMES = ["BENIGN", "DDoS"]


# ============================================================================
# INTERNAL UTILITIES
# ============================================================================

def _setup_logging(verbose: bool) -> None:
    """Set logger level based on verbose flag."""
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


def _validate_binary_labels(
    y: np.ndarray,
    name: str = "y",
) -> None:
    """
    Validate that array contains only binary labels (0 and 1).
    
    Raises ValueError with informative message if validation fails.
    """
    unique_values = np.unique(y)
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"{name} contains unexpected values: {sorted(unique_values)}. "
            "Expected only 0 and 1."
        )


def _validate_probabilities(
    y_proba: np.ndarray,
    name: str = "y_proba",
) -> None:
    """
    Validate that array contains valid probabilities in [0, 1].
    
    Raises ValueError if values are outside valid range.
    """
    if np.any(y_proba < 0) or np.any(y_proba > 1):
        raise ValueError(
            f"{name} contains values outside [0, 1]: "
            f"min={np.min(y_proba):.4f}, max={np.max(y_proba):.4f}"
        )


def _validate_inputs(
    y_true: np.ndarray,
    y_pred_or_proba: np.ndarray,
    is_probability: bool = False,
) -> None:
    """
    Common validation for y_true and y_pred/y_proba arrays.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred_or_proba : np.ndarray
        Predicted labels or probabilities.
    is_probability : bool
        If True, validate as probabilities; otherwise as binary labels.
    
    Raises
    ------
    ValueError
        If arrays are empty, have mismatched lengths, or contain invalid values.
    """
    if len(y_true) == 0:
        raise ValueError("y_true is empty (0 samples)")
    
    if len(y_true) != len(y_pred_or_proba):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"second array has {len(y_pred_or_proba)} samples"
        )
    
    _validate_binary_labels(y_true, "y_true")
    
    if is_probability:
        _validate_probabilities(y_pred_or_proba, "y_proba")
    else:
        _validate_binary_labels(y_pred_or_proba, "y_pred")


def _save_figure(
    output_path: Optional[Union[str, Path]],
    dpi: int = PLOT_STYLES["save_dpi"],
) -> Optional[str]:
    """
    Save current figure to file and close it.
    
    Returns the path as string if saved, None otherwise.
    """
    if output_path is None:
        return None
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    
    logger.info(f"Saved plot: {output_path.name}")
    return str(output_path)


def _apply_common_style(ax: plt.Axes, title: str = "") -> None:
    """Apply common styling to axes: grid, title, font sizes."""
    ax.grid(True, alpha=PLOT_STYLES["grid_alpha"])
    
    if title:
        ax.set_title(title, fontsize=PLOT_STYLES["title_size"], fontweight="bold")
    
    ax.tick_params(labelsize=PLOT_STYLES["tick_size"])


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    *,
    normalize: bool = False,
    class_names: Optional[List[str]] = None,
    title: str = "",
    cmap: str = "",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Plot confusion matrix with cell annotations.

    Supports both raw counts and row-normalized (percentage) display.
    Row normalization shows what fraction of each true class was
    predicted as each class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 and 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 and 1).
    normalize : bool, default False
        If True, normalize by row (true label) and show percentages.
        If False, show raw counts.
    class_names : list of str, optional
        Class labels for axes. Default: ['BENIGN', 'DDoS'].
    title : str, default ""
        Custom plot title. If empty, auto-generates based on normalize flag.
    cmap : str, default ""
        Colormap name. If empty, uses PLOT_STYLES default.
    output_path : str or Path, optional
        If provided, saves the figure to this path.
    verbose : bool, default True
        If False, suppress INFO logging.

    Returns
    -------
    cm : np.ndarray of shape (2, 2)
        The confusion matrix (raw counts, even if normalized display).

    Raises
    ------
    ValueError
        If inputs are empty, have mismatched lengths,
        or contain values other than 0 and 1.

    Examples
    --------
    >>> cm = plot_confusion_matrix(y_test, y_pred, output_path="cm.png")
    >>> print(f"True Positives: {cm[1, 1]}")
    True Positives: 12543

    >>> # Normalized version
    >>> cm = plot_confusion_matrix(
    ...     y_test, y_pred, normalize=True, output_path="cm_norm.png"
    ... )
    """
    _setup_logging(verbose)
    
    # ─── Input validation ────────────────────────────────────────────────────
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _validate_inputs(y_true, y_pred, is_probability=False)
    
    if class_names is None:
        class_names = CLASS_NAMES
    
    if not cmap:
        cmap = PLOT_STYLES["cmap_cm"]
    
    # ─── Calculate confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape != (2, 2):
        raise ValueError(
            f"Expected binary confusion matrix (2x2), got shape {cm.shape}. "
            "Ensure both classes are present in predictions."
        )
    
    # ─── Prepare display values ──────────────────────────────────────────────
    if normalize:
        # Row normalization: divide each row by its sum
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(float) / row_sums
        fmt = ".2%"
        default_title = "Confusion Matrix (Normalized)"
    else:
        cm_display = cm
        fmt = "d"
        default_title = "Confusion Matrix"
    
    if not title:
        title = default_title
    
    # ─── Create figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=PLOT_STYLES["figsize_single"])
    
    # Plot heatmap
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=PLOT_STYLES["tick_size"])
    
    # ─── Add text annotations ────────────────────────────────────────────────
    # Determine text color based on background intensity
    thresh = cm_display.max() / 2.0
    
    for i in range(2):
        for j in range(2):
            # Format value
            if normalize:
                text = f"{cm_display[i, j]:.1%}"
            else:
                text = f"{cm_display[i, j]:,}"
            
            # Choose text color for visibility
            color = "white" if cm_display[i, j] > thresh else "black"
            
            ax.text(
                j, i, text,
                ha="center", va="center",
                color=color,
                fontsize=PLOT_STYLES["annotation_size"] + 2,
                fontweight="bold",
            )
    
    # ─── Labels and title ────────────────────────────────────────────────────
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=PLOT_STYLES["label_size"])
    ax.set_yticklabels(class_names, fontsize=PLOT_STYLES["label_size"])
    
    ax.set_xlabel("Predicted Label", fontsize=PLOT_STYLES["label_size"])
    ax.set_ylabel("True Label", fontsize=PLOT_STYLES["label_size"])
    ax.set_title(title, fontsize=PLOT_STYLES["title_size"], fontweight="bold")
    
    # ─── Save and return ─────────────────────────────────────────────────────
    _save_figure(output_path)
    
    if output_path is None:
        plt.show()
    
    logger.info(
        f"Confusion matrix: TN={cm[0,0]:,}, FP={cm[0,1]:,}, "
        f"FN={cm[1,0]:,}, TP={cm[1,1]:,}"
    )
    
    return cm


# ============================================================================
# ROC CURVE
# ============================================================================

def plot_roc_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_proba: Union[np.ndarray, pd.Series],
    *,
    label: str = "Model",
    title: str = "",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> float:
    """
    Plot ROC curve and return AUC.

    The ROC (Receiver Operating Characteristic) curve shows the trade-off
    between True Positive Rate and False Positive Rate at various
    classification thresholds.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 and 1).
    y_proba : array-like of shape (n_samples,)
        Predicted probabilities for the positive class (DDoS).
    label : str, default "Model"
        Label for the curve in legend.
    title : str, default ""
        Custom plot title. If empty, uses "ROC Curve".
    output_path : str or Path, optional
        If provided, saves the figure to this path.
    verbose : bool, default True
        If False, suppress INFO logging.

    Returns
    -------
    auc_value : float
        Area Under the ROC Curve.

    Raises
    ------
    ValueError
        If inputs are empty, have mismatched lengths,
        or y_proba contains values outside [0, 1].

    Examples
    --------
    >>> auc = plot_roc_curve(y_test, y_proba, label="RandomForest",
    ...                      output_path="roc.png")
    >>> print(f"AUC = {auc:.4f}")
    AUC = 0.9985
    """
    _setup_logging(verbose)
    
    # ─── Input validation ────────────────────────────────────────────────────
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    _validate_inputs(y_true, y_proba, is_probability=True)
    
    # ─── Calculate ROC curve and AUC ─────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_value = float(roc_auc_score(y_true, y_proba))
    
    logger.debug(f"ROC curve: {len(fpr)} points, AUC = {auc_value:.4f}")
    
    # ─── Create figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=PLOT_STYLES["figsize_single"])
    
    # Plot ROC curve
    ax.plot(
        fpr, tpr,
        color=PLOT_STYLES["roc_color"],
        linewidth=2,
        label=f"{label} (AUC = {auc_value:.4f})",
    )
    
    # Plot diagonal (random classifier)
    ax.plot(
        [0, 1], [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )
    
    # ─── Styling ─────────────────────────────────────────────────────────────
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=PLOT_STYLES["label_size"])
    ax.set_ylabel("True Positive Rate", fontsize=PLOT_STYLES["label_size"])
    
    if not title:
        title = "ROC Curve"
    _apply_common_style(ax, title)
    
    ax.legend(loc="lower right", fontsize=PLOT_STYLES["legend_size"])
    
    # ─── Save and return ─────────────────────────────────────────────────────
    _save_figure(output_path)
    
    if output_path is None:
        plt.show()
    
    logger.info(f"ROC AUC: {auc_value:.4f}")
    
    return auc_value


def plot_roc_curves_comparison(
    results: List[Dict[str, Any]],
    *,
    title: str = "",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Plot multiple ROC curves on one figure for model comparison.

    Each model's ROC curve is plotted with a different color, and the
    legend shows model names with their AUC values.

    Parameters
    ----------
    results : list of dict
        Each dict must contain:
        - 'name': str — model name for legend
        - 'y_true': array-like — true labels
        - 'y_proba': array-like — predicted probabilities
        
        Example:
        [
            {'name': 'RandomForest', 'y_true': y_test, 'y_proba': rf_proba},
            {'name': 'XGBoost', 'y_true': y_test, 'y_proba': xgb_proba},
        ]
    title : str, default ""
        Custom plot title. If empty, uses "ROC Curve Comparison".
    output_path : str or Path, optional
        If provided, saves the figure to this path.
    verbose : bool, default True
        If False, suppress INFO logging.

    Returns
    -------
    auc_dict : dict
        Dictionary mapping model names to AUC values.
        Example: {'RandomForest': 0.9985, 'XGBoost': 0.9991}

    Raises
    ------
    ValueError
        If results list is empty, or if any dict is missing required keys.
    """
    _setup_logging(verbose)
    
    # ─── Validation ──────────────────────────────────────────────────────────
    if not results:
        raise ValueError("results list is empty")
    
    required_keys = {"name", "y_true", "y_proba"}
    for i, res in enumerate(results):
        missing = required_keys - set(res.keys())
        if missing:
            raise ValueError(
                f"Result {i} is missing required keys: {missing}. "
                f"Found keys: {list(res.keys())}"
            )
    
    # ─── Create figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=PLOT_STYLES["figsize_comparison"])
    colors = PLOT_STYLES["palette"]
    auc_dict = {}
    
    for i, res in enumerate(results):
        name = res["name"]
        y_true = np.asarray(res["y_true"])
        y_proba = np.asarray(res["y_proba"])
        
        # Validate inputs
        _validate_inputs(y_true, y_proba, is_probability=True)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_value = float(roc_auc_score(y_true, y_proba))
        auc_dict[name] = auc_value
        
        # Plot
        color = colors[i % len(colors)]
        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f"{name} (AUC = {auc_value:.4f})",
        )
    
    # ─── Random classifier diagonal ──────────────────────────────────────────
    ax.plot(
        [0, 1], [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Random",
    )
    
    # ─── Styling ─────────────────────────────────────────────────────────────
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=PLOT_STYLES["label_size"])
    ax.set_ylabel("True Positive Rate", fontsize=PLOT_STYLES["label_size"])
    
    if not title:
        title = "ROC Curve Comparison"
    _apply_common_style(ax, title)
    
    ax.legend(loc="lower right", fontsize=PLOT_STYLES["legend_size"])
    
    # ─── Save and return ─────────────────────────────────────────────────────
    _save_figure(output_path)
    
    if output_path is None:
        plt.show()
    
    logger.info(f"Compared {len(results)} models: " + 
                ", ".join(f"{k}={v:.4f}" for k, v in auc_dict.items()))
    
    return auc_dict


# ============================================================================
# PRECISION-RECALL CURVE
# ============================================================================

def plot_precision_recall_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_proba: Union[np.ndarray, pd.Series],
    *,
    label: str = "Model",
    title: str = "",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> float:
    """
    Plot Precision-Recall curve and return Average Precision (AP).

    The PR curve shows the trade-off between Precision and Recall
    at various classification thresholds. Particularly useful for
    imbalanced datasets.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 and 1).
    y_proba : array-like of shape (n_samples,)
        Predicted probabilities for the positive class (DDoS).
    label : str, default "Model"
        Label for the curve in legend.
    title : str, default ""
        Custom plot title. If empty, uses "Precision-Recall Curve".
    output_path : str or Path, optional
        If provided, saves the figure to this path.
    verbose : bool, default True
        If False, suppress INFO logging.

    Returns
    -------
    ap_value : float
        Average Precision (AP), also known as PR-AUC.

    Raises
    ------
    ValueError
        If inputs are empty, have mismatched lengths,
        or y_proba contains values outside [0, 1].
    """
    _setup_logging(verbose)
    
    # ─── Input validation ────────────────────────────────────────────────────
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    _validate_inputs(y_true, y_proba, is_probability=True)
    
    # ─── Calculate PR curve and AP ───────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap_value = float(average_precision_score(y_true, y_proba))
    
    logger.debug(f"PR curve: {len(precision)} points, AP = {ap_value:.4f}")
    
    # ─── Calculate baseline (random classifier) ─────────────────────────────
    baseline = float(np.mean(y_true))
    
    # ─── Create figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=PLOT_STYLES["figsize_single"])
    
    # Plot PR curve
    ax.plot(
        recall, precision,
        color=PLOT_STYLES["pr_color"],
        linewidth=2,
        label=f"{label} (AP = {ap_value:.4f})",
    )
    
    # Plot baseline (horizontal line at positive class proportion)
    ax.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Random (P = {baseline:.3f})",
    )
    
    # ─── Styling ─────────────────────────────────────────────────────────────
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=PLOT_STYLES["label_size"])
    ax.set_ylabel("Precision", fontsize=PLOT_STYLES["label_size"])
    
    if not title:
        title = "Precision-Recall Curve"
    _apply_common_style(ax, title)
    
    ax.legend(loc="lower left", fontsize=PLOT_STYLES["legend_size"])
    
    # ─── Save and return ─────────────────────────────────────────────────────
    _save_figure(output_path)
    
    if output_path is None:
        plt.show()
    
    logger.info(f"Average Precision (PR-AUC): {ap_value:.4f}")
    
    return ap_value


def plot_pr_curves_comparison(
    results: List[Dict[str, Any]],
    *,
    title: str = "",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Plot multiple Precision-Recall curves on one figure for model comparison.

    Each model's PR curve is plotted with a different color, and the
    legend shows model names with their Average Precision (AP) values.

    Parameters
    ----------
    results : list of dict
        Each dict must contain:
        - 'name': str — model name for legend
        - 'y_true': array-like — true labels
        - 'y_proba': array-like — predicted probabilities
        
        Example:
        [
            {'name': 'RandomForest', 'y_true': y_test, 'y_proba': rf_proba},
            {'name': 'XGBoost', 'y_true': y_test, 'y_proba': xgb_proba},
        ]
    title : str, default ""
        Custom plot title. If empty, uses "Precision-Recall Curve Comparison".
    output_path : str or Path, optional
        If provided, saves the figure to this path.
    verbose : bool, default True
        If False, suppress INFO logging.

    Returns
    -------
    ap_dict : dict
        Dictionary mapping model names to Average Precision values.
        Example: {'RandomForest': 0.9978, 'XGBoost': 0.9983}
    """
    _setup_logging(verbose)
    
    # ─── Validation ──────────────────────────────────────────────────────────
    if not results:
        raise ValueError("results list is empty")
    
    required_keys = {"name", "y_true", "y_proba"}
    for i, res in enumerate(results):
        missing = required_keys - set(res.keys())
        if missing:
            raise ValueError(
                f"Result {i} is missing required keys: {missing}. "
                f"Found keys: {list(res.keys())}"
            )
    
    # ─── Create figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=PLOT_STYLES["figsize_comparison"])
    colors = PLOT_STYLES["palette"]
    ap_dict = {}
    
    # Calculate baseline from first result (assuming same y_true for all)
    baseline = float(np.mean(np.asarray(results[0]["y_true"])))
    
    for i, res in enumerate(results):
        name = res["name"]
        y_true = np.asarray(res["y_true"])
        y_proba = np.asarray(res["y_proba"])
        
        # Validate inputs
        _validate_inputs(y_true, y_proba, is_probability=True)
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap_value = float(average_precision_score(y_true, y_proba))
        ap_dict[name] = ap_value
        
        # Plot
        color = colors[i % len(colors)]
        ax.plot(
            recall, precision,
            color=color,
            linewidth=2,
            label=f"{name} (AP = {ap_value:.4f})",
        )
    
    # ─── Random classifier baseline ──────────────────────────────────────────
    ax.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Random (P = {baseline:.3f})",
    )
    
    # ─── Styling ─────────────────────────────────────────────────────────────
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=PLOT_STYLES["label_size"])
    ax.set_ylabel("Precision", fontsize=PLOT_STYLES["label_size"])
    
    if not title:
        title = "Precision-Recall Curve Comparison"
    _apply_common_style(ax, title)
    
    ax.legend(loc="lower left", fontsize=PLOT_STYLES["legend_size"])
    
    # ─── Save and return ─────────────────────────────────────────────────────
    _save_figure(output_path)
    
    if output_path is None:
        plt.show()
    
    logger.info(f"Compared {len(results)} models: " +
                ", ".join(f"{k}={v:.4f}" for k, v in ap_dict.items()))
    
    return ap_dict


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(
    importances: Union[np.ndarray, List[float]],
    feature_names: Sequence[str],
    *,
    top_k: int = 15,
    title: str = "",
    xlabel: str = "Importance",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Plot horizontal bar chart of top-K most important features.

    Features are sorted by importance in descending order and displayed
    as horizontal bars for easy reading of feature names.

    Parameters
    ----------
    importances : array-like of shape (n_features,)
        Importance scores for each feature (e.g., from tree.feature_importances_).
    feature_names : sequence of str
        Names of all features.
    top_k : int, default 15
        Number of top features to display. If more features exist,
        only the top_k most important are shown.
    title : str, default ""
        Custom plot title. If empty, uses "Top {k} Feature Importances".
    xlabel : str, default "Importance"
        Label for the x-axis (importance values).
    output_path : str or Path, optional
        If provided, saves the figure to this path.
    verbose : bool, default True
        If False, suppress INFO logging.

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame with columns ['feature', 'importance'], sorted by importance
        (all features, not just top_k).

    Raises
    ------
    ValueError
        If feature_names and importances have different lengths,
        or if either is empty.
    """
    _setup_logging(verbose)
    
    # ─── Input validation ────────────────────────────────────────────────────
    feature_names = list(feature_names)
    importances = np.asarray(importances)
    
    if len(feature_names) == 0:
        raise ValueError("feature_names is empty")
    
    if len(importances) == 0:
        raise ValueError("importances is empty")
    
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Length mismatch: feature_names has {len(feature_names)} items, "
            f"importances has {len(importances)} items"
        )
    
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    
    # ─── Create importance DataFrame ─────────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    importance_df = importance_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    
    # Select top_k features
    top_k = min(top_k, len(importance_df))
    plot_df = importance_df.head(top_k).iloc[::-1]  # Reverse for horizontal bar
    
    logger.debug(f"Plotting top {top_k} of {len(importance_df)} features")
    
    # ─── Create figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=PLOT_STYLES["figsize_importance"])
    
    # Horizontal bar chart
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(
        y_pos,
        plot_df["importance"],
        color=PLOT_STYLES["palette"][0],
        edgecolor="black",
        linewidth=0.5,
    )
    
    # ─── Labels ──────────────────────────────────────────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"], fontsize=PLOT_STYLES["tick_size"])
    ax.set_xlabel(xlabel, fontsize=PLOT_STYLES["label_size"])
    
    if not title:
        title = f"Top {top_k} Feature Importances"
    _apply_common_style(ax, title)
    
    # Add value labels on bars
    for bar, val in zip(bars, plot_df["importance"]):
        ax.text(
            val + max(plot_df["importance"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=PLOT_STYLES["annotation_size"],
        )
    
    # ─── Save and return ─────────────────────────────────────────────────────
    _save_figure(output_path)
    
    if output_path is None:
        plt.show()
    
    logger.info(
        f"Top 3 features: "
        f"{importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.4f}), "
        f"{importance_df.iloc[1]['feature']} ({importance_df.iloc[1]['importance']:.4f}), "
        f"{importance_df.iloc[2]['feature']} ({importance_df.iloc[2]['importance']:.4f})"
    )
    
    return importance_df


# ============================================================================
# F1 SCORE BAR CHART COMPARISON
# ============================================================================

def plot_f1_comparison(
    model_names: Sequence[str],
    f1_scores: Sequence[float],
    *,
    title: str = "",
    xlabel: str = "Model",
    ylabel: str = "F1 Score",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> None:
    """
    Plot bar chart comparing F1 scores across multiple models.

    Creates a simple, clean bar chart for comparing model performance.
    Bars are colored by model and labeled with exact F1 values.

    Parameters
    ----------
    model_names : list of str
        Names of models to compare.
    f1_scores : array-like
        F1 scores for each model (same order as model_names).
    title : str, default ""
        Custom plot title. If empty, uses "F1 Score Comparison".
    xlabel : str, default "Model"
        Label for x-axis.
    ylabel : str, default "F1 Score"
        Label for y-axis.
    output_path : str or Path, optional
        If provided, saves the figure to this path.
    verbose : bool, default True
        If False, suppress INFO logging.

    Raises
    ------
    ValueError
        If model_names and f1_scores have different lengths,
        or if either is empty.
    """
    _setup_logging(verbose)
    
    # ─── Input validation ────────────────────────────────────────────────────
    if len(model_names) == 0:
        raise ValueError("model_names is empty")