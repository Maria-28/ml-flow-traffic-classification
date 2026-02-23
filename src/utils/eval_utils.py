"""
Evaluation utilities for DDoS detection experiments.
Provides metrics calculation, statistical tests, and result saving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any, Callable
from scipy import stats
import warnings


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    *,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0/1).
    y_pred : np.ndarray
        Predicted labels (0/1).
    y_proba : np.ndarray, optional
        Prediction probabilities for positive class.
        Required for AUC metrics.
    verbose : bool, default False
        Print metrics as they're calculated.
        
    Returns
    -------
    Dict[str, float]
        Dictionary with metrics: precision, recall, f1,
        roc_auc (if y_proba provided), pr_auc (if y_proba provided).
        
    Examples
    --------
    >>> metrics = calculate_metrics(y_test, y_pred, y_proba)
    >>> print(f"F1: {metrics['f1']:.4f}")
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    
    metrics = {}
    
    # Basic metrics
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC metrics (require probabilities)
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    
    if verbose:
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        if y_proba is not None:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    
    return metrics


# ============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# ============================================================================

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    continuity_correction: bool = True,
) -> Tuple[float, float]:
    """
    Perform McNemar's test to compare two classifiers.
    
    Tests whether two classifiers have significantly different error rates.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred_a : np.ndarray
        Predictions from classifier A (baseline).
    y_pred_b : np.ndarray
        Predictions from classifier B (candidate).
    continuity_correction : bool, default True
        Apply Edwards' continuity correction.
        
    Returns
    -------
    chi2 : float
        McNemar's chi-squared statistic.
    p_value : float
        Two-tailed p-value.
        
    Notes
    -----
    If p < 0.05, classifiers are significantly different.
    """
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)
    
    b = np.sum((correct_a == 1) & (correct_b == 0))
    c = np.sum((correct_a == 0) & (correct_b == 1))
    
    if b + c == 0:
        return 0.0, 1.0
    
    if continuity_correction:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)
    
    p_value = stats.chi2.sf(chi2, df=1)
    
    return float(chi2), float(p_value)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for any metric.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    metric_func : callable
        Function that takes (y_true, y_pred) and returns float.
    n_iterations : int, default 1000
        Number of bootstrap iterations.
    confidence_level : float, default 0.95
        Confidence level (e.g., 0.95 for 95% CI).
    random_state : int, default 42
        Random seed.
        
    Returns
    -------
    point_estimate : float
        Metric on original data.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    point_estimate = metric_func(y_true, y_pred)
    
    bootstrap_scores = []
    for _ in range(n_iterations):
        indices = rng.randint(0, n_samples, size=n_samples)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return point_estimate, float(ci_lower), float(ci_upper)


def f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Helper function for F1 bootstrap."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, zero_division=0))


def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for two proportions.
    
    Parameters
    ----------
    p1 : float
        First proportion (e.g., accuracy of model 1).
    p2 : float
        Second proportion (e.g., accuracy of model 2).
        
    Returns
    -------
    float
        Cohen's h effect size.
        
    Notes
    -----
    Interpretation: 0.2 = small, 0.5 = medium, 0.8 = large.
    """
    from math import asin, sqrt
    return 2 * abs(asin(sqrt(p1)) - asin(sqrt(p2)))


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = f1_metric,
    *,
    n_iterations: int = 10000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Paired bootstrap test for comparing two models.
    
    Tests if the difference in metric is significantly different from zero.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred_a : np.ndarray
        Predictions from model A.
    y_pred_b : np.ndarray
        Predictions from model B.
    metric_func : callable, default f1_metric
        Metric to compare.
    n_iterations : int, default 10000
        Number of bootstrap iterations.
    random_state : int, default 42
        Random seed.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with test results.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    # Observed difference
    metric_a = metric_func(y_true, y_pred_a)
    metric_b = metric_func(y_true, y_pred_b)
    observed_diff = metric_b - metric_a
    
    # Bootstrap differences
    bootstrap_diffs = []
    for _ in range(n_iterations):
        indices = rng.randint(0, n_samples, size=n_samples)
        diff = (metric_func(y_true[indices], y_pred_b[indices]) - 
                metric_func(y_true[indices], y_pred_a[indices]))
        bootstrap_diffs.append(diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Confidence interval for difference
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    # P-value (two-tailed)
    p_value = np.mean(np.abs(bootstrap_diffs) >= abs(observed_diff))
    
    return {
        'metric_a': metric_a,
        'metric_b': metric_b,
        'observed_diff': observed_diff,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


# ============================================================================
# RESULT SAVING (UNIVERSAL VERSION)
# ============================================================================

def save_results_table(
    df: pd.DataFrame,
    output_prefix: Union[str, Path],
    *,
    caption: str = "",
    label: str = "",
    formats: List[str] = None,
    highlight_best: bool = True,
    verbose: bool = True,
) -> None:
    """
    Save results table in multiple formats (CSV, LaTeX, PNG).
    
    UNIVERSAL function that works with ANY DataFrame structure.
    Automatically detects if this is a model comparison table
    (contains 'Model' column) or a generic results table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame (any structure).
    output_prefix : str or Path
        Prefix for output files (e.g., "tables/model_comparison").
    caption : str, optional
        LaTeX table caption.
    label : str, optional
        LaTeX table label.
    formats : List[str], optional
        Output formats: ['csv', 'tex', 'png']. Default: all.
    highlight_best : bool, default True
        Highlight best model in PNG table (if 'Model' and 'F1' exist).
    verbose : bool, default True
        Print save messages.
        
    Examples
    --------
    >>> # Save model comparison
    >>> save_results_table(results_df, "tables/model_comparison")
    >>> 
    >>> # Save statistical tests
    >>> save_results_table(stats_df, "tables/statistical_tests")
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    if formats is None:
        formats = ['csv', 'tex', 'png']
    
    # Clean DataFrame
    df_clean = df.copy()
    
    # Handle NaN values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # ========================================================================
    # 1. CSV EXPORT (always works, no formatting)
    # ========================================================================
    if 'csv' in formats:
        csv_path = output_prefix.with_suffix(".csv")
        df_clean.to_csv(csv_path, index=False, encoding='utf-8-sig')
        if verbose:
            print(f"  Saved: {csv_path}")
    
    # ========================================================================
    # 2. LATEX EXPORT (intelligent formatting)
    # ========================================================================
    if 'tex' in formats:
        tex_path = output_prefix.with_suffix(".tex")
        
        # Prepare DataFrame for LaTeX
        latex_df = df_clean.copy()
        
        # Format numeric columns appropriately
        for col in latex_df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            
            # Scientific notation for very small p-values
            if 'p-value' in col_lower or 'pvalue' in col_lower:
                latex_df[col] = latex_df[col].apply(
                    lambda x: f"{x:.2e}" if pd.notna(x) and x < 0.001 
                    else (f"{x:.4f}" if pd.notna(x) else "---")
                )
            # Format for chi-square and other test statistics
            elif any(x in col_lower for x in ['chi', 'mcnemar', 'statistic']):
                latex_df[col] = latex_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "---"
                )
            # Format for metrics (0-1 range)
            elif any(x in col_lower for x in ['f1', 'auc', 'precision', 'recall', 'accuracy']):
                latex_df[col] = latex_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "---"
                )
            # Format for percentages
            elif any(x in col_lower for x in ['rate', 'ratio', '%']):
                latex_df[col] = latex_df[col].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "---"
                )
            # Default numeric formatting
            else:
                latex_df[col] = latex_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) and abs(x) < 1000
                    else (f"{x:.1f}" if pd.notna(x) else "---")
                )
        
        # Determine column format for LaTeX
        is_model_table = 'Model' in latex_df.columns
        
        if is_model_table:
            # Model table: first column left-aligned (model names), others centered
            col_format = "l" + "c" * (len(latex_df.columns) - 1)
        else:
            # Generic table: all columns left-aligned (likely text)
            # But check if columns contain mostly numbers
            text_cols = sum(latex_df[col].astype(str).str.contains('[a-zA-Z]').any() 
                           for col in latex_df.columns)
            if text_cols > len(latex_df.columns) / 2:
                col_format = "l" * len(latex_df.columns)  # Mostly text
            else:
                col_format = "c" * len(latex_df.columns)  # Mostly numbers
        
        # Generate caption and label
        if not caption:
            if is_model_table:
                caption = "Performance comparison of DDoS detection models"
            else:
                caption = "Statistical significance test results"
        
        if not label:
            if is_model_table:
                label = "tab:model_comparison"
            else:
                label = "tab:statistical_tests"
        
        # Create LaTeX table
        latex_str = latex_df.to_latex(
            index=False,
            caption=caption,
            label=label,
            column_format=col_format,
            escape=True,
            na_rep="---",
            float_format="%.4f",
        )
        
        # Add booktabs styling for better appearance
        latex_str = latex_str.replace("\\begin{tabular}", "\\begin{tabular}")
        latex_str = latex_str.replace("\\toprule", "\\midrule")
        
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        
        if verbose:
            print(f"  Saved: {tex_path}")
    
    # ========================================================================
    # 3. PNG TABLE EXPORT (publication-ready)
    # ========================================================================
    if 'png' in formats:
        png_path = output_prefix.with_suffix(".png")
        
        # Prepare display DataFrame
        display_df = df_clean.copy()
        
        # Format for display (similar to LaTeX)
        for col in display_df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            
            if 'p-value' in col_lower or 'pvalue' in col_lower:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2e}" if pd.notna(x) and x < 0.001 
                    else (f"{x:.4f}" if pd.notna(x) else "---")
                )
            elif any(x in col_lower for x in ['f1', 'auc', 'precision', 'recall']):
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "---"
                )
            elif any(x in col_lower for x in ['rate', 'ratio']):
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "---"
                )
            else:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) and abs(x) < 1000
                    else (f"{x:.1f}" if pd.notna(x) else "---")
                )
        
        # Calculate figure size dynamically
        n_rows = len(display_df)
        n_cols = len(display_df.columns)
        
        # Base dimensions
        row_height = 0.45
        col_width = 1.8
        header_height = 1.0
        
        fig_width = max(8, n_cols * col_width)
        fig_height = max(3, n_rows * row_height + header_height)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        
        # Create table
        col_widths = [1.0 / n_cols] * n_cols
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=col_widths,
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        # Style header
        for j in range(n_cols):
            cell = table[0, j]
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        
        # Style rows
        is_model_table = 'Model' in display_df.columns
        best_row_idx = None
        
        if is_model_table and highlight_best and 'F1' in display_df.columns:
            # Find best model by F1
            f1_values = []
            for i in range(1, n_rows + 1):
                try:
                    val = float(display_df.iloc[i-1]['F1'])
                    f1_values.append(val)
                except (ValueError, TypeError):
                    f1_values.append(-1)
            
            if f1_values:
                best_row_idx = np.argmax(f1_values)
        
        for i in range(1, n_rows + 1):
            row_idx = i - 1
            
            # Alternating row colors
            if row_idx == best_row_idx:
                bg_color = '#d4edda'  # Light green for best model
            elif row_idx % 2 == 0:
                bg_color = '#f0f4f8'  # Light blue for even rows
            else:
                bg_color = '#ffffff'   # White for odd rows
            
            for j in range(n_cols):
                table[i, j].set_facecolor(bg_color)
        
        plt.title(
            caption if caption else "Experiment Results",
            fontsize=12,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        if verbose:
            print(f"  Saved: {png_path}")


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
    *,
    model_name: str = "Model",
    output_dir: Union[str, Path],
    y_proba: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive error analysis for a trained model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    X_test : pd.DataFrame
        Test features.
    model_name : str, default "Model"
        Name of the model (for file naming).
    output_dir : str or Path
        Directory to save error analysis files.
    y_proba : np.ndarray, optional
        Prediction probabilities.
    verbose : bool, default True
        Print progress.
        
    Returns
    -------
    Dict[str, Any]
        Error statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Identify errors
    fp_mask = (y_pred == 1) & (y_true == 0)  # False Positives
    fn_mask = (y_pred == 0) & (y_true == 1)  # False Negatives
    
    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    
    n_benign = (y_true == 0).sum()
    n_ddos = (y_true == 1).sum()
    
    fp_rate = fp_count / n_benign * 100 if n_benign > 0 else 0
    fn_rate = fn_count / n_ddos * 100 if n_ddos > 0 else 0
    
    # Save error samples
    X_reset = X_test.reset_index(drop=True)
    
    if fp_count > 0:
        fp_df = X_reset.loc[fp_mask].copy()
        fp_df['true_label'] = 'BENIGN'
        fp_df['predicted_label'] = 'DDoS'
        fp_df['sample_index'] = np.where(fp_mask)[0]
        
        fp_path = output_dir / f"{model_name.lower()}_fp_samples.csv"
        fp_df.to_csv(fp_path, index=False)
        if verbose:
            print(f"  Saved: {fp_path}")
    
    if fn_count > 0:
        fn_df = X_reset.loc[fn_mask].copy()
        fn_df['true_label'] = 'DDoS'
        fn_df['predicted_label'] = 'BENIGN'
        fn_df['sample_index'] = np.where(fn_mask)[0]
        
        fn_path = output_dir / f"{model_name.lower()}_fn_samples.csv"
        fn_df.to_csv(fn_path, index=False)
        if verbose:
            print(f"  Saved: {fn_path}")
    
    # Confidence analysis (if probabilities available)
    if y_proba is not None and (fp_count > 0 or fn_count > 0):
        conf_fp = y_proba[fp_mask] if fp_count > 0 else []
        conf_fn = y_proba[fn_mask] if fn_count > 0 else []
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        if fp_count > 0:
            axes[0].hist(conf_fp, bins=20, alpha=0.7, color='red')
            axes[0].set_xlabel('Predicted Probability (DDoS)')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'False Positives (n={fp_count})')
            axes[0].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        
        if fn_count > 0:
            axes[1].hist(conf_fn, bins=20, alpha=0.7, color='orange')
            axes[1].set_xlabel('Predicted Probability (DDoS)')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'False Negatives (n={fn_count})')
            axes[1].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Error Analysis - {model_name}', fontweight='bold')
        plt.tight_layout()
        
        plot_path = output_dir / f"{model_name.lower()}_error_hist.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"  Saved: {plot_path}")
    
    # Return statistics
    return {
        'model': model_name,
        'fp_count': int(fp_count),
        'fn_count': int(fn_count),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn_rate),
        'total_errors': int(fp_count + fn_count),
        'error_rate': float((fp_count + fn_count) / len(y_true) * 100),
    }


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(
    results_list: List[Dict[str, Any]],
    *,
    y_true: Optional[np.ndarray] = None,
    predictions_dict: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Create comparison table from multiple model results.
    
    Parameters
    ----------
    results_list : List[Dict]
        List of result dictionaries from calculate_metrics().
    y_true : np.ndarray, optional
        True labels (required for statistical tests).
    predictions_dict : Dict[str, np.ndarray], optional
        Dictionary of model predictions for pairwise comparisons.
        
    Returns
    -------
    pd.DataFrame
        Comparison table with rankings and pairwise statistics.
    """
    df = pd.DataFrame(results_list)
    
    # Add ranking
    if 'f1' in df.columns:
        df = df.sort_values('f1', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    preferred_order = ['Rank', 'Model', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc']
    available_cols = [c for c in preferred_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in preferred_order]
    df = df[available_cols + other_cols]
    
    return df