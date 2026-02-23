"""
================================================================================
BASELINE MODELS FOR DDOS DETECTION — CIC-IDS2017
Master's Thesis | Binary Classification: BENIGN vs DDoS
================================================================================

USAGE:
  cd ml-flow-traffic-classification
  python src/baseline/baseline_ddos.py

EXPECTED PROJECT LAYOUT:
  ml-flow-traffic-classification/          <- PROJECT_ROOT
  |-- data/raw/Friday-DDos.pcap_ISCX.csv  <- input data
  |-- results/tables/                      <- CSV, LaTeX, PNG table output
  |-- results/figures/                     <- all plots output
  |-- src/baseline/baseline_ddos.py       <- THIS FILE

WHAT THIS SCRIPT DOES:
  1.  Load and clean CIC-IDS2017 data
  2.  Remove data-leakage features (Flow Bytes/s, Subflow*, Bulk*, etc.)
  3.  Remove constant (zero-variance) features automatically
  4.  Temporal split: first 80% -> train, last 20% -> test  (no shuffling)
  5.  Train 3 baseline pipelines: LogisticRegression, DecisionTree, RandomForest
  6.  Evaluate: Precision, Recall, F1, ROC-AUC, PR-AUC
  7.  Per-model plots: confusion matrix, ROC curve, PR curve,
      feature importance (tree models only)
  8.  Combined plots: all ROC curves, metrics comparison bar chart
  9.  Save results: CSV, LaTeX table (.tex), PNG table
  10. Error analysis on best model: FP/FN CSVs + temporal scatter plot
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend, works everywhere
import matplotlib.pyplot as plt
import seaborn as sns          # noqa: F401  (imported for style consistency)

from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings('ignore')
np.random.seed(42)


# ============================================================================
# SECTION 2: PATHS AND CONFIGURATION
# ============================================================================

# ---------------------------------------------------------------------------
# Path resolution
# This file lives at: <PROJECT_ROOT>/src/baseline/baseline_ddos.py
# Walk up 2 directories to reach PROJECT_ROOT.
# ---------------------------------------------------------------------------
_THIS_FILE    = os.path.abspath(__file__)
_BASELINE_DIR = os.path.dirname(_THIS_FILE)      # .../src/baseline/
_SRC_DIR      = os.path.dirname(_BASELINE_DIR)   # .../src/
PROJECT_ROOT  = os.path.dirname(_SRC_DIR)         # .../ml-flow-traffic-classification/

DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "raw", "Friday-DDos.pcap_ISCX.csv")
DIR_TABLES  = os.path.join(PROJECT_ROOT, "results", "tables")
DIR_FIGURES = os.path.join(PROJECT_ROOT, "results", "figures")

os.makedirs(DIR_TABLES,  exist_ok=True)
os.makedirs(DIR_FIGURES, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')   # fallback for older matplotlib versions

plt.rcParams.update({
    'font.size':      12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize':10,
    'font.family':    'DejaVu Sans',
    'figure.dpi':     100,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ['BENIGN', 'DDoS']
TRAIN_RATIO = 0.80

MODEL_COLORS = {
    'LogisticRegression': '#1f77b4',   # blue
    'DecisionTree':       '#2ca02c',   # green
    'RandomForest':       '#d62728',   # red
}

# ---------------------------------------------------------------------------
# Data-leakage features
# These are computed over the ENTIRE completed flow, so they are
# unavailable at detection time in a real-time IDS.
# ---------------------------------------------------------------------------
LEAKAGE_FEATURES = [
    'Flow Bytes/s',
    'Flow Packets/s',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Avg Packet Size',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',
]


# ============================================================================
# SECTION 3: UTILITIES
# ============================================================================

def _section(title: str):
    """Print a clearly visible section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _save_fig(filename: str):
    """Save the current matplotlib figure to DIR_FIGURES at 300 DPI."""
    path = os.path.join(DIR_FIGURES, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     Saved: {path}")


def _fail(message: str):
    """Print a formatted error message and exit."""
    print(f"\nERROR:\n  {message}\n")
    sys.exit(1)


# ============================================================================
# SECTION 4: DATA LOADING AND CLEANING
# ============================================================================

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Load CIC-IDS2017 CSV and apply cleaning:
      - Strip whitespace from column names
      - Replace +/-inf with NaN, then drop all NaN rows
      - Validate Label column and keep BENIGN / DDoS only
      - Create integer target column 'y'

    Returns
    -------
    df : pd.DataFrame with clean data and columns 'Label', 'y'
    """
    _section("STEP 1: LOADING AND CLEANING DATA")

    print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  DATA_PATH    : {path}")
    print(f"  DIR_TABLES   : {DIR_TABLES}")
    print(f"  DIR_FIGURES  : {DIR_FIGURES}\n")

    if not os.path.exists(path):
        _fail(
            f"Data file not found:\n"
            f"    {path}\n\n"
            f"  Expected project layout:\n"
            f"    ml-flow-traffic-classification/\n"
            f"      data/raw/Friday-DDos.pcap_ISCX.csv  <- place file here\n"
            f"      src/baseline/baseline_ddos.py\n\n"
            f"  Run from the project root:\n"
            f"    cd ml-flow-traffic-classification\n"
            f"    python src/baseline/baseline_ddos.py"
        )

    print("  Reading CSV (may take a moment)...")
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Raw shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"  Rows removed (NaN/Inf) : {before - len(df):,}")

    if 'Label' not in df.columns:
        _fail("Column 'Label' not found in the CSV. Check your data file.")

    df['Label'] = df['Label'].astype(str).str.strip()

    unexpected = set(df['Label'].unique()) - {'BENIGN', 'DDoS'}
    if unexpected:
        print(f"  WARNING: unexpected labels dropped: {unexpected}")
    df = df[df['Label'].isin(['BENIGN', 'DDoS'])].copy()

    df['y'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1}).astype(int)

    print(f"  Clean shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  BENIGN : {(df['y']==0).sum():,}   DDoS : {(df['y']==1).sum():,}")
    return df


# ============================================================================
# SECTION 5: FEATURE ENGINEERING
# ============================================================================

def build_feature_matrix(df: pd.DataFrame):
    """
    Build X (feature matrix) and y (target), removing:
      1. Label / target helper columns
      2. Known data-leakage features (listed in LEAKAGE_FEATURES)
      3. Constant features (zero variance on training data)
      4. Any non-numeric columns that remain

    Returns
    -------
    X             : pd.DataFrame
    y             : pd.Series  (int 0/1)
    feature_names : list[str]
    """
    _section("STEP 2: FEATURE ENGINEERING")

    drop_cols = ['Label', 'y']

    leakage_present = [f for f in LEAKAGE_FEATURES if f in df.columns]
    drop_cols += leakage_present

    print(f"  Leakage features removed ({len(leakage_present)}):")
    for f in leakage_present:
        print(f"    - {f}")

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['y'].copy()

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Remove constant features
    constant_cols = X.columns[X.var() == 0].tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
        print(f"\n  Constant features removed ({len(constant_cols)}): {constant_cols}")
    else:
        print("\n  No constant features found.")

    print(f"\n  Final features : {X.shape[1]}")
    print(f"  Final samples  : {X.shape[0]:,}")
    return X, y, X.columns.tolist()


# ============================================================================
# SECTION 6: TEMPORAL SPLIT
# ============================================================================

def temporal_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.80):
    """
    Chronological (temporal) split - NO shuffling at any point.
    First `train_ratio` fraction is train, remainder is test.

    Returns X_train, X_test, y_train, y_test
    """
    _section("STEP 3: TEMPORAL SPLIT")

    n         = len(X)
    split_idx = int(n * train_ratio)

    X_train = X.iloc[:split_idx].copy()
    X_test  = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test  = y.iloc[split_idx:].copy()

    def dist(s):
        return f"BENIGN={int((s==0).sum()):,}  DDoS={int((s==1).sum()):,}"

    print(f"  Total samples : {n:,}")
    print(f"  Train         : {len(X_train):,}  ({train_ratio*100:.0f}%)  -> {dist(y_train)}")
    print(f"  Test          : {len(X_test):,}  ({(1-train_ratio)*100:.0f}%)  -> {dist(y_test)}")
    print(f"  Split at row  : {split_idx:,}")

    return X_train, X_test, y_train, y_test


# ============================================================================
# SECTION 7: MODEL PIPELINES
# ============================================================================

def build_pipelines() -> dict:
    """
    Return sklearn Pipelines for each baseline model.
    All pipelines expose predict_proba for ROC / PR calculation.
    """
    return {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000, random_state=42,
                solver='lbfgs', n_jobs=-1,
            )),
        ]),
        'DecisionTree': Pipeline([
            ('clf', DecisionTreeClassifier(random_state=42)),
        ]),
        'RandomForest': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1,
            )),
        ]),
    }


# ============================================================================
# SECTION 8: TRAINING AND EVALUATION
# ============================================================================

def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test) -> dict:
    """
    Fit pipeline on train set and evaluate on test set.

    Returns a metrics dict; keys starting with '_' hold objects
    needed for plotting (not exported to the results table).
    """
    print(f"\n  > {name}")

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    prec    = precision_score(y_test, y_pred,  pos_label=1, zero_division=0)
    rec     = recall_score(y_test,    y_pred,  pos_label=1, zero_division=0)
    f1      = f1_score(y_test,        y_pred,  pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test,   y_proba)
    pr_auc  = average_precision_score(y_test,  y_proba)

    print(
        f"    Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}"
        f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}"
        f"  ({train_time:.1f}s)"
    )

    return {
        'Model'    : name,
        'Precision': round(prec,    4),
        'Recall'   : round(rec,     4),
        'F1'       : round(f1,      4),
        'ROC-AUC'  : round(roc_auc, 4),
        'PR-AUC'   : round(pr_auc,  4),
        '_pipeline': pipeline,
        '_y_pred'  : y_pred,
        '_y_proba' : y_proba,
        '_train_s' : round(train_time, 1),
    }


# ============================================================================
# SECTION 9: PER-MODEL PLOTS
# ============================================================================

def plot_confusion_matrix(name: str, y_test, y_pred):
    """Side-by-side: raw counts (left) and row-normalised (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, normalize, suffix in zip(
        axes,
        [None, 'true'],
        ['(counts)', '(normalised)'],
    ):
        cm   = confusion_matrix(y_test, y_pred, normalize=normalize)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(
            ax=ax, cmap='Blues', colorbar=False,
            values_format=('.0f' if normalize is None else '.2f'),
        )
        ax.set_title(f'{name} - Confusion Matrix {suffix}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label',      fontsize=11)

    plt.tight_layout()
    _save_fig(f'{name}_confusion_matrix.png')


def plot_roc_curve(name: str, y_test, y_proba):
    """ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
            label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate',  fontweight='bold')
    ax.set_title(f'ROC Curve - {name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(f'{name}_roc_curve.png')


def plot_pr_curve(name: str, y_test, y_proba):
    """Precision-Recall curve with Average Precision score."""
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
    ap       = average_precision_score(y_test, y_proba)
    baseline = float((np.array(y_test) == 1).sum()) / len(y_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_vals, prec_vals, color=MODEL_COLORS[name], lw=2,
            label=f'AP = {ap:.4f}')
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1,
               label=f'Baseline DDoS ratio = {baseline:.2f}')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall',    fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title(f'Precision-Recall Curve - {name}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(f'{name}_pr_curve.png')


def plot_feature_importance(name: str, pipeline, feature_names: list, top_n: int = 15):
    """
    Horizontal bar chart of top-N Gini importances.
    Silently skips LogisticRegression (no feature_importances_ attribute).
    """
    clf = pipeline.named_steps['clf']
    if not hasattr(clf, 'feature_importances_'):
        return

    importances = clf.feature_importances_

    if len(feature_names) != len(importances):
        print(f"     WARNING: feature name / importance length mismatch, skipping.")
        return

    top_n      = min(top_n, len(feature_names))
    indices    = np.argsort(importances)[::-1][:top_n]
    top_names  = [feature_names[i] for i in indices]
    top_vals   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        range(top_n), top_vals[::-1],
        color=MODEL_COLORS[name], alpha=0.82,
        edgecolor='black', linewidth=0.6,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel('Feature Importance (mean Gini decrease)',
                  fontweight='bold')
    ax.set_title(f'Top-{top_n} Feature Importances - {name}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(val + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    _save_fig(f'{name}_feature_importance.png')


# ============================================================================
# SECTION 10: COMBINED PLOTS
# ============================================================================

def plot_all_roc_curves(results_list: list, y_test):
    """All models on a single ROC figure for direct comparison."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')

    for res in results_list:
        name        = res['Model']
        fpr, tpr, _ = roc_curve(y_test, res['_y_proba'])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
                label=f'{name}  (AUC = {roc_auc:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate',  fontweight='bold')
    ax.set_title('ROC Curves - All Baseline Models',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig('all_models_roc.png')


def plot_metrics_comparison(results_list: list):
    """Grouped bar chart of all models vs all metrics."""
    metric_cols = ['Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
    n_models    = len(results_list)
    x           = np.arange(len(metric_cols))
    width       = 0.22
    offsets     = np.linspace(
        -(n_models - 1) / 2 * width,
         (n_models - 1) / 2 * width,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for res, offset in zip(results_list, offsets):
        name = res['Model']
        vals = [res[m] for m in metric_cols]
        bars = ax.bar(
            x + offset, vals, width,
            label=name, color=MODEL_COLORS[name],
            alpha=0.85, edgecolor='black', linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8,
            )

    ax.set_ylim(0, 1.13)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=12)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Baseline Model Comparison - All Metrics',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_fig('all_models_metrics_comparison.png')


# ============================================================================
# SECTION 11: RESULTS TABLE
# ============================================================================

def save_results_table(results_list: list) -> pd.DataFrame:
    """
    Save results in three formats:
      - CSV   -> results/tables/baseline_results.csv
      - LaTeX -> results/tables/baseline_results.tex
      - PNG   -> results/tables/baseline_results_table.png
    """
    _section("STEP 7: SAVING RESULTS TABLE")

    metric_cols = ['Model', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
    results_df  = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith('_')}
         for r in results_list]
    )[metric_cols]

    # CSV
    csv_path = os.path.join(DIR_TABLES, 'baseline_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # LaTeX
    latex_str = results_df.to_latex(
        index=False,
        caption=(
            'Baseline classifier performance on CIC-IDS2017 DDoS detection '
            '(temporal 80/20 split, data-leakage features removed).'
        ),
        label='tab:baseline_results',
        column_format='lccccc',
        float_format='%.4f',
        escape=True,
    )
    tex_path = os.path.join(DIR_TABLES, 'baseline_results.tex')
    with open(tex_path, 'w', encoding='utf-8') as fh:
        fh.write(latex_str)
    print(f"  Saved: {tex_path}")

    # PNG table
    _save_table_png(results_df)

    return results_df


def _save_table_png(results_df: pd.DataFrame):
    """Render the metrics DataFrame as a styled PNG table."""
    metric_cols = ['Model', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
    tbl_df = results_df[metric_cols].copy()

    for col in metric_cols[1:]:
        tbl_df[col] = tbl_df[col].apply(lambda v: f'{float(v):.4f}')

    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis('off')

    col_widths = [0.24, 0.14, 0.14, 0.14, 0.14, 0.14]
    tbl = ax.table(
        cellText  = tbl_df.values,
        colLabels = tbl_df.columns.tolist(),
        cellLoc   = 'center',
        loc       = 'center',
        colWidths = col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.1)

    # Header
    for j in range(len(tbl_df.columns)):
        cell = tbl[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Rows: alternating background, green for best F1
    f1_values   = results_df['F1'].astype(float).values
    best_f1_row = int(np.argmax(f1_values))

    for i in range(1, len(tbl_df) + 1):
        row_idx = i - 1
        bg = '#f0f4f8' if row_idx % 2 == 0 else '#ffffff'
        if row_idx == best_f1_row:
            bg = '#d4edda'
        for j in range(len(tbl_df.columns)):
            tbl[i, j].set_facecolor(bg)

    ax.set_title(
        'Table 1. Baseline Classification Results  '
        '(CIC-IDS2017, Temporal Split 80/20)',
        fontsize=11, fontweight='bold', pad=14,
    )
    plt.tight_layout()

    path = os.path.join(DIR_TABLES, 'baseline_results_table.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# SECTION 12: ERROR ANALYSIS
# ============================================================================

def error_analysis(best_result: dict, X_test: pd.DataFrame,
                   y_test: pd.Series, feature_names: list):
    """
    Analyse false positives and false negatives for the best model.

    Saves:
      - fp_samples.csv / fn_samples.csv  (error rows with labels)
      - <Model>_error_analysis.png        (temporal scatter)
    Prints full sklearn classification report.
    """
    _section("STEP 8: ERROR ANALYSIS")

    name   = best_result['Model']
    y_pred = best_result['_y_pred']

    print(f"  Best model : {name}\n")

    y_arr    = np.array(y_test)
    fp_mask  = (y_pred == 1) & (y_arr == 0)   # predicted DDoS, truly BENIGN
    fn_mask  = (y_pred == 0) & (y_arr == 1)   # predicted BENIGN, truly DDoS

    n_benign = int((y_arr == 0).sum())
    n_ddos   = int((y_arr == 1).sum())
    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())

    fp_rate  = fp_count / n_benign * 100 if n_benign > 0 else 0.0
    fn_rate  = fn_count / n_ddos   * 100 if n_ddos   > 0 else 0.0

    print(f"  False Positives (FP) : {fp_count:,}  ({fp_rate:.2f}% of BENIGN in test)")
    print(f"  False Negatives (FN) : {fn_count:,}  ({fn_rate:.2f}% of DDoS  in test)")

    X_reset = X_test.reset_index(drop=True)

    fp_df = X_reset.loc[fp_mask].copy()
    fp_df['true_label']      = 'BENIGN'
    fp_df['predicted_label'] = 'DDoS'
    fp_df['sample_index']    = np.where(fp_mask)[0]

    fn_df = X_reset.loc[fn_mask].copy()
    fn_df['true_label']      = 'DDoS'
    fn_df['predicted_label'] = 'BENIGN'
    fn_df['sample_index']    = np.where(fn_mask)[0]

    fp_path = os.path.join(DIR_TABLES, 'fp_samples.csv')
    fn_path = os.path.join(DIR_TABLES, 'fn_samples.csv')
    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    print(f"\n  Saved: {fp_path}")
    print(f"  Saved: {fn_path}")

    # Temporal scatter: where in test sequence do errors appear?
    total_test = len(X_test)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 4))

    for ax, err_df, label, color in zip(
        axes,
        [fp_df, fn_df],
        [f'False Positives (FP)  n={fp_count:,}',
         f'False Negatives (FN)  n={fn_count:,}'],
        ['#e74c3c', '#e67e22'],
    ):
        if len(err_df) > 0:
            ax.scatter(
                err_df['sample_index'], np.ones(len(err_df)),
                c=color, alpha=0.35, s=12, marker='|',
            )
        ax.set_xlim(0, total_test)
        ax.set_xlabel('Sample index (temporal order)', fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Error Analysis - {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(f'{name}_error_analysis.png')

    print(f"\n  Classification report ({name}):")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()

    _section(
        f"BASELINE DDoS DETECTION - CIC-IDS2017\n"
        f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # 1. Load and clean
    df = load_and_clean_data(DATA_PATH)

    # 2. Feature matrix
    X, y, feature_names = build_feature_matrix(df)

    # 3. Temporal split
    X_train, X_test, y_train, y_test = temporal_split(X, y, TRAIN_RATIO)

    # 4. Train and evaluate
    _section("STEP 4: TRAINING AND EVALUATION")
    pipelines    = build_pipelines()
    results_list = []

    for name, pipeline in pipelines.items():
        result = evaluate_model(name, pipeline,
                                X_train, X_test, y_train, y_test)
        results_list.append(result)

    # 5. Per-model plots
    _section("STEP 5: PER-MODEL VISUALISATIONS")
    for res in results_list:
        name = res['Model']
        print(f"\n  -- {name} --")
        plot_confusion_matrix(name, y_test, res['_y_pred'])
        plot_roc_curve(name, y_test, res['_y_proba'])
        plot_pr_curve(name, y_test, res['_y_proba'])
        plot_feature_importance(name, res['_pipeline'], feature_names)

    # 6. Combined comparison plots
    _section("STEP 6: COMBINED COMPARISON PLOTS")
    plot_all_roc_curves(results_list, y_test)
    plot_metrics_comparison(results_list)

    # 7. Results table
    results_df = save_results_table(results_list)

    # 8. Error analysis on best model
    best_result = max(results_list, key=lambda r: r['F1'])
    error_analysis(best_result, X_test, y_test, feature_names)

    # 9. Final summary
    elapsed  = time.time() - start_time
    n_plots  = len([f for f in os.listdir(DIR_FIGURES) if f.endswith('.png')])
    n_tables = len([f for f in os.listdir(DIR_TABLES)
                    if f.endswith(('.csv', '.tex', '.png'))])

    _section("DONE - FINAL SUMMARY")
    print(f"  Total runtime : {elapsed:.1f} seconds\n")

    print("  Results table:")
    print()
    print(results_df.to_string(index=False))
    print()

    print(f"  Best model  : {best_result['Model']}")
    print(f"    Precision = {best_result['Precision']:.4f}")
    print(f"    Recall    = {best_result['Recall']:.4f}")
    print(f"    F1        = {best_result['F1']:.4f}")
    print(f"    ROC-AUC   = {best_result['ROC-AUC']:.4f}")
    print(f"    PR-AUC    = {best_result['PR-AUC']:.4f}")
    print()

    print(f"  Output locations:")
    print(f"    {DIR_FIGURES}/   <- {n_plots} PNG plots")
    print(f"    {DIR_TABLES}/baseline_results.csv")
    print(f"    {DIR_TABLES}/baseline_results.tex")
    print(f"    {DIR_TABLES}/baseline_results_table.png")
    print(f"    {DIR_TABLES}/fp_samples.csv")
    print(f"    {DIR_TABLES}/fn_samples.csv")
    print()
    print("  All outputs are ready for your Master's thesis.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
