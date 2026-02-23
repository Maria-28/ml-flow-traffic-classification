"""
Shared data loading and preprocessing utilities for CIC-IDS2017 DDoS detection.

This module provides production-ready functions for loading, cleaning, and
preparing network traffic data for machine learning experiments.

Key features:
  - Robust error handling with informative messages
  - Structured logging (can be disabled)
  - Full type annotations
  - Reproducibility: deterministic operations, no random shuffling
  - Performance: optimized pandas operations, optional parquet caching
  - Flexibility: configurable leakage features, train ratio, validation flags
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
import numpy as np

# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default handler if none configured (prevents "No handler" warnings)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter('%(levelname)s - %(message)s')
    )
    logger.addHandler(handler)

# ============================================================================
# DEFAULT DATA LEAKAGE FEATURES
# ============================================================================
# These features are computed over the ENTIRE flow after completion,
# so they cannot be known in real-time IDS scenarios.
# Source: CIC-IDS2017 technical documentation + domain knowledge.
# ============================================================================

DEFAULT_LEAKAGE_FEATURES = [
    'Flow Bytes/s',
    'Flow Packets/s',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Average Packet Size',
    'Avg Packet Size',           # alternative spelling in some versions
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
# MAIN API FUNCTIONS
# ============================================================================

def load_and_clean_data(
    path: Union[str, Path],
    *,
    encoding: str = 'latin-1',
    remove_duplicates: bool = False,
    expected_labels: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load CIC-IDS2017 CSV and apply robust cleaning.

    Steps performed:
      1. Validate file existence and non-emptiness
      2. Load CSV with explicit encoding
      3. Strip whitespace from column names
      4. Replace ±inf with NaN, then drop all NaN rows
      5. Validate Label column exists
      6. Filter to expected labels (default: BENIGN, DDoS)
      7. Create integer target column 'y'
      8. Optionally remove duplicate rows
      9. Validate final dataset is non-empty

    Parameters
    ----------
    path : str or Path
        Path to CSV file.
    encoding : str, default 'latin-1'
        Character encoding (CIC-IDS2017 uses latin-1).
    remove_duplicates : bool, default False
        If True, drop duplicate rows based on all columns.
    expected_labels : list of str, optional
        Labels to keep. Default is ['BENIGN', 'DDoS'].
    verbose : bool, default True
        If False, suppress INFO-level logs (WARNING+ still shown).

    Returns
    -------
    df : pd.DataFrame
        Clean dataframe with columns 'Label' (str) and 'y' (int 0/1).

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If Label column missing, no data remains after cleaning,
        or unexpected label values found.

    Examples
    --------
    >>> df = load_and_clean_data('data/raw/Friday-DDos.pcap_ISCX.csv')
    >>> print(df.shape)
    (225709, 80)
    """
    if not verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    path = Path(path)
    if expected_labels is None:
        expected_labels = ['BENIGN', 'DDoS']

    # ─── Validation: file exists and is non-empty ───────────────────────────
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Data file is empty (0 bytes): {path}")

    logger.info(f"Loading CSV: {path.name} ({file_size / 1024 / 1024:.1f} MB)")

    # ─── Load CSV ────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(path, encoding=encoding, low_memory=False)
    except Exception as e:
        raise IOError(f"Failed to read CSV file: {e}") from e

    if df.empty:
        raise ValueError(f"Loaded DataFrame is empty: {path}")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    logger.info(f"Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ─── Handle inf and NaN ──────────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)
    rows_before_na = len(df)
    df = df.dropna()
    rows_removed_na = rows_before_na - len(df)

    if rows_removed_na > 0:
        logger.info(f"Removed {rows_removed_na:,} rows with NaN/Inf "
                    f"({rows_removed_na / rows_before_na * 100:.2f}%)")

    if df.empty:
        raise ValueError("All rows contain NaN/Inf — dataset is unusable.")

    # ─── Validate and clean Label column ─────────────────────────────────────
    if 'Label' not in df.columns:
        raise ValueError(
            f"Column 'Label' not found. Available columns: {df.columns.tolist()}"
        )

    df['Label'] = df['Label'].astype(str).str.strip()

    # Check for unexpected labels
    unique_labels = df['Label'].unique()
    unexpected = set(unique_labels) - set(expected_labels)
    if unexpected:
        logger.warning(
            f"Unexpected labels found and will be dropped: {sorted(unexpected)}"
        )

    # Filter to expected labels
    df = df[df['Label'].isin(expected_labels)].copy()

    if df.empty:
        raise ValueError(
            f"No rows remain after filtering to expected labels {expected_labels}. "
            f"Found labels: {sorted(unique_labels)}"
        )

    # ─── Create integer target ───────────────────────────────────────────────
    # Map first expected label to 0, second to 1 (typically BENIGN=0, DDoS=1)
    label_map = {label: idx for idx, label in enumerate(expected_labels)}
    df['y'] = df['Label'].map(label_map)

    # Validate y contains only expected values
    unique_y = df['y'].unique()
    if not set(unique_y).issubset({0, 1}):
        raise ValueError(
            f"Target column 'y' contains unexpected values: {sorted(unique_y)}"
        )

    # ─── Remove duplicates (optional) ────────────────────────────────────────
    if remove_duplicates:
        rows_before_dup = len(df)
        df = df.drop_duplicates(keep='first')
        rows_removed_dup = rows_before_dup - len(df)
        if rows_removed_dup > 0:
            logger.info(f"Removed {rows_removed_dup:,} duplicate rows "
                        f"({rows_removed_dup / rows_before_dup * 100:.2f}%)")

    # ─── Final validation ────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Dataset is empty after all cleaning steps.")

    logger.info(f"Clean shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(
        f"Class distribution: " +
        " | ".join(f"{label}={int((df['y'] == idx).sum()):,}"
                   for label, idx in label_map.items())
    )

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    remove_leakage: bool = True,
    leakage_features: Optional[List[str]] = None,
    remove_constant: bool = True,
    constant_threshold: float = 0.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, int]]:
    """
    Build feature matrix X and target y from a clean DataFrame.

    Steps performed:
      1. Drop label/target helper columns (Label, y)
      2. Optionally remove data-leakage features
      3. Keep only numeric columns
      4. Optionally remove constant features (variance threshold)
      5. Validate final feature set is non-empty

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Label' and 'y' columns.
    remove_leakage : bool, default True
        If True, remove features listed in `leakage_features`.
        Set False for ablation studies.
    leakage_features : list of str, optional
        Custom list of leakage features. If None, uses DEFAULT_LEAKAGE_FEATURES.
    remove_constant : bool, default True
        If True, remove features with variance <= constant_threshold.
    constant_threshold : float, default 0.0
        Variance threshold below which features are considered constant.
    verbose : bool, default True
        If False, suppress INFO-level logs.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all numeric, no leakage, no constants).
    y : pd.Series
        Target variable (int 0/1).
    feature_names : list of str
        Final feature names (column names of X).
    removed_info : dict
        Counts of removed features by category:
        {'leakage': int, 'non_numeric': int, 'constant': int}

    Raises
    ------
    ValueError
        If 'y' column missing, or if no features remain after filtering.

    Examples
    --------
    >>> X, y, feat_names, info = build_feature_matrix(df)
    >>> print(f"Features: {len(feat_names)}, Samples: {len(X)}")
    Features: 59, Samples: 225709
    """
    if not verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    if 'y' not in df.columns:
        raise ValueError("Column 'y' not found. Run load_and_clean_data() first.")

    removed_info: Dict[str, int] = {
        'leakage': 0,
        'non_numeric': 0,
        'constant': 0,
    }

    # ─── Step 1: Drop label columns ─────────────────────────────────────────
    drop_cols = ['Label', 'y']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['y'].copy()

    # ─── Step 2: Remove leakage features ────────────────────────────────────
    if remove_leakage:
        if leakage_features is None:
            leakage_features = DEFAULT_LEAKAGE_FEATURES

        leakage_present = [f for f in leakage_features if f in X.columns]
        removed_info['leakage'] = len(leakage_present)

        if leakage_present:
            X = X.drop(columns=leakage_present)
            logger.info(f"Removed {len(leakage_present)} leakage features:")
            for feat in leakage_present[:5]:
                logger.info(f"  - {feat}")
            if len(leakage_present) > 5:
                logger.info(f"  ... and {len(leakage_present) - 5} more")

    # ─── Step 3: Keep only numeric columns ──────────────────────────────────
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    removed_info['non_numeric'] = len(X.columns) - len(numeric_cols)

    if removed_info['non_numeric'] > 0:
        non_numeric = set(X.columns) - set(numeric_cols)
        logger.debug(f"Removed {removed_info['non_numeric']} non-numeric columns: {non_numeric}")

    X = X[numeric_cols]

    # ─── Step 4: Remove constant features ───────────────────────────────────
    if remove_constant:
        variances = X.var()
        constant_mask = variances <= constant_threshold
        constant_cols = variances[constant_mask].index.tolist()

        removed_info['constant'] = len(constant_cols)

        if constant_cols:
            X = X.drop(columns=constant_cols)
            logger.info(f"Removed {len(constant_cols)} constant features "
                        f"(variance ≤ {constant_threshold})")
            if len(constant_cols) <= 5:
                for feat in constant_cols:
                    logger.debug(f"  - {feat}")

    # ─── Final validation ────────────────────────────────────────────────────
    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "No features remain after filtering. "
            f"Removed: leakage={removed_info['leakage']}, "
            f"non-numeric={removed_info['non_numeric']}, "
            f"constant={removed_info['constant']}"
        )

    feature_names = X.columns.tolist()

    logger.info(f"Final feature matrix: {X.shape[1]} features, {X.shape[0]:,} samples")
    logger.info(
        f"Removed features: "
        f"leakage={removed_info['leakage']}, "
        f"non-numeric={removed_info['non_numeric']}, "
        f"constant={removed_info['constant']}"
    )

    return X, y, feature_names, removed_info


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.80,
    *,
    validate: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological (temporal) train/test split — NO shuffling.

    Critical for time-series data like network traffic: the model must
    be trained on earlier data and tested on later data to simulate
    real-world deployment.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    train_ratio : float, default 0.80
        Fraction of data for training (must be in (0, 1)).
    validate : bool, default True
        If True, check that train/test have both classes.
    verbose : bool, default True
        If False, suppress INFO-level logs.

    Returns
    -------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series

    Raises
    ------
    ValueError
        If train_ratio invalid, or if validation fails
        (e.g., test set has only one class).

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = temporal_split(X, y, train_ratio=0.80)
    >>> print(len(X_train), len(X_test))
    180567 45142
    """
    if not verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    # ─── Validate inputs ─────────────────────────────────────────────────────
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    if len(X) == 0:
        raise ValueError("Cannot split empty dataset")

    # ─── Perform split ───────────────────────────────────────────────────────
    n = len(X)
    split_idx = int(n * train_ratio)

    X_train = X.iloc[:split_idx].copy()
    X_test  = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test  = y.iloc[split_idx:].copy()

    # ─── Log split info ──────────────────────────────────────────────────────
    def class_dist(s: pd.Series) -> str:
        unique = s.unique()
        return " | ".join(f"class_{val}={int((s == val).sum()):,}" for val in sorted(unique))

    logger.info(
        f"Temporal split: {len(X_train):,} train ({train_ratio * 100:.1f}%) | "
        f"{len(X_test):,} test ({(1 - train_ratio) * 100:.1f}%)"
    )
    logger.info(f"  Train distribution: {class_dist(y_train)}")
    logger.info(f"  Test distribution:  {class_dist(y_test)}")
    logger.info(f"  Split at row index: {split_idx:,}")

    # ─── Validation ──────────────────────────────────────────────────────────
    if validate:
        # Check train set has both classes
        if len(y_train.unique()) < 2:
            raise ValueError(
                f"Train set has only one class: {y_train.unique()}. "
                f"Increase train_ratio or check data distribution."
            )

        # Check test set has both classes
        if len(y_test.unique()) < 2:
            logger.warning(
                f"Test set has only one class: {y_test.unique()}. "
                f"Evaluation metrics may be unreliable."
            )

    return X_train, X_test, y_train, y_test


# ============================================================================
# OPTIONAL: SAVE/LOAD CLEANED DATA IN PARQUET FORMAT
# ============================================================================

def save_cleaned_parquet(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    compression: str = 'snappy',
) -> None:
    """
    Save cleaned DataFrame to Parquet format for fast reloading.

    Parquet preserves dtypes and compresses well (~10x faster loading
    than CSV on subsequent runs).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from load_and_clean_data().
    output_path : str or Path
        Where to save (e.g., 'data/processed/clean.parquet').
    compression : str, default 'snappy'
        Compression algorithm ('snappy', 'gzip', 'brotli', 'none').

    Examples
    --------
    >>> save_cleaned_parquet(df, 'data/processed/Friday_DDos_clean.parquet')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, compression=compression, index=False)
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved cleaned data to {output_path.name} ({file_size_mb:.1f} MB)")


def load_cleaned_parquet(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load cleaned DataFrame from Parquet (much faster than CSV).

    Parameters
    ----------
    input_path : str or Path
        Path to .parquet file.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with 'Label' and 'y' columns.

    Raises
    ------
    FileNotFoundError
        If parquet file does not exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {input_path}")

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded from {input_path.name}: {df.shape[0]:,} rows")
    return df


# ============================================================================
# UTILITY: FEATURE REMOVAL REPORT
# ============================================================================

def print_feature_removal_report(removed_info: Dict[str, int]) -> None:
    """
    Print a formatted summary of removed features (for thesis documentation).

    Parameters
    ----------
    removed_info : dict
        Output from build_feature_matrix() with keys:
        'leakage', 'non_numeric', 'constant'.

    Examples
    --------
    >>> _, _, _, info = build_feature_matrix(df)
    >>> print_feature_removal_report(info)
    """
    total = sum(removed_info.values())
    print("\n" + "=" * 60)
    print("FEATURE REMOVAL REPORT")
    print("=" * 60)
    print(f"  Data leakage features : {removed_info.get('leakage', 0):3d}")
    print(f"  Non-numeric features  : {removed_info.get('non_numeric', 0):3d}")
    print(f"  Constant features     : {removed_info.get('constant', 0):3d}")
    print("-" * 60)
    print(f"  Total removed         : {total:3d}")
    print("=" * 60)