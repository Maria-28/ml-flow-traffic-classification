"""
PortScan Detection Research Framework: Boosting vs Baseline Comparison.
Dataset: CIC-IDS2017 (Friday-PortScan)

Extended scenario: same pipeline as DDoS, applied to PortScan classification.
Labels: BENIGN vs PortScan (instead of BENIGN vs DDoS).

Usage:
    python src/portscan/boosting_portscan.py --config configs/portscan/portscan_baseline.yaml

Author: [Your Name]
Thesis: Master's Dissertation, [University], 2024
License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import yaml
import joblib
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# ============================================================================
# OPTIONAL DEPENDENCIES (graceful degradation)
# ============================================================================

XGBOOST_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None  # type: ignore

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = None  # type: ignore

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Автоопределение корня проекта (поддержка разных структур)
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent

# Fallback: если структура другая, используем CWD
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))

# Импорт утилит проекта (с graceful fallback)
try:
    from src.utils.data_utils import (
        load_and_clean_data,
        build_feature_matrix,
        temporal_split,
    )
    from src.utils.eval_utils import (
        calculate_metrics,
        compare_models,
        save_results_table,
        analyze_errors,
    )
    from src.utils.plot_utils import (
        plot_confusion_matrix,
        plot_roc_curve,
        plot_precision_recall_curve,
        plot_feature_importance,
        plot_roc_curves_comparison,
        plot_pr_curves_comparison,
        plot_f1_comparison,
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    UTILS_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    path: str = "data/raw/Friday-PortScan.pcap_ISCX.csv"
    train_ratio: float = 0.80
    remove_leakage: bool = True
    cache_processed: bool = True
    cache_dir: str = "data/processed"
    expected_labels: list = field(default_factory=lambda: ["BENIGN", "PortScan"])
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.5 <= self.train_ratio <= 0.95:
            raise ValueError(
                f"train_ratio must be in [0.5, 0.95], got {self.train_ratio}"
            )


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    
    enabled: bool = True
    use_scaler: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """
    Master configuration for the entire experiment.
    
    Supports loading from YAML file with validation.
    """
    
    # Experiment metadata
    name: str = "boosting_vs_baseline"
    random_state: int = 42
    
    # Cross-validation settings
    cross_validation: bool = False
    cv_folds: int = 5
    
    # Persistence settings
    save_models: bool = True
    cache_predictions: bool = True
    results_root: str = "experiments/boosting"
    
    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)
    
    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize defaults and validate."""
        # Convert nested dicts to dataclasses if needed
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        
        # Initialize default models if none provided
        if not self.models:
            self.models = self._default_models()
        else:
            # Convert dicts to ModelConfig
            for name, cfg in self.models.items():
                if isinstance(cfg, dict):
                    self.models[name] = ModelConfig(**cfg)
        
        # Validate CV folds
        if self.cross_validation and not 2 <= self.cv_folds <= 20:
            raise ValueError(
                f"cv_folds must be in [2, 20], got {self.cv_folds}"
            )
    
    def _default_models(self) -> Dict[str, ModelConfig]:
        """Return default model configurations."""
        return {
            "LogisticRegression": ModelConfig(
                enabled=True,
                use_scaler=True,
                params={
                    "max_iter": 1000,
                    "n_jobs": -1,
                    "random_state": self.random_state,
                },
            ),
            "DecisionTree": ModelConfig(
                enabled=True,
                use_scaler=False,
                params={
                    "max_depth": 10,
                    "random_state": self.random_state,
                },
            ),
            "RandomForest": ModelConfig(
                enabled=True,
                use_scaler=False,
                params={
                    "n_estimators": 200,
                    "max_depth": 15,
                    "n_jobs": -1,
                    "random_state": self.random_state,
                },
            ),
            "XGBoost": ModelConfig(
                enabled=True,
                use_scaler=False,
                params={
                    "n_estimators": 500,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "n_jobs": -1,
                    "random_state": self.random_state,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                },
            ),
            "CatBoost": ModelConfig(
                enabled=True,
                use_scaler=False,
                params={
                    "iterations": 500,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "random_seed": self.random_state,
                    "verbose": 0,
                    "allow_writing_files": False,
                },
            ),
        }
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file.
            
        Returns
        -------
        ExperimentConfig
            Validated configuration object.
            
        Raises
        ------
        FileNotFoundError
            If YAML file does not exist.
        ValueError
            If YAML content is invalid.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        
        if raw_config is None:
            raise ValueError(f"Config file is empty: {path}")
        
        return cls(**raw_config)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for serialization
        config_dict = self._to_serializable_dict()
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dictionary."""
        result = {
            "name": self.name,
            "random_state": self.random_state,
            "cross_validation": self.cross_validation,
            "cv_folds": self.cv_folds,
            "save_models": self.save_models,
            "cache_predictions": self.cache_predictions,
            "results_root": self.results_root,
            "data": asdict(self.data),
            "models": {
                name: asdict(cfg) for name, cfg in self.models.items()
            },
        }
        return result


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
    
    Tests whether the two classifiers have significantly different
    error rates on the same test set.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred_a : np.ndarray
        Predictions from classifier A (baseline).
    y_pred_b : np.ndarray
        Predictions from classifier B (candidate).
    continuity_correction : bool, default True
        Apply Edwards' continuity correction for small samples.
        
    Returns
    -------
    chi2_statistic : float
        McNemar's chi-squared statistic.
    p_value : float
        Two-tailed p-value.
        
    Notes
    -----
    The contingency table counts:
    - b: samples correct by A but incorrect by B
    - c: samples incorrect by A but correct by B
    
    If p < 0.05, the classifiers are significantly different.
    
    References
    ----------
    McNemar, Q. (1947). "Note on the sampling error of the difference
    between correlated proportions or percentages". Psychometrika.
    """
    # Compute correctness
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)
    
    # Build contingency table
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    b = np.sum((correct_a == 1) & (correct_b == 0))
    c = np.sum((correct_a == 0) & (correct_b == 1))
    
    # Handle edge case: no disagreements
    if b + c == 0:
        return 0.0, 1.0
    
    # McNemar's chi-squared statistic
    if continuity_correction:
        # Edwards' correction for small samples
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)
    
    # p-value from chi-squared distribution (df=1)
    from scipy import stats
    p_value = float(stats.chi2.sf(chi2, df=1))
    
    return float(chi2), p_value


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    metric_func : callable
        Function that takes (y_true, y_pred) and returns a float.
    n_iterations : int, default 1000
        Number of bootstrap iterations.
    confidence_level : float, default 0.95
        Confidence level (e.g., 0.95 for 95% CI).
    random_state : int, default 42
        Random seed for reproducibility.
        
    Returns
    -------
    point_estimate : float
        Metric computed on original data.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    # Point estimate on original data
    point_estimate = metric_func(y_true, y_pred)
    
    # Bootstrap sampling
    bootstrap_scores = []
    for _ in range(n_iterations):
        indices = rng.randint(0, n_samples, size=n_samples)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Percentile method for CI
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))
    
    return point_estimate, ci_lower, ci_upper


def f1_score_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score helper for bootstrap CI."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, zero_division=0))


# ============================================================================
# MODEL REGISTRY
# ============================================================================


def get_model_class(model_name: str) -> Optional[Type]:
    """
    Get model class by name with availability check.
    
    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "RandomForest", "XGBoost").
        
    Returns
    -------
    model_class : Type or None
        Model class if available, None if not installed.
    """
    model_registry: Dict[str, Tuple[Optional[Type], bool]] = {
        "LogisticRegression": (LogisticRegression, True),
        "DecisionTree": (DecisionTreeClassifier, True),
        "RandomForest": (RandomForestClassifier, True),
        "XGBoost": (XGBClassifier, XGBOOST_AVAILABLE),
        "CatBoost": (CatBoostClassifier, CATBOOST_AVAILABLE),
    }
    
    if model_name not in model_registry:
        return None
    
    model_class, is_available = model_registry[model_name]
    
    if not is_available:
        return None
    
    return model_class


# ============================================================================
# MAIN EXPERIMENT CLASS
# ============================================================================


class DDoSExperiment:
    """
    Research Framework for DDoS Attack Detection.
    
    Handles the complete ML pipeline: data loading, model training,
    evaluation, statistical testing, and artifact persistence.
    
    Attributes
    ----------
    config : ExperimentConfig
        Experiment configuration.
    run_id : str
        Unique identifier for this experiment run.
    exp_dir : Path
        Directory for all experiment artifacts.
    trained_models : dict
        Dictionary of trained model pipelines.
    results : list
        List of evaluation results for each model.
        
    Examples
    --------
    >>> config = ExperimentConfig.from_yaml("config.yaml")
    >>> experiment = DDoSExperiment(config)
    >>> experiment.run()
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        verbose: bool = True,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Initialize experiment.
        
        Parameters
        ----------
        config : ExperimentConfig
            Validated experiment configuration.
        verbose : bool, default True
            Enable console logging.
        run_id : str, optional
            Custom run ID. If None, generates from timestamp.
        """
        self.config = config
        self.verbose = verbose
        
        # Generate unique run ID
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directory structure
        self.exp_dir = Path(config.results_root) / self.run_id
        self.model_dir = self.exp_dir / "models"
        self.plot_dir = self.exp_dir / "figures"
        self.table_dir = self.exp_dir / "tables"
        self.log_dir = self.exp_dir / "logs"
        self.cache_dir = self.exp_dir / "cache"
        
        for directory in [
            self.model_dir, self.plot_dir, self.table_dir,
            self.log_dir, self.cache_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Log startup info
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"DDoS Detection Experiment: {config.name}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Output directory: {self.exp_dir}")
        self.logger.info(f"{'=' * 60}")
        
        # Check optional dependencies
        self._check_dependencies()
        
        # Initialize state containers
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_names: List[str] = []
        
        self.trained_models: Dict[str, Pipeline] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.probabilities: Dict[str, np.ndarray] = {}
        self.results: List[Dict[str, Any]] = []
        self.statistical_tests: List[Dict[str, Any]] = []
    
    def _setup_logging(self) -> None:
        """Configure logging to file and console."""
        self.logger = logging.getLogger(f"Experiment_{self.run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Prevent duplicate handlers
        
        # File handler: all levels
        log_file = self.log_dir / "experiment.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)
        
        # Console handler: INFO+ only
        if self.verbose:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            self.logger.addHandler(ch)
    
    def _check_dependencies(self) -> None:
        """Check and log availability of optional dependencies."""
        if not UTILS_AVAILABLE:
            self.logger.error(
                f"Required utilities not found: {_IMPORT_ERROR}\n"
                f"Ensure data_utils.py, eval_utils.py, plot_utils.py "
                f"are in src/utils/"
            )
            raise ImportError(_IMPORT_ERROR)
        
        if not XGBOOST_AVAILABLE:
            self.logger.warning(
                "XGBoost not installed. XGBoost model will be skipped. "
                "Install with: pip install xgboost"
            )
        
        if not CATBOOST_AVAILABLE:
            self.logger.warning(
                "CatBoost not installed. CatBoost model will be skipped. "
                "Install with: pip install catboost"
            )
    
    # ────────────────────────────────────────────────────────────────────
    # DATA PREPARATION
    # ────────────────────────────────────────────────────────────────────
    
    def prepare_data(self) -> None:
        """
        Load, clean, and split data with validation and optional caching.
        
        Raises
        ------
        FileNotFoundError
            If data file does not exist.
        ValueError
            If data is empty or invalid after cleaning.
        """
        self.logger.info("=" * 40)
        self.logger.info("STAGE 1: Data Preparation")
        self.logger.info("=" * 40)
        
        data_path = PROJECT_ROOT / self.config.data.path
        
        # Check for cached processed data
        cache_path = self._get_data_cache_path(data_path)
        
        if cache_path.exists() and self.config.data.cache_processed:
            self.logger.info(f"Loading cached data from: {cache_path.name}")
            cached = joblib.load(cache_path)
            
            self.X_train = cached["X_train"]
            self.X_test = cached["X_test"]
            self.y_train = cached["y_train"]
            self.y_test = cached["y_test"]
            self.feature_names = cached["feature_names"]
            
            self.logger.info(
                f"Loaded: {len(self.X_train):,} train, "
                f"{len(self.X_test):,} test samples"
            )
            return
        
        # Load raw data
        self.logger.info(f"Loading data from: {data_path.name}")
        
        try:
            df = load_and_clean_data(
                data_path,
                expected_labels=list(self.config.data.expected_labels),
                verbose=self.verbose,
            )
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {data_path}")
            self.logger.error(f"Current working directory: {Path.cwd()}")
            raise
        
        # Build feature matrix
        self.logger.info("Building feature matrix...")
        X, y, self.feature_names, removed_info = build_feature_matrix(
            df,
            remove_leakage=self.config.data.remove_leakage,
            verbose=self.verbose,
        )
        
        self.logger.info(
            f"Features after preprocessing: {len(self.feature_names)} "
            f"(removed: leakage={removed_info['leakage']}, "
            f"constant={removed_info['constant']})"
        )
        
        # Temporal split (no shuffling!)
        self.logger.info(
            f"Temporal split: {self.config.data.train_ratio:.0%} train / "
            f"{1 - self.config.data.train_ratio:.0%} test"
        )
        
        self.X_train, self.X_test, self.y_train, self.y_test = temporal_split(
            X, y,
            train_ratio=self.config.data.train_ratio,
            verbose=self.verbose,
        )
        
        # Cache processed data
        if self.config.data.cache_processed:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "X_train": self.X_train,
                    "X_test": self.X_test,
                    "y_train": self.y_train,
                    "y_test": self.y_test,
                    "feature_names": self.feature_names,
                },
                cache_path,
            )
            self.logger.info(f"Cached processed data to: {cache_path.name}")
        
        self.logger.info(
            f"Data ready: Train={len(self.X_train):,}, Test={len(self.X_test):,}"
        )
    
    def _get_data_cache_path(self, data_path: Path) -> Path:
        """Generate cache path based on data file and config hash."""
        # Create hash of relevant config parameters
        config_str = (
            f"{data_path.name}_{self.config.data.train_ratio}_"
            f"{self.config.data.remove_leakage}"
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        cache_dir = PROJECT_ROOT / self.config.data.cache_dir
        return cache_dir / f"processed_{config_hash}.joblib"
    
    # ────────────────────────────────────────────────────────────────────
    # MODEL TRAINING
    # ────────────────────────────────────────────────────────────────────
    
    def train(
        self,
        force_retrain: bool = False,
        model_filter: Optional[Literal["all", "baseline", "boosting"]] = None,
    ) -> None:
        """
        Train all enabled models with progress tracking and caching.
        
        Parameters
        ----------
        force_retrain : bool, default False
            If True, ignore cached models and retrain all.
        model_filter : {'all', 'baseline', 'boosting'}, optional
            Filter models by category:
            - 'baseline': LogisticRegression, DecisionTree, RandomForest
            - 'boosting': XGBoost, CatBoost
            - 'all' or None: all models
        """
        self.logger.info("=" * 40)
        self.logger.info("STAGE 2: Model Training")
        self.logger.info("=" * 40)
        
        if self.X_train is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")
        
        # Determine which models to train
        models_to_train = self._filter_models(model_filter)
        
        if not models_to_train:
            self.logger.warning("No models to train after filtering!")
            return
        
        self.logger.info(f"Training {len(models_to_train)} models: {models_to_train}")
        
        # Training loop with progress bar
        for model_name in tqdm(
            models_to_train,
            desc="Training",
            disable=not self.verbose,
        ):
            self._train_single_model(model_name, force_retrain)
        
        self.logger.info(
            f"Training complete. {len(self.trained_models)} models ready."
        )
    
    def _filter_models(
        self,
        model_filter: Optional[str],
    ) -> List[str]:
        """Get list of models to train based on filter."""
        baseline_models = {"LogisticRegression", "DecisionTree", "RandomForest"}
        boosting_models = {"XGBoost", "CatBoost"}
        
        filtered = []
        
        for name, cfg in self.config.models.items():
            # Skip disabled models
            if not cfg.enabled:
                continue
            
            # Check availability
            if get_model_class(name) is None:
                self.logger.warning(f"Skipping {name}: not available")
                continue
            
            # Apply category filter
            if model_filter == "baseline" and name not in baseline_models:
                continue
            if model_filter == "boosting" and name not in boosting_models:
                continue
            
            filtered.append(name)
        
        return filtered
    
    def _train_single_model(
        self,
        model_name: str,
        force_retrain: bool,
    ) -> None:
        model_path = self.model_dir / f"{model_name.lower()}.joblib"
        
        # Check cache
        if model_path.exists() and not force_retrain:
            self.logger.info(f"Loading cached model: {model_name}")
            try:
                pipeline = joblib.load(model_path)
                self.trained_models[model_name] = pipeline
                return
            except Exception as e:
                self.logger.warning(
                    f"Failed to load cached {model_name}: {e}. Retraining..."
                )
        
        # Build pipeline
        self.logger.info(f"Training: {model_name}")
        
        try:
            pipeline = self._build_pipeline(model_name)
            
            # ====================================================================
            # FIX FOR EARLY STOPPING
            # ====================================================================
            start_time = time.time()
            
            # Check if this model needs early stopping with eval_set
            is_xgb = 'xgb' in model_name.lower() or model_name == 'XGBoost'
            is_catboost = 'catboost' in model_name.lower() or model_name == 'CatBoost'
            
            model_params = self.config.models[model_name].params
            needs_eval_set = False
            
            if is_xgb and 'early_stopping_rounds' in model_params:
                needs_eval_set = True
                self.logger.info(f"  XGBoost with early stopping detected")
            elif is_catboost and (
                model_params.get('early_stopping_rounds') or
                model_params.get('use_best_model', False)
            ):
                needs_eval_set = True
                self.logger.info(f"  CatBoost with early stopping / use_best_model detected")
            
            if needs_eval_set:
                # Create validation set (20% of training data)
                from sklearn.model_selection import train_test_split
                X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                    self.X_train, self.y_train,
                    test_size=0.2,
                    random_state=self.config.random_state,
                    stratify=self.y_train
                )
                pipeline.fit(
                    X_train_sub, y_train_sub,
                    clf__eval_set=[(X_val, y_val)],
                    clf__verbose=False
                )
            else:
                # Standard fit
                pipeline.fit(self.X_train, self.y_train)
            
            train_time = time.time() - start_time
            # ====================================================================
            
            self.trained_models[model_name] = pipeline
            
            # Save model
            if self.config.save_models:
                joblib.dump(pipeline, model_path)
            
            self.logger.info(
                f"Trained {model_name} in {train_time:.2f}s "
                f"(saved: {model_path.name})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to train {model_name}: {e}")
            raise
    
    def _build_pipeline(self, model_name: str) -> Pipeline:
        """
        Build sklearn Pipeline for a model.
        
        Applies StandardScaler if configured, then the classifier.
        """
        cfg = self.config.models[model_name]
        model_class = get_model_class(model_name)
        
        if model_class is None:
            raise ValueError(f"Model class not available: {model_name}")
        
        steps = []
        
        # Optional scaler
        if cfg.use_scaler:
            steps.append(("scaler", StandardScaler()))
        
        # Classifier
        clf = model_class(**cfg.params)
        steps.append(("clf", clf))
        
        return Pipeline(steps)
    
    # ────────────────────────────────────────────────────────────────────
    # EVALUATION
    # ────────────────────────────────────────────────────────────────────
    
    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate all trained models and generate artifacts.
        
        Returns
        -------
        results_df : pd.DataFrame
            Comparison table sorted by F1 score.
        """
        self.logger.info("=" * 40)
        self.logger.info("STAGE 3: Evaluation")
        self.logger.info("=" * 40)
        
        if not self.trained_models:
            raise RuntimeError("No trained models. Call train() first.")
        
        # Collect predictions and metrics
        model_payloads = []  # For comparative plots
        
        for model_name, pipeline in tqdm(
            self.trained_models.items(),
            desc="Evaluating",
            disable=not self.verbose,
        ):
            metrics = self._evaluate_single_model(model_name, pipeline)
            self.results.append(metrics)
            
            # Prepare payload for multi-model plots
            model_payloads.append({
                "name": model_name,
                "y_true": self.y_test,
                "y_proba": self.probabilities[model_name],
            })
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)
        results_df["rank"] = range(1, len(results_df) + 1)
        
        # Reorder columns
        cols = ["rank", "Model", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        cols = [c for c in cols if c in results_df.columns]
        results_df = results_df[cols]
        
        # Save comparison table
        self._save_comparison_table(results_df)
        
        # Generate comparative plots
        self._generate_comparative_plots(model_payloads)
        
        # Log summary
        self.logger.info("\n" + "=" * 50)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 50)
        for _, row in results_df.iterrows():
            self.logger.info(
                f"  #{int(row['rank'])} {row['Model']}: "
                f"F1={row['f1']:.4f}, AUC={row['roc_auc']:.4f}"
            )
        
        return results_df
    
    def _evaluate_single_model(
        self,
        model_name: str,
        pipeline: Pipeline,
    ) -> Dict[str, Any]:
        """Evaluate a single model and generate individual plots."""
        
        # Check prediction cache
        pred_cache = self.cache_dir / f"{model_name.lower()}_predictions.pkl"
        
        if pred_cache.exists() and self.config.cache_predictions:
            with open(pred_cache, "rb") as f:
                cached = pickle.load(f)
                y_pred = cached["y_pred"]
                y_proba = cached["y_proba"]
        else:
            # Generate predictions
            y_pred = pipeline.predict(self.X_test)
            y_proba = pipeline.predict_proba(self.X_test)[:, 1]
            
            # Cache predictions
            if self.config.cache_predictions:
                with open(pred_cache, "wb") as f:
                    pickle.dump({"y_pred": y_pred, "y_proba": y_proba}, f)
        
        # Store for later use
        self.predictions[model_name] = y_pred
        self.probabilities[model_name] = y_proba
        
        # Calculate metrics
        metrics = calculate_metrics(
            self.y_test, y_pred, y_proba,
            verbose=False,
        )
        metrics["Model"] = model_name
        
        # Generate individual plots
        self._generate_model_plots(model_name, pipeline, y_pred, y_proba)
        
        return metrics
    
    def _generate_model_plots(
        self,
        model_name: str,
        pipeline: Pipeline,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> None:
        """Generate plots for a single model."""
        prefix = model_name.lower()
        
        # Confusion matrix
        plot_confusion_matrix(
            self.y_test, y_pred,
            title=f"{model_name} Confusion Matrix",
            output_path=self.plot_dir / f"{prefix}_confusion_matrix.png",
            verbose=False,
        )
        
        # Confusion matrix (normalized)
        plot_confusion_matrix(
            self.y_test, y_pred,
            normalize=True,
            title=f"{model_name} Confusion Matrix (Normalized)",
            output_path=self.plot_dir / f"{prefix}_confusion_matrix_norm.png",
            verbose=False,
        )
        
        # ROC curve
        plot_roc_curve(
            self.y_test, y_proba,
            label=model_name,
            output_path=self.plot_dir / f"{prefix}_roc.png",
            verbose=False,
        )
        
        # PR curve
        plot_precision_recall_curve(
            self.y_test, y_proba,
            label=model_name,
            output_path=self.plot_dir / f"{prefix}_pr.png",
            verbose=False,
        )
        
        # Feature importance (for tree-based models)
        clf = pipeline.named_steps.get("clf")
        if hasattr(clf, "feature_importances_"):
            plot_feature_importance(
                clf.feature_importances_,
                self.feature_names,
                top_k=15,
                title=f"{model_name} Feature Importance",
                output_path=self.plot_dir / f"{prefix}_feature_importance.png",
                verbose=False,
            )
    
    def _generate_comparative_plots(
        self,
        model_payloads: List[Dict[str, Any]],
    ) -> None:
        """Generate multi-model comparison plots."""
        self.logger.info("Generating comparative plots...")
        
        # Multi-model ROC curves
        plot_roc_curves_comparison(
            model_payloads,
            title="ROC Curves: All Models",
            output_path=self.plot_dir / "comparison_roc.png",
            verbose=False,
        )
        
        # Multi-model PR curves
        plot_pr_curves_comparison(
            model_payloads,
            title="Precision-Recall Curves: All Models",
            output_path=self.plot_dir / "comparison_pr.png",
            verbose=False,
        )
        
        # F1 score bar chart
        results_df = pd.DataFrame(self.results)
        plot_f1_comparison(
            results_df["Model"].tolist(),
            results_df["f1"].tolist(),
            title="F1 Score Comparison",
            output_path=self.plot_dir / "comparison_f1_bars.png",
            verbose=False,
        )
    
    def _save_comparison_table(self, results_df: pd.DataFrame) -> None:
        """Save comparison table in multiple formats."""
        # Rename columns for publication
        display_df = results_df.copy()
        display_df.columns = [
            "Rank", "Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"
        ]
        
        save_results_table(
            display_df,
            self.table_dir / "model_comparison",
            caption="Performance comparison of DDoS detection models",
            label="tab:model_comparison",
            verbose=False,
        )
        
        self.logger.info(f"Saved comparison table to: {self.table_dir}")
    
    # ────────────────────────────────────────────────────────────────────
    # STATISTICAL SIGNIFICANCE
    # ────────────────────────────────────────────────────────────────────
    
    def test_significance(
        self,
        baseline_model: Optional[str] = None,
        candidate_model: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Test statistical significance of model improvements.
        
        Uses McNemar test to compare classifiers and Bootstrap CI
        for F1 score confidence intervals.
        
        Parameters
        ----------
        baseline_model : str, optional
            Model to use as baseline. Default: worst performing model.
        candidate_model : str, optional
            Model to compare. Default: best performing model.
            
        Returns
        -------
        significance_df : pd.DataFrame
            Table with statistical test results.
        """
        self.logger.info("=" * 40)
        self.logger.info("STAGE 4: Statistical Significance Testing")
        self.logger.info("=" * 40)
        
        if len(self.results) < 2:
            self.logger.warning("Need at least 2 models for significance testing")
            return pd.DataFrame()
        
        # Sort by F1 to find best/worst
        results_sorted = sorted(self.results, key=lambda x: x["f1"])
        
        if baseline_model is None:
            baseline_model = results_sorted[0]["Model"]  # Worst
        
        if candidate_model is None:
            candidate_model = results_sorted[-1]["Model"]  # Best
        
        if baseline_model not in self.predictions:
            raise ValueError(f"Baseline model '{baseline_model}' not found")
        if candidate_model not in self.predictions:
            raise ValueError(f"Candidate model '{candidate_model}' not found")
        
        self.logger.info(
            f"Comparing: {candidate_model} (best) vs {baseline_model} (baseline)"
        )
        
        # McNemar test
        try:
            chi2, p_value = mcnemar_test(
                np.asarray(self.y_test),
                self.predictions[baseline_model],
                self.predictions[candidate_model],
            )
            
            significance = "YES" if p_value < 0.05 else "NO"
            
            self.logger.info(
                f"McNemar test: χ²={chi2:.4f}, p={p_value:.4f}"
            )
            
            if p_value < 0.05:
                self.logger.info(
                    f"✓ {candidate_model} is SIGNIFICANTLY better than "
                    f"{baseline_model} (p={p_value:.4f})"
                )
            else:
                self.logger.info(
                    f"✗ No significant difference between models (p={p_value:.4f})"
                )
        except ImportError:
            self.logger.warning("scipy not installed, skipping McNemar test")
            chi2, p_value = np.nan, np.nan
            significance = "N/A"
        
        # Bootstrap CI for best model
        self.logger.info(f"Computing 95% CI for {candidate_model} F1 score...")
        
        f1_point, f1_lower, f1_upper = bootstrap_confidence_interval(
            np.asarray(self.y_test),
            self.predictions[candidate_model],
            f1_score_metric,
            n_iterations=1000,
            random_state=self.config.random_state,
        )
        
        self.logger.info(
            f"{candidate_model} F1: {f1_point:.4f} "
            f"(95% CI: [{f1_lower:.4f}, {f1_upper:.4f}])"
        )
        
        # Build results table
        significance_results = {
            "Comparison": f"{candidate_model} vs {baseline_model}",
            "McNemar χ²": chi2,
            "p-value": p_value,
            "Significant (α=0.05)": significance,
            "Best Model F1": f1_point,
            "F1 95% CI Lower": f1_lower,
            "F1 95% CI Upper": f1_upper,
        }
        
        self.statistical_tests.append(significance_results)
        
        # Save to table
        sig_df = pd.DataFrame([significance_results])
        save_results_table(
            sig_df,
            self.table_dir / "statistical_significance",
            caption="Statistical significance test results",
            label="tab:significance",
            verbose=False,
        )
        
        return sig_df
    
    # ────────────────────────────────────────────────────────────────────
    # ERROR ANALYSIS
    # ────────────────────────────────────────────────────────────────────
    
    def analyze_best_model_errors(
        self,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform error analysis on the best (or specified) model.
        
        Parameters
        ----------
        model_name : str, optional
            Model to analyze. Default: best by F1 score.
            
        Returns
        -------
        error_stats : dict
            Error analysis statistics.
        """
        self.logger.info("=" * 40)
        self.logger.info("STAGE 5: Error Analysis")
        self.logger.info("=" * 40)
        
        if model_name is None:
            # Find best model
            best_result = max(self.results, key=lambda x: x["f1"])
            model_name = best_result["Model"]
        
        if model_name not in self.predictions:
            raise ValueError(f"Model '{model_name}' not found")
        
        self.logger.info(f"Analyzing errors for: {model_name}")
        
        y_pred = self.predictions[model_name]
        y_proba = self.probabilities.get(model_name)
        
        error_stats = analyze_errors(
            self.y_test,
            y_pred,
            self.X_test,
            model_name=model_name,
            output_dir=self.table_dir,
            y_proba=y_proba,
            verbose=self.verbose,
        )
        
        self.logger.info(
            f"Error analysis complete: "
            f"FP={error_stats['fp_count']}, FN={error_stats['fn_count']}"
        )
        
        return error_stats
    
    # ────────────────────────────────────────────────────────────────────
    # CROSS-VALIDATION (OPTIONAL)
    # ────────────────────────────────────────────────────────────────────
    
    def run_cross_validation(self, n_folds: int = 5) -> pd.DataFrame:
        """
        Run stratified cross-validation for all models.
        
        Parameters
        ----------
        n_folds : int, default 5
            Number of CV folds.
            
        Returns
        -------
        cv_results : pd.DataFrame
            Mean ± std for each metric and model.
        """
        self.logger.info("=" * 40)
        self.logger.info(f"CROSS-VALIDATION ({n_folds}-fold)")
        self.logger.info("=" * 40)
        
        if self.X_train is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")
        
        # Combine train + test for CV
        X_full = pd.concat([self.X_train, self.X_test], ignore_index=True)
        y_full = pd.concat([self.y_train, self.y_test], ignore_index=True)
        
        cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        
        cv_results = []
        
        for model_name in tqdm(
            list(self.trained_models.keys()),
            desc="CV Progress",
            disable=not self.verbose,
        ):
            self.logger.info(f"Cross-validating: {model_name}")
            
            pipeline = self._build_pipeline(model_name)
            
            # Get predictions from CV
            y_pred_cv = cross_val_predict(pipeline, X_full, y_full, cv=cv)
            y_proba_cv = cross_val_predict(
                pipeline, X_full, y_full, cv=cv, method="predict_proba"
            )[:, 1]
            
            # Calculate metrics
            metrics = calculate_metrics(y_full, y_pred_cv, y_proba_cv, verbose=False)
            metrics["Model"] = model_name
            metrics["CV_folds"] = n_folds
            
            cv_results.append(metrics)
        
        cv_df = pd.DataFrame(cv_results)
        cv_df = cv_df.sort_values("f1", ascending=False).reset_index(drop=True)
        
        # Save CV results
        save_results_table(
            cv_df,
            self.table_dir / "cross_validation_results",
            caption=f"{n_folds}-fold cross-validation results",
            label="tab:cv_results",
            verbose=False,
        )
        
        self.logger.info(f"Cross-validation results saved to: {self.table_dir}")
        
        return cv_df
    
    # ────────────────────────────────────────────────────────────────────
    # METADATA & REPRODUCIBILITY
    # ────────────────────────────────────────────────────────────────────
    
    def save_metadata(self) -> None:
        """Save experiment configuration and metadata for reproducibility."""
        self.logger.info("Saving experiment metadata...")
        
        # Save config as YAML
        self.config.to_yaml(self.exp_dir / "config.yaml")
        
        # Save comprehensive metadata as JSON
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "config": self.config._to_serializable_dict(),
            "features_used": self.feature_names,
            "train_samples": len(self.X_train) if self.X_train is not None else 0,
            "test_samples": len(self.X_test) if self.X_test is not None else 0,
            "models_trained": list(self.trained_models.keys()),
            "results_summary": self.results,
            "statistical_tests": self.statistical_tests,
            "dependencies": {
                "xgboost_available": XGBOOST_AVAILABLE,
                "catboost_available": CATBOOST_AVAILABLE,
            },
        }
        
        with open(self.exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Metadata saved to: {self.exp_dir / 'metadata.json'}")
    
    # ────────────────────────────────────────────────────────────────────
    # MAIN RUN METHOD
    # ────────────────────────────────────────────────────────────────────
    
    def run(
        self,
        force_retrain: bool = False,
        model_filter: Optional[str] = None,
        cv_folds: int = 0,
    ) -> pd.DataFrame:
        """
        Execute complete experiment pipeline.
        
        Parameters
        ----------
        force_retrain : bool, default False
            Ignore cached models and retrain.
        model_filter : str, optional
            Filter models ('all', 'baseline', 'boosting').
        cv_folds : int, default 0
            If > 0, run cross-validation with this many folds.
            
        Returns
        -------
        results_df : pd.DataFrame
            Final comparison table.
        """
        start_time = time.time()
        
        try:
            # 1. Data preparation
            self.prepare_data()
            
            # 2. Training
            self.train(force_retrain=force_retrain, model_filter=model_filter)
            
            # 3. Evaluation
            results_df = self.evaluate()
            
            # 4. Statistical significance
            if len(self.trained_models) >= 2:
                self.test_significance()
            
            # 5. Error analysis on best model
            self.analyze_best_model_errors()
            
            # 6. Cross-validation (optional)
            if cv_folds > 0:
                self.run_cross_validation(n_folds=cv_folds)
            
            # 7. Save metadata
            self.save_metadata()
            
            # Final summary
            total_time = time.time() - start_time
            self.logger.info("=" * 60)
            self.logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
            self.logger.info(f"All artifacts saved to: {self.exp_dir}")
            self.logger.info("=" * 60)
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.save_metadata()  # Save partial results
            raise


# ============================================================================
# CLI INTERFACE
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DDoS Detection Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python boosting_ddos.py
  python boosting_ddos.py --config custom.yaml --cv 5
  python boosting_ddos.py --models boosting --force-retrain
  python boosting_ddos.py --quiet
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--models",
        type=str,
        choices=["all", "baseline", "boosting"],
        default="all",
        help="Which models to train (default: all)",
    )
    
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Number of CV folds (0 = no CV, default: 0)",
    )
    
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached models and retrain all",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (logs still saved to file)",
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (default: timestamp)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Suppress warnings in quiet mode
    if args.quiet:
        warnings.filterwarnings("ignore")
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = ExperimentConfig()
        print("Using default configuration")
    
    # Override CV setting from CLI
    if args.cv > 0:
        config.cross_validation = True
        config.cv_folds = args.cv
    
    # Initialize and run experiment
    experiment = DDoSExperiment(
        config,
        verbose=not args.quiet,
        run_id=args.run_id,
    )
    
    results = experiment.run(
        force_retrain=args.force_retrain,
        model_filter=args.models if args.models != "all" else None,
        cv_folds=args.cv,
    )
    
    # Print final summary to console
    if not args.quiet:
        print("\n" + "=" * 60)
        print("🎯 FINAL RESULTS")
        print("=" * 60)
        print(results.to_string(index=False))
        print(f"\n📁 All artifacts: {experiment.exp_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()