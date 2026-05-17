"""
Pairwise Statistical Tests: McNemar + Paired Bootstrap.
Dataset: CIC-IDS2017

Три фиксированные пары:
  1. XGBoost(v1_default)  vs  LogisticRegression(v1_default)
  2. XGBoost(v1_default)  vs  CatBoost(v2_tuned)
  3. XGBoost(v1_default)  vs  XGBoost(v2_tuned)

Bootstrap CI — для всех 6 моделей (v1 и v2).

Формат кэша предсказаний: dict с ключами y_pred, y_proba
y_true берётся из data/processed/processed_43d973e2.joblib (ключ y_test)

Артефакты:
  experiments/statistical_tests/bootstrap_ci_all_models.csv
  experiments/statistical_tests/pairwise_comparisons.csv
  experiments/statistical_tests/pairwise_comparisons_report.txt

Usage:
    python src/statistical_tests/run_tests.py
"""

import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.eval_utils import bootstrap_ci, mcnemar_test, paired_bootstrap_test

V1_DIR   = ROOT / "experiments/boosting/v1_default/20260222_190338/cache"
V2_DIR   = ROOT / "experiments/boosting/v2_tuned/cache"
PROC_PATH = ROOT / "data/processed/processed_43d973e2.joblib"
OUT_DIR  = ROOT / "experiments/statistical_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_y_true():
    """Загружает y_test из кэша предобработки."""
    cache = joblib.load(PROC_PATH)
    return np.array(cache["y_test"])


def load_preds(cache_dir, model_name):
    """Загружает (y_pred, y_proba) из pkl-кэша модели."""
    path = cache_dir / f"{model_name.lower()}_predictions.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Кэш не найден: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data["y_pred"]), np.array(data["y_proba"])


def compute_bootstrap_ci_all(y_true, models_dict, n_iterations=1000):
    """
    95% Bootstrap CI для F1, ROC-AUC, PR-AUC.
    bootstrap_ci(y_true, y_pred, metric_func, n_iterations, random_state)
    -> (point_estimate, ci_lower, ci_upper)
    """
    rows = []
    for label, (y_pred, y_proba) in models_dict.items():
        f1_pt, f1_lo, f1_hi = bootstrap_ci(
            y_true, y_pred,
            metric_func=lambda yt, yp: f1_score(yt, yp, zero_division=0),
            n_iterations=n_iterations, random_state=42,
        )
        roc_pt, roc_lo, roc_hi = bootstrap_ci(
            y_true, y_proba,
            metric_func=roc_auc_score,
            n_iterations=n_iterations, random_state=42,
        )
        pr_pt, pr_lo, pr_hi = bootstrap_ci(
            y_true, y_proba,
            metric_func=average_precision_score,
            n_iterations=n_iterations, random_state=42,
        )
        rows.append({
            "Model":            label,
            "F1":               round(f1_pt,  6),
            "F1_CI_lower":      round(f1_lo,  6),
            "F1_CI_upper":      round(f1_hi,  6),
            "ROC_AUC":          round(roc_pt, 6),
            "ROC_AUC_CI_lower": round(roc_lo, 6),
            "ROC_AUC_CI_upper": round(roc_hi, 6),
            "PR_AUC":           round(pr_pt,  6),
            "PR_AUC_CI_lower":  round(pr_lo,  6),
            "PR_AUC_CI_upper":  round(pr_hi,  6),
        })
        print(f"  {label}: F1={f1_pt:.5f}  95%CI=[{f1_lo:.5f}, {f1_hi:.5f}]")
    return pd.DataFrame(rows)


def run_pairwise_tests(y_true, pairs, n_iterations=1000):
    """
    mcnemar_test(y_true, y_pred_a, y_pred_b) -> (chi2, p_value)
    paired_bootstrap_test(y_true, y_pred_a, y_pred_b, metric_func,
                          n_iterations, random_state)
      -> dict: observed_diff, ci_lower_95, ci_upper_95, p_value
    """
    rows = []
    lines = []

    for label_a, y_pred_a, label_b, y_pred_b in pairs:
        pair_name = f"{label_a} vs {label_b}"
        print(f"\n  Пара: {pair_name}")

        chi2, p_mcn = mcnemar_test(y_true, y_pred_a, y_pred_b)

        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)
        b = int(np.sum( correct_a & ~correct_b))
        c = int(np.sum(~correct_a &  correct_b))

        bs = paired_bootstrap_test(
            y_true, y_pred_a, y_pred_b,
            metric_func=lambda yt, yp: f1_score(yt, yp, zero_division=0),
            n_iterations=n_iterations, random_state=42,
        )
        obs_diff = bs["observed_diff"]
        bs_lo    = bs["ci_lower_95"]
        bs_hi    = bs["ci_upper_95"]
        bs_p     = bs["p_value"]

        f1_a = f1_score(y_true, y_pred_a, zero_division=0)
        f1_b = f1_score(y_true, y_pred_b, zero_division=0)

        rows.append({
            "Pair":                 pair_name,
            "Model_A":              label_a,
            "Model_B":              label_b,
            "F1_A":                 round(f1_a, 6),
            "F1_B":                 round(f1_b, 6),
            "F1_diff_A_minus_B":    round(f1_a - f1_b, 6),
            "McNemar_b":            b,
            "McNemar_c":            c,
            "McNemar_chi2":         round(chi2, 4),
            "McNemar_p":            f"{p_mcn:.4e}",
            "McNemar_significant":  "YES" if p_mcn < 0.05 else "NO",
            "PairedBS_diff":        round(obs_diff, 6),
            "PairedBS_CI_lower":    round(bs_lo, 6),
            "PairedBS_CI_upper":    round(bs_hi, 6),
            "PairedBS_p":           f"{bs_p:.4e}",
            "PairedBS_significant": "YES" if bs_p < 0.05 else "NO",
        })

        sig_m = "ЗНАЧИМО" if p_mcn < 0.05 else "не значимо"
        sig_b = "ЗНАЧИМО" if bs_p  < 0.05 else "не значимо"
        lines += [
            f"\n{'='*60}",
            f"Пара: {pair_name}",
            f"  F1({label_a}) = {f1_a:.6f}",
            f"  F1({label_b}) = {f1_b:.6f}",
            f"  Разность A−B  = {f1_a - f1_b:+.6f}",
            f"  McNemar: b={b}, c={c}, χ²={chi2:.4f}, p={p_mcn:.4e} → {sig_m}",
            f"  Paired Bootstrap: diff={obs_diff:+.6f}, "
            f"95%CI=[{bs_lo:.6f}, {bs_hi:.6f}], p={bs_p:.4e} → {sig_b}",
        ]
        print(f"    McNemar χ²={chi2:.4f}, p={p_mcn:.4e} ({sig_m})")
        print(f"    Bootstrap diff={obs_diff:+.6f}, 95%CI=[{bs_lo:.6f},{bs_hi:.6f}],"
              f" p={bs_p:.4e} ({sig_b})")

    return pd.DataFrame(rows), "\n".join(lines)


def main():
    print("=" * 60)
    print("СТАТИСТИЧЕСКИЕ ТЕСТЫ")
    print("=" * 60)

    print("\n[1] Загрузка y_true из кэша предобработки...")
    y_true = load_y_true()
    print(f"  Тестовая выборка: {len(y_true):,} примеров, "
          f"{int(y_true.sum()):,} атакующих ({y_true.mean()*100:.1f}%)")

    print("\n[2] Загрузка предсказаний моделей...")
    try:
        y_pred_xgb_v1, y_proba_xgb_v1 = load_preds(V1_DIR, "xgboost")
        y_pred_cb_v1,  y_proba_cb_v1  = load_preds(V1_DIR, "catboost")
        y_pred_lr_v1,  y_proba_lr_v1  = load_preds(V1_DIR, "logisticregression")

        y_pred_xgb_v2, y_proba_xgb_v2 = load_preds(V2_DIR, "xgboost")
        y_pred_cb_v2,  y_proba_cb_v2  = load_preds(V2_DIR, "catboost")
        y_pred_lr_v2,  y_proba_lr_v2  = load_preds(V2_DIR, "logisticregression")
    except FileNotFoundError as e:
        print(f"\nОШИБКА: {e}")
        sys.exit(1)
    print("  Все кэши загружены.")

    print("\n[3] Bootstrap CI для всех 6 моделей (1000 итераций)...")
    all_models = {
        "XGBoost(v1_default)": (y_pred_xgb_v1, y_proba_xgb_v1),
        "CatBoost(v1_default)":(y_pred_cb_v1,  y_proba_cb_v1),
        "LR(v1_default)":      (y_pred_lr_v1,  y_proba_lr_v1),
        "XGBoost(v2_tuned)":   (y_pred_xgb_v2, y_proba_xgb_v2),
        "CatBoost(v2_tuned)":  (y_pred_cb_v2,  y_proba_cb_v2),
        "LR(v2_tuned)":        (y_pred_lr_v2,  y_proba_lr_v2),
    }
    ci_df = compute_bootstrap_ci_all(y_true, all_models)
    ci_path = OUT_DIR / "bootstrap_ci_all_models.csv"
    ci_df.to_csv(ci_path, index=False, encoding="utf-8")
    print(f"\n  → {ci_path.relative_to(ROOT)}")

    print("\n[4] Три фиксированные пары (McNemar + Paired Bootstrap)...")
    pairs = [
        ("XGBoost(v1_default)", y_pred_xgb_v1, "LR(v1_default)",     y_pred_lr_v1),
        ("XGBoost(v1_default)", y_pred_xgb_v1, "CatBoost(v2_tuned)", y_pred_cb_v2),
        ("XGBoost(v1_default)", y_pred_xgb_v1, "XGBoost(v2_tuned)",  y_pred_xgb_v2),
    ]
    pairs_df, report = run_pairwise_tests(y_true, pairs)

    pairs_path  = OUT_DIR / "pairwise_comparisons.csv"
    report_path = OUT_DIR / "pairwise_comparisons_report.txt"
    pairs_df.to_csv(pairs_path, index=False, encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")

    print(f"\n  → {pairs_path.relative_to(ROOT)}")
    print(f"  → {report_path.relative_to(ROOT)}")
    print("\n" + "=" * 60)
    print("ГОТОВО.")
    print("=" * 60)


if __name__ == "__main__":
    main()
