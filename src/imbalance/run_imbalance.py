"""
Imbalance Study: влияние дисбаланса классов на качество XGBoost.
Dataset: CIC-IDS2017 (Friday-PortScan)

Методология:
  - Тестовая выборка фиксирована (temporal split 80/20, без сабсэмплинга)
  - Undersampling применяется ТОЛЬКО к обучающей выборке
  - Три уровня дисбаланса: 1:1, 1:10, 1:100 (PortScan:BENIGN)
  - Две конфигурации XGBoost: без балансировки и со scale_pos_weight
  - Дополнительно: threshold tuning XGBoost(v1_default) на DDoS-данных

Артефакты:
  experiments/imbalance/imbalance_results.csv     — 6 строк (3 уровня × 2 режима)
  experiments/imbalance/metrics_vs_imbalance.png  — линейный график
  experiments/imbalance/threshold_tuning_ddos.csv — оптимальный порог DDoS
  experiments/imbalance/pr_curve_ddos.png         — PR-кривая + оптимальный порог

Usage:
    python src/imbalance/run_imbalance.py
"""

import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.data_utils import load_and_clean_data, build_feature_matrix, temporal_split

DDOS_PROC_PATH = ROOT / "data/processed/processed_43d973e2.joblib"
DDOS_V1_CACHE  = ROOT / "experiments/boosting/v1_default/20260222_190338/cache"
OUT_DIR        = ROOT / "experiments/imbalance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
RATIOS = [
    ("1:1",   1),
    ("1:10",  10),
    ("1:100", 100),
]

# Параметры XGBoost — те же что в v1_default
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="logloss",
    verbosity=0,
)


def undersample(X, y, ratio_minority_to_majority, random_state=42):
    """
    Undersampling миноритарного класса (PortScan=1).
    ratio = n_minority / n_majority  →  нужный target_n_minority.
    """
    rng = np.random.RandomState(random_state)
    idx_maj = np.where(y == 0)[0]
    idx_min = np.where(y == 1)[0]

    n_maj = len(idx_maj)
    target_min = max(1, int(n_maj * ratio_minority_to_majority))
    target_min = min(target_min, len(idx_min))   # не больше чем есть

    chosen_min = rng.choice(idx_min, size=target_min, replace=False)
    idx_all = np.concatenate([idx_maj, chosen_min])
    rng.shuffle(idx_all)
    return X.iloc[idx_all], y.iloc[idx_all] if hasattr(y, "iloc") else y[idx_all]


def train_and_eval(X_train, y_train, X_test, y_test, scale_pos_weight=None):
    """Обучает XGBoost и возвращает метрики на тестовой выборке."""
    params = dict(XGB_PARAMS)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 6),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 6),
        "ROC_AUC":   round(roc_auc_score(y_test, y_proba), 6),
        "PR_AUC":    round(average_precision_score(y_test, y_proba), 6),
    }


def run_imbalance_study():
    """Основная часть: 6 экспериментов на PortScan."""
    print("\n[1] Загрузка PortScan-данных...")
    df = load_and_clean_data(
        ROOT / "data/raw/Friday-PortScan.pcap_ISCX.csv",
        expected_labels=["BENIGN", "PortScan"],
        verbose=False,
    )
    X, y, feature_names, _ = build_feature_matrix(df, verbose=False)
    X_train, X_test, y_train, y_test = temporal_split(X, y, verbose=False)

    n_min_train = int((y_train == 1).sum())
    n_maj_train = int((y_train == 0).sum())
    print(f"  Train исходный: {len(y_train):,} "
          f"(PortScan={n_min_train:,}, BENIGN={n_maj_train:,})")
    print(f"  Test (фиксирован): {len(y_test):,} "
          f"(PortScan={int((y_test==1).sum()):,}, BENIGN={int((y_test==0).sum()):,})")

    rows = []
    print("\n[2] Эксперименты (3 уровня × 2 режима)...")
    for ratio_label, ratio_val in RATIOS:
        # Обратный ratio: ratio_val = n_BENIGN / n_PortScan
        # Нам нужен PortScan:BENIGN = 1:ratio_val, значит undersample PortScan до 1/ratio_val * n_BENIGN
        ratio_minority = 1.0 / ratio_val

        X_tr, y_tr = undersample(X_train, y_train, ratio_minority, RANDOM_STATE)
        n_min_new = int((y_tr == 1).sum())
        n_maj_new = int((y_tr == 0).sum())
        actual_ratio = f"{n_min_new}:{n_maj_new}"

        print(f"\n  Уровень дисбаланса {ratio_label} "
              f"(PortScan={n_min_new:,}, BENIGN={n_maj_new:,}):")

        # Без балансировки
        metrics_plain = train_and_eval(X_tr, y_tr, X_test, y_test)
        rows.append({
            "Ratio_label":    ratio_label,
            "n_PortScan_train": n_min_new,
            "n_BENIGN_train":   n_maj_new,
            "Mode":           "no_balancing",
            **metrics_plain,
        })
        print(f"    Без балансировки: F1={metrics_plain['F1']:.5f}, "
              f"Recall={metrics_plain['Recall']:.5f}, PR-AUC={metrics_plain['PR_AUC']:.5f}")

        # С scale_pos_weight = n_BENIGN / n_PortScan
        spw = n_maj_new / max(n_min_new, 1)
        metrics_spw = train_and_eval(X_tr, y_tr, X_test, y_test, scale_pos_weight=spw)
        rows.append({
            "Ratio_label":    ratio_label,
            "n_PortScan_train": n_min_new,
            "n_BENIGN_train":   n_maj_new,
            "Mode":           f"scale_pos_weight={spw:.1f}",
            **metrics_spw,
        })
        print(f"    scale_pos_weight={spw:.1f}: F1={metrics_spw['F1']:.5f}, "
              f"Recall={metrics_spw['Recall']:.5f}, PR-AUC={metrics_spw['PR_AUC']:.5f}")

    return pd.DataFrame(rows)


def run_threshold_tuning():
    """
    Threshold tuning XGBoost(v1_default) на DDoS-данных.
    Находим порог, максимизирующий F1 по PR-кривой.
    """
    print("\n[3] Threshold tuning — DDoS XGBoost(v1_default)...")

    # Загружаем y_true и y_proba из кэша
    proc  = joblib.load(DDOS_PROC_PATH)
    y_true = np.array(proc["y_test"])

    with open(DDOS_V1_CACHE / "xgboost_predictions.pkl", "rb") as f:
        cache = pickle.load(f)
    y_proba = np.array(cache["y_proba"])

    # PR-кривая
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)

    # F1 по каждому порогу (избегаем деления на 0)
    denom = precision_arr[:-1] + recall_arr[:-1]
    denom = np.where(denom == 0, 1e-9, denom)
    f1_arr = 2 * precision_arr[:-1] * recall_arr[:-1] / denom

    best_idx = np.argmax(f1_arr)
    best_thr = thresholds[best_idx]
    best_f1  = f1_arr[best_idx]
    best_prec = precision_arr[best_idx]
    best_rec  = recall_arr[best_idx]

    # Дефолтный порог (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    f1_default = f1_score(y_true, y_pred_default)

    print(f"  Дефолтный порог 0.5: F1={f1_default:.6f}")
    print(f"  Оптимальный порог {best_thr:.4f}: "
          f"F1={best_f1:.6f}, Precision={best_prec:.6f}, Recall={best_rec:.6f}")

    # Сохраняем CSV
    thr_df = pd.DataFrame([{
        "Threshold_default":  0.5,
        "F1_default":         round(f1_default, 6),
        "Threshold_optimal":  round(float(best_thr), 6),
        "F1_optimal":         round(float(best_f1), 6),
        "Precision_optimal":  round(float(best_prec), 6),
        "Recall_optimal":     round(float(best_rec), 6),
        "F1_gain":            round(float(best_f1) - float(f1_default), 6),
    }])
    thr_path = OUT_DIR / "threshold_tuning_ddos.csv"
    thr_df.to_csv(thr_path, index=False, encoding="utf-8")
    print(f"  → {thr_path.relative_to(ROOT)}")

    # PR-кривая с отмеченным оптимальным порогом
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall_arr, precision_arr, color="steelblue", lw=1.5,
            label=f"PR-кривая (PR-AUC={average_precision_score(y_true, y_proba):.4f})")
    ax.scatter([best_rec], [best_prec], color="red", zorder=5, s=80,
               label=f"Оптимальный порог {best_thr:.3f}\nF1={best_f1:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR-кривая XGBoost(v1_default) — DDoS-сценарий\n"
                 "с оптимальным порогом по F1", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = OUT_DIR / "pr_curve_ddos.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {pr_path.relative_to(ROOT)}")


def plot_metrics_vs_imbalance(df):
    """График: метрики vs уровень дисбаланса."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_to_plot = ["F1", "Precision", "Recall", "PR_AUC", "ROC_AUC"]
    colors = ["steelblue", "orange", "green", "red", "purple"]

    for ax, mode, title in [
        (axes[0], "no_balancing",       "Без балансировки"),
        (axes[1], None,                 "С scale_pos_weight"),
    ]:
        sub = df[df["Mode"] == mode] if mode else df[df["Mode"].str.startswith("scale")]
        for metric, color in zip(metrics_to_plot, colors):
            ax.plot(sub["Ratio_label"], sub[metric], marker="o",
                    color=color, label=metric)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Дисбаланс (PortScan:BENIGN)")
        ax.set_ylabel("Значение метрики")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)

    plt.suptitle("Влияние дисбаланса классов на качество XGBoost (PortScan-сценарий)",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "metrics_vs_imbalance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.relative_to(ROOT)}")


def main():
    print("=" * 60)
    print("IMBALANCE STUDY")
    print("=" * 60)

    # 1. Imbalance study на PortScan
    df = run_imbalance_study()
    results_path = OUT_DIR / "imbalance_results.csv"
    df.to_csv(results_path, index=False, encoding="utf-8")
    print(f"\n  → {results_path.relative_to(ROOT)}")

    # 2. График
    print("\n[3] График метрик vs дисбаланс...")
    plot_metrics_vs_imbalance(df)

    # 3. Threshold tuning DDoS
    run_threshold_tuning()

    print("\n" + "=" * 60)
    print("ГОТОВО. Артефакты:")
    for fname in ["imbalance_results.csv", "metrics_vs_imbalance.png",
                  "threshold_tuning_ddos.csv", "pr_curve_ddos.png"]:
        print(f"  experiments/imbalance/{fname}")
    print("=" * 60)


if __name__ == "__main__":
    main()
