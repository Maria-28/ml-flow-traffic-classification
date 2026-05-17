"""
Cross-Attack Transfer: DDoS модель на PortScan-данных и наоборот.
Dataset: CIC-IDS2017

Логика:
  - "Лучшая модель работы" = XGBoost(v1_default), обученная на DDoS
  - Для PortScan-стороны берём XGBoost из experiments/portscan/baseline/<latest>/
  - Тестируем каждую модель на СВОЁМ тесте и на ЧУЖОМ
  - Матрица переноса 2×2: строки = модель, столбцы = тест

ВАЖНО: оба препроцессора должны вернуть одинаковый список признаков.
Скрипт явно проверяет пересечение и сообщает о расхождениях.

Артефакты:
  experiments/cross_attack/transfer_matrix.csv
  experiments/cross_attack/cm_ddos_on_portscan.png
  experiments/cross_attack/cm_portscan_on_ddos.png
  experiments/cross_attack/transfer_report.txt

Usage:
    python src/cross_attack/run_transfer.py

    # Или явно указать путь к PortScan-эксперименту:
    python src/cross_attack/run_transfer.py --portscan-dir experiments/portscan/baseline/<timestamp>
"""

import argparse
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
    roc_auc_score, average_precision_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DDOS_MODEL_PATH  = ROOT / "experiments/boosting/v1_default/20260222_190338/models/xgboost.joblib"
DDOS_PROC_PATH   = ROOT / "data/processed/processed_43d973e2.joblib"
OUT_DIR          = ROOT / "experiments/cross_attack"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_portscan_dir(explicit=None):
    """Находит директорию последнего PortScan-эксперимента."""
    if explicit:
        p = ROOT / explicit
        if not p.exists():
            raise FileNotFoundError(f"Указанная директория не найдена: {p}")
        return p
    base = ROOT / "experiments/portscan/baseline"
    if not base.exists():
        raise FileNotFoundError(
            f"Директория {base} не найдена.\n"
            f"Сначала запустите:\n"
            f"  python src/portscan/boosting_portscan.py --config configs/portscan/portscan_baseline.yaml"
        )
    subdirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(
            f"В {base} нет поддиректорий с результатами.\n"
            f"Запустите PortScan-эксперимент и повторите."
        )
    chosen = subdirs[-1]   # берём последний (по имени = по времени)
    print(f"  Используем PortScan-директорию: {chosen.relative_to(ROOT)}")
    return chosen


def load_portscan_model_and_data(ps_dir):
    """Загружает XGBoost-модель и тестовые данные PortScan-эксперимента."""
    model_path = ps_dir / "models" / "xgboost.joblib"
    proc_path  = ps_dir / "data_cache.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost-модель не найдена: {model_path}")

    ps_model = joblib.load(model_path)

    # Тестовые данные PortScan могут лежать в разных местах в зависимости от запуска.
    # Ищем data_cache или любой joblib в директории:
    cache_candidates = list(ps_dir.glob("*.joblib")) + \
                       list((ps_dir / "cache").glob("*.joblib")) if (ps_dir / "cache").exists() else \
                       list(ps_dir.glob("*.joblib"))

    # Fallback: перестраиваем тестовую выборку из raw-данных через data_utils
    # (это надёжнее, чем искать кэш)
    print("  Загрузка PortScan-данных через data_utils (temporal split 80/20)...")
    from src.utils.data_utils import load_and_clean_data, build_feature_matrix, temporal_split
    df_ps = load_and_clean_data(
        ROOT / "data/raw/Friday-PortScan.pcap_ISCX.csv",
        expected_labels=["BENIGN", "PortScan"],
        verbose=False,
    )
    X_ps, y_ps, feat_ps, _ = build_feature_matrix(df_ps, verbose=False)
    X_train_ps, X_test_ps, y_train_ps, y_test_ps = temporal_split(X_ps, y_ps, verbose=False)
    return ps_model, X_test_ps, y_test_ps, feat_ps


def align_features(X, src_features, tgt_features):
    """
    Приводит матрицу X (с признаками src_features) к порядку tgt_features.
    Отсутствующие признаки заполняются нулями, лишние — отбрасываются.
    """
    src_set = set(src_features)
    tgt_set = set(tgt_features)
    missing = tgt_set - src_set
    extra   = src_set - tgt_set

    if missing:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: {len(missing)} признаков отсутствуют в источнике "
              f"(будут заполнены 0): {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}")
    if extra:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: {len(extra)} лишних признаков в источнике "
              f"(будут отброшены): {sorted(extra)[:5]}{'...' if len(extra)>5 else ''}")

    src_df = pd.DataFrame(X, columns=src_features)
    for feat in missing:
        src_df[feat] = 0.0
    return src_df[tgt_features].values


def compute_metrics(y_true, y_pred, y_proba, label):
    """Считает основные метрики для одной пары (модель, тест)."""
    return {
        "Scenario":    label,
        "Precision":   round(precision_score(y_true, y_pred, zero_division=0), 6),
        "Recall":      round(recall_score(y_true, y_pred, zero_division=0), 6),
        "F1":          round(f1_score(y_true, y_pred, zero_division=0), 6),
        "ROC_AUC":     round(roc_auc_score(y_true, y_proba), 6),
        "PR_AUC":      round(average_precision_score(y_true, y_proba), 6),
    }


def plot_cm(y_true, y_pred, title, out_path, labels=("BENIGN", "ATTACK")):
    """Рисует матрицу ошибок."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_proba(model, X):
    """Получает вероятности класса 1."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portscan-dir", default=None,
                        help="Путь к директории PortScan-эксперимента")
    args = parser.parse_args()

    print("=" * 60)
    print("CROSS-ATTACK TRANSFER")
    print("=" * 60)

    # --- Загрузка DDoS-данных ---
    print("\n[1] Загрузка DDoS-модели и тестовых данных...")
    ddos_model = joblib.load(DDOS_MODEL_PATH)
    ddos_cache = joblib.load(DDOS_PROC_PATH)
    X_test_ddos  = np.array(ddos_cache["X_test"])
    y_test_ddos  = np.array(ddos_cache["y_test"])
    feat_ddos    = list(ddos_cache["feature_names"])
    print(f"  DDoS тест: {X_test_ddos.shape}, атакующих: {int(y_test_ddos.sum())}")

    # --- Загрузка PortScan-данных ---
    print("\n[2] Загрузка PortScan-модели и тестовых данных...")
    try:
        ps_dir = find_portscan_dir(args.portscan_dir)
        ps_model, X_test_ps, y_test_ps, feat_ps = load_portscan_model_and_data(ps_dir)
    except FileNotFoundError as e:
        print(f"\nОШИБКА: {e}")
        sys.exit(1)
    print(f"  PortScan тест: {X_test_ps.shape}, атакующих: {int(y_test_ps.sum())}")

    # --- Проверка совпадения признаков ---
    print("\n[3] Проверка признакового пространства...")
    common = sorted(set(feat_ddos) & set(feat_ps))
    print(f"  Общих признаков: {len(common)} / DDoS: {len(feat_ddos)} / PortScan: {len(feat_ps)}")
    if len(common) == len(feat_ddos) == len(feat_ps):
        print("  Признаки совпадают полностью — выравнивание не требуется.")
    else:
        print("  Признаки различаются — применяем выравнивание по пересечению.")

    # --- Четыре сценария ---
    print("\n[4] Тестирование четырёх сценариев...")
    results = []

    # Выравниваем данные под признаки каждой модели
    X_ddos_for_ddos = X_test_ddos                                   # нативные данные
    X_ps_for_ddos   = align_features(X_test_ps,  feat_ps,  feat_ddos)  # PS-тест → DDoS-признаки
    X_ddos_for_ps   = align_features(X_test_ddos, feat_ddos, feat_ps)  # DDoS-тест → PS-признаки
    X_ps_for_ps     = X_test_ps                                      # нативные данные

    scenarios = [
        ("DDoS model → DDoS test",    ddos_model, X_ddos_for_ddos, y_test_ddos, "DDoS"),
        ("DDoS model → PortScan test",ddos_model, X_ps_for_ddos,   y_test_ps,   "PortScan"),
        ("PortScan model → DDoS test",ps_model,   X_ddos_for_ps,   y_test_ddos, "DDoS"),
        ("PortScan model → PortScan test",ps_model,X_ps_for_ps,    y_test_ps,   "PortScan"),
    ]

    for label, model, X, y_true, test_type in scenarios:
        y_pred  = model.predict(X)
        y_proba = get_proba(model, X)
        metrics = compute_metrics(y_true, y_pred, y_proba, label)
        results.append(metrics)
        print(f"  {label}: F1={metrics['F1']:.5f}, "
              f"ROC-AUC={metrics['ROC_AUC']:.5f}")

        # Матрицы ошибок для двух cross-сценариев
        if label == "DDoS model → PortScan test":
            plot_cm(y_true, y_pred,
                    "DDoS Model on PortScan Test",
                    OUT_DIR / "cm_ddos_on_portscan.png",
                    labels=("BENIGN", "PortScan"))
        elif label == "PortScan model → DDoS test":
            plot_cm(y_true, y_pred,
                    "PortScan Model on DDoS Test",
                    OUT_DIR / "cm_portscan_on_ddos.png",
                    labels=("BENIGN", "DDoS"))

    # --- Сохранение ---
    df = pd.DataFrame(results)
    matrix_path = OUT_DIR / "transfer_matrix.csv"
    df.to_csv(matrix_path, index=False, encoding="utf-8")
    print(f"\n  → {matrix_path.relative_to(ROOT)}")

    # Текстовый отчёт
    report_lines = [
        "Cross-Attack Transfer Results",
        "=" * 60,
        "",
        df.to_string(index=False),
        "",
        "Интерпретация:",
        "  Строки 1,4 — in-domain (модель на своих данных)",
        "  Строки 2,3 — cross-domain (перенос на чужие данные)",
        "  Снижение F1 в строках 2,3 vs 1,4 = деградация при переносе",
    ]
    report_path = OUT_DIR / "transfer_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  → {report_path.relative_to(ROOT)}")

    print("\n" + "=" * 60)
    print("ГОТОВО.")
    print("=" * 60)


if __name__ == "__main__":
    main()
