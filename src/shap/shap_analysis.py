"""
SHAP Analysis for XGBoost (v1_default).
Dataset: CIC-IDS2017

Использует сохранённую модель XGBoost(v1_default) + тестовую выборку из кэша.
TreeExplainer, выборка 5000 наблюдений из теста (полная тестовая избыточна).

Артефакты:
  experiments/shap_analysis/shap_summary_beeswarm.png    — beeswarm топ-20
  experiments/shap_analysis/shap_bar.png                 — mean|SHAP| топ-20 (bar)
  experiments/shap_analysis/shap_dependence_init_win.png — dependence для Init_Win_bytes_forward
  experiments/shap_analysis/gain_vs_shap_ranking.csv     — сопоставление gain vs SHAP

Usage:
    python src/shap/shap_analysis.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_PATH   = ROOT / "experiments/boosting/v1_default/20260222_190338/models/xgboost.joblib"
PROC_PATH    = ROOT / "data/processed/processed_43d973e2.joblib"
OUT_DIR      = ROOT / "experiments/shap_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE  = 5000
RANDOM_STATE = 42
TOP_N        = 20


def main():
    print("=" * 60)
    print("SHAP ANALYSIS — XGBoost(v1_default)")
    print("=" * 60)

    # --- Загрузка модели и данных ---
    print("\n[1] Загрузка модели и тестовой выборки...")
    model = joblib.load(MODEL_PATH)
    cache = joblib.load(PROC_PATH)
    X_test       = np.array(cache["X_test"])
    feature_names = list(cache["feature_names"])
    y_test       = np.array(cache["y_test"])
    print(f"  X_test: {X_test.shape}, признаков: {len(feature_names)}")

    # --- Сабсэмпл ---
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]
    print(f"  Выборка для SHAP: {len(X_sample)} наблюдений "
          f"({y_sample.sum()} атакующих, {(~y_sample.astype(bool)).sum()} BENIGN)")

    # --- SHAP ---
    print("\n[2] Вычисление SHAP values (TreeExplainer)...")
    # Извлекаем XGBClassifier из Pipeline (модель сохранена как Pipeline)
    xgb_clf = model.steps[-1][1] if hasattr(model, "steps") else model
    explainer   = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_sample)
    # Для бинарной XGBoost shap_values — 2D array (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    print(f"  SHAP matrix: {shap_values.shape}")

    # --- Beeswarm plot ---
    print("\n[3] Beeswarm plot (топ-20)...")
    shap_exp = shap.Explanation(
        values=shap_values,
        data=X_sample,
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_exp, max_display=TOP_N, show=False)
    plt.title(f"SHAP Beeswarm — XGBoost(v1_default), n={len(X_sample)}", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "shap_summary_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.relative_to(ROOT)}")

    # --- Bar plot ---
    print("[4] Bar plot mean|SHAP| (топ-20)...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":        feature_names,
        "mean_abs_shap":  mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    top = shap_df.head(TOP_N)
    ax.barh(range(TOP_N), top["mean_abs_shap"].values[::-1], color="steelblue")
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{TOP_N} Features by Mean |SHAP| — XGBoost(v1_default)", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "shap_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.relative_to(ROOT)}")

    # --- Dependence plot: Init_Win_bytes_forward ---
    print("[5] Dependence plot — Init_Win_bytes_forward...")
    feat = "Init_Win_bytes_forward"
    if feat in feature_names:
        feat_idx = feature_names.index(feat)
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            X_sample[:, feat_idx],
            shap_values[:, feat_idx],
            c=y_sample, cmap="coolwarm", alpha=0.4, s=8,
        )
        plt.colorbar(sc, ax=ax, label="Label (0=BENIGN, 1=DDoS)")
        ax.set_xlabel(feat)
        ax.set_ylabel(f"SHAP value для {feat}")
        ax.set_title(f"SHAP Dependence Plot — {feat}", fontsize=12)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        out = OUT_DIR / "shap_dependence_init_win.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {out.relative_to(ROOT)}")
    else:
        print(f"  Признак {feat!r} не найден в feature_names, пропускаем")

    # --- Gain vs SHAP ranking ---
    print("[6] Сопоставление gain vs SHAP ranking...")

    # Gain из модели XGBoost
    xgb_for_gain = model.steps[-1][1] if hasattr(model, "steps") else model
    booster   = xgb_for_gain.get_booster() if hasattr(xgb_for_gain, "get_booster") else xgb_for_gain
    gain_raw  = booster.get_score(importance_type="gain")
    # Ключи booster — могут быть "f0", "f1", ... или имена признаков
    # Если имена — используем напрямую, иначе маппим по индексу
    sample_key = next(iter(gain_raw))
    if sample_key.startswith("f") and sample_key[1:].isdigit():
        gain_by_name = {
            feature_names[int(k[1:])]: v
            for k, v in gain_raw.items()
            if int(k[1:]) < len(feature_names)
        }
    else:
        gain_by_name = dict(gain_raw)

    gain_series = pd.Series(gain_by_name).sort_values(ascending=False)
    gain_total  = gain_series.sum()
    gain_df = pd.DataFrame({
        "feature":    gain_series.index,
        "gain":       gain_series.values,
        "gain_share": gain_series.values / gain_total,
        "gain_rank":  range(1, len(gain_series) + 1),
    })

    # SHAP ranking
    shap_df["shap_rank"] = range(1, len(shap_df) + 1)

    # Объединяем
    merged = pd.merge(shap_df.head(30)[["feature", "mean_abs_shap", "shap_rank"]],
                      gain_df[["feature", "gain_share", "gain_rank"]],
                      on="feature", how="outer").fillna({"mean_abs_shap": 0,
                                                          "shap_rank": 99,
                                                          "gain_share": 0,
                                                          "gain_rank": 99})
    merged = merged.sort_values("shap_rank").reset_index(drop=True)
    merged["rank_diff"] = (merged["shap_rank"] - merged["gain_rank"]).abs().astype(int)

    # Spearman корреляция рангов (топ-20)
    top20 = merged[merged["shap_rank"] <= TOP_N].copy()
    if len(top20) >= 5:
        from scipy.stats import spearmanr
        rho, p_sp = spearmanr(top20["shap_rank"], top20["gain_rank"])
        print(f"  Spearman ρ(shap_rank, gain_rank) топ-20: {rho:.3f}, p={p_sp:.4e}")
        merged.attrs["spearman_rho"] = rho
        merged.attrs["spearman_p"]   = p_sp
    else:
        rho, p_sp = float("nan"), float("nan")

    out = OUT_DIR / "gain_vs_shap_ranking.csv"
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"  → {out.relative_to(ROOT)}")

    # Сохраняем итоговую сводку
    summary_path = OUT_DIR / "shap_summary_stats.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"SHAP Analysis Summary — XGBoost(v1_default)\n")
        f.write(f"Sample size: {len(X_sample)}\n")
        f.write(f"Spearman ρ(shap_rank, gain_rank), top-{TOP_N}: {rho:.4f}, p={p_sp:.4e}\n\n")
        f.write(f"Top-{TOP_N} by mean |SHAP|:\n")
        f.write(shap_df.head(TOP_N).to_string(index=False))
        f.write(f"\n\nTop-{TOP_N} by Gain:\n")
        f.write(gain_df.head(TOP_N).to_string(index=False))

    print(f"  → {summary_path.relative_to(ROOT)}")

    print("\n" + "=" * 60)
    print("ГОТОВО. Артефакты:")
    for fname in ["shap_summary_beeswarm.png", "shap_bar.png",
                  "shap_dependence_init_win.png", "gain_vs_shap_ranking.csv",
                  "shap_summary_stats.txt"]:
        print(f"  experiments/shap_analysis/{fname}")
    print("=" * 60)


if __name__ == "__main__":
    main()
