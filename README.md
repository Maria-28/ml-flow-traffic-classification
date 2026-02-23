# ML-Flow-Traffic-Classification

## 🎯 DDoS Detection in Encrypted Network Traffic using Machine Learning
**Master's Thesis Research Framework**

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-green.svg)](https://catboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 О проекте

Исследовательский фреймворк для магистерской диссертации по обнаружению DDoS-атак в зашифрованном сетевом трафике на основе flow-признаков (без анализа содержимого пакетов).

**Ключевые особенности:**
- ✅ Удаление признаков с временной утечкой (data leakage)
- ✅ Временной сплит (без перемешивания) для честной валидации
- ✅ Полное логирование и кэширование экспериментов
- ✅ Статистическая значимость (McNemar test, bootstrap CI)
- ✅ Воспроизводимость (фиксированные seed'ы)

---

## 📊 Результаты экспериментов

### Baseline модели (DDoS vs BENIGN)

| Модель | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| LogisticRegression | 0.9895 | 0.9954 | 0.9925 | 0.9985 |
| DecisionTree | *в разработке* | | | |
| RandomForest | *в разработке* | | | |

### Бустинг модели (v1_default)

| Модель | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 🏆 **XGBoost** | 0.9998 | 0.9976 | **0.9987** | 0.9999 |
| CatBoost | 0.9922 | 0.9976 | 0.9949 | 0.9999 |
| LogisticRegression | 0.9895 | 0.9954 | 0.9925 | 0.9985 |

### Бустинг модели (v2_tuned)

| Модель | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 🏆 **CatBoost** | 0.9923 | 0.9979 | **0.9951** | 0.9999 |
| XGBoost | 0.9925 | 0.9972 | 0.9948 | 0.9992 |
| LogisticRegression | 0.9905 | 0.9952 | 0.9929 | 0.9986 |

### Статистическая значимость

```
CatBoost vs LogisticRegression: p = 0.0010 (McNemar test)
95% CI for CatBoost F1: [0.9941, 0.9959]
```

### Анализ ошибок (CatBoost)

- **False Negatives**: 24 из 11 366 DDoS (0.21%)
- **False Positives**: 88 из 33 777 BENIGN (0.26%)
- Все FN — SYN-флуд атаки на порту 80

---

## 🏗️ Структура проекта

```
ml-flow-traffic-classification/
├── configs/                    # YAML конфигурации
│   ├── baseline/               # Baseline эксперименты
│   ├── boosting/               # Бустинг эксперименты
│   │   ├── v1_default.yaml
│   │   └── v2_tuned.yaml
│   └── portscan/               # PortScan эксперименты
├── src/                        # Исходный код
│   ├── baseline/               # Baseline модели
│   │   └── baseline_ddos.py
│   ├── boosting/               # Бустинг модели
│   │   └── boosting_ddos.py
│   ├── eda/                    # Разведочный анализ
│   │   └── eda_ddos.py
│   └── utils/                  # Вспомогательные функции
│       ├── data_utils.py
│       ├── eval_utils.py
│       └── plot_utils.py
├── experiments/                # Результаты (не в git)
│   ├── baseline/
│   ├── boosting/
│   └── portscan/
├── data/                       # Данные (не в git)
│   └── raw/
├── requirements.txt            # Зависимости
└── README.md                   # Этот файл
```

---

## 🚀 Быстрый старт

### 1. Клонирование

```bash
git clone https://github.com/Maria-28/ml-flow-traffic-classification.git
cd ml-flow-traffic-classification
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Загрузка данных

Поместите файл `Friday-DDos.pcap_ISCX.csv` в `data/raw/`

### 4. Запуск EDA

```bash
python src/eda/eda_ddos.py
```

### 5. Baseline эксперимент

```bash
python src/baseline/baseline_ddos.py
```

### 6. Бустинг эксперимент

```bash
python src/boosting/boosting_ddos.py --config configs/boosting/v2_tuned.yaml
```

---

## 📦 Зависимости

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
catboost>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
joblib>=1.3.0
tqdm>=4.65.0
scipy>=1.10.0
```

---

## 📄 Лицензия

MIT License — свободное использование с указанием авторства.

---

## 👩‍💻 Автор

**Мария**  
Магистерская диссертация, 2026  
Научный руководитель: [Елагин В.С.]

---

## 📚 Цитирование

Если вы используете этот код в своих исследованиях, пожалуйста, укажите ссылку на репозиторий:

```bibtex
@misc{maria2026ddos,
  author    = {Мария},
  title     = {ML-Flow-Traffic-Classification: DDoS Detection Framework},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Maria-28/ml-flow-traffic-classification}
}
```
