# 📈 SBER Intraday Alpha — 5-Minute Candle Research Pipeline

> **Quantitative research pipeline for intraday trading of Sberbank (MOEX: SBER) stocks.**  
> Feature engineering on raw 5-minute OHLCV candles → supervised ML model → probability calibration.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research-yellow)]()

---

## 🗂️ Project Structure

```
sber-intraday-alpha/
│
├── notebooks/
│   └── sber_intraday_pipeline.ipynb   # Main research notebook
│
├── data/
│   └── README.md                      # Instructions for placing raw data
│
├── src/
│   └── features.py                    # Feature engineering functions (extracted from notebook)
│
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Standard Python / Jupyter gitignore
└── README.md                          # This file
```

---

## 🎯 Research Goal

Find a **positive-expectancy signal** on 5-minute SBER candles by:

1. Engineering interpretable price-action features from raw candle geometry
2. Adding rolling context features (volatility, volume regime)
3. Adding calendar / session-aware features
4. Adding technical indicators via `pandas_ta`
5. Training a baseline classifier to predict **next-candle direction** (green / red)
6. Evaluating with strict **time-series cross-validation** (no shuffle, chronological split)
7. Calibrating output probabilities with **Platt scaling**

> ⚠️ **This is a research notebook, not a production trading system.** All results must be validated out-of-sample before any capital allocation.

---

## 📦 Data Source

Data is downloaded from [Finam Export](https://www.finam.ru/profile/moex-akcii/sberbank/export/) in the standard format:

```
<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>
SBER;5;20200101;100000;262.50;263.10;262.30;262.90;1540200
...
```

Place the raw CSV file at:
```
./data/Сбербанк/year_result.csv
```

or update `DATA_PATH` in **Cell 5** of the notebook.

---

## 🧱 Feature Groups

| Group | Count | Description |
|---|---|---|
| **Candle geometry** | ~12 | Body, wicks, ratios, direction, doji flag |
| **Returns / momentum** | ~8 | `ret_1`, `ret_3`, `ret_6`, `ret_12`, gap, close-vs-prev-high/low |
| **Rolling context** | ~18 | Mean/std/zscore for range, body, close (windows 6, 12, 24) |
| **Volume regime** | ~9 | `vol_ma_w`, `vol_ratio_w`, `vol_std_w`, `signed_vol` |
| **Calendar / session** | ~14 | Hour, minute, day-of-week, bar-in-day, cyclic sin/cos encodings |
| **Technical indicators** | ~20 | EMA, SMA, RSI, Stoch, MACD, ATR, Bollinger Bands, OBV |
| **Supervised lags** | dynamic | `series_to_supervised(n_in=5)` — 5 bars of lagged features |

---

## 🎓 Target Variable

```python
target_is_green_next = 1  # if CLOSE[t+1] > OPEN[t+1]
target_is_green_next = 0  # otherwise
```

This is a binary classification problem: predict whether the **next 5-minute candle** will close above its open.

---

## 🔀 Train / Validation / Calibration Split

Strictly **chronological**, no shuffling:

```
|────────── Train 70% ──────────|── Valid 15% ──|── Calibration 15% ──|
```

- **Train** — model fitting
- **Validation** — hyperparameter evaluation and metrics reporting
- **Calibration** — Platt scaling (`CalibratedClassifierCV(method='sigmoid', cv='prefit')`)

---

## 📊 Models

| Model | Description |
|---|---|
| `LogisticRegression` | Baseline linear model with `StandardScaler`, `class_weight='balanced'` |
| `GradientBoostingClassifier` | Tree-based ensemble (imported, ready to plug in) |
| `CalibratedClassifierCV` | Probability calibration via Platt scaling on holdout calibration set |

---

## 📐 Metrics

- **Accuracy** — raw directional hit rate
- **ROC AUC** — discriminative ability
- **Log Loss** — quality of probability estimates
- **Brier Score** — mean squared error of probabilities (lower = better)
- **Calibration Curve** — reliability diagram (predicted vs actual frequency)
- **Classification Report** — precision / recall / F1 per class
- **Confusion Matrix** — error breakdown

---

## ⚠️ Known Biases & Risks

| Bias | Mitigation in this notebook |
|---|---|
| **Look-ahead bias** | All features use only `shift(>=1)` or current-bar-close values |
| **Survivorship bias** | Not mitigated — only SBER data used, single surviving issuer |
| **Overfitting** | Chronological split; calibration on separate holdout |
| **Regime shift** | Not explicitly modeled — results are period-specific |
| **Transaction costs** | Not modeled at this stage — pure signal research |
| **Warm-up NaNs** | `min_periods=w` on rolling windows; first rows have NaN features |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/jekamalyshev/sber-intraday-alpha.git
cd sber-intraday-alpha

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place data
mkdir -p "data/Сбербанк"
# copy your year_result.csv to data/Сбербанк/year_result.csv

# 5. Launch notebook
jupyter lab notebooks/sber_intraday_pipeline.ipynb
```

---

## 📋 Requirements

See [requirements.txt](requirements.txt) for the full list. Key packages:

- `pandas >= 2.0`
- `numpy >= 1.24`
- `scikit-learn >= 1.4`
- `pandas_ta >= 0.3.14b`
- `matplotlib >= 3.7`
- `seaborn >= 0.13`
- `jupyter >= 1.0`

---

## 🗺️ Roadmap

- [ ] Walk-forward validation (expanding window)
- [ ] GradientBoostingClassifier vs LogisticRegression comparison
- [ ] Feature importance analysis (permutation importance)
- [ ] Transaction cost simulation
- [ ] Multi-ticker extension (GAZP, LKOH, GMKN)
- [ ] Regime detection (HMM or rolling volatility regime)
- [ ] Meta-labeling for signal filtering

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Evgeniy Malishev** · Moscow, Russia  
[github.com/jekamalyshev](https://github.com/jekamalyshev)
