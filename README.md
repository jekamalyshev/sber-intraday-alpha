# 📈 SBER Intraday Alpha — 5-Minute ML Research Pipeline

> **Research pipeline для поиска статистических закономерностей в 5-минутных данных акций Сбербанка (MOEX: SBER).**  
> Цель: разработать торговую стратегию с положительным математическим ожиданием после учёта комиссий и slippage.

---

## Архитектура

```
sber-intraday-alpha/
├── notebooks/
│   └── sber_intraday_pipeline.ipynb   # Основной research-ноутбук (11 ячеек)
├── src/
│   └── models.py                       # Model factory, calibration, evaluate_model
├── data/
│   └── Сбербанк/
│       └── year_result.csv             # Сырые 5m-котировки Finam (не включены в репо)
├── requirements.txt
└── README.md
```

---

## Пайплайн (ноутбук)

| Ячейка | Что делает |
|--------|------------|
| **1** | Импорты, совместимость sklearn, версии зависимостей |
| **2** | `series_to_supervised` — преобразование ряда в supervised dataset |
| **3** | Feature engineering: геометрия свечи, returns, rolling context, calendar, TA-индикаторы |
| **4** | Master pipeline builder, leakage guard, `build_X_y_for_model` |
| **5** | Загрузка данных Finam, запуск полного пайплайна |
| **6** | EDA: распределения признаков, корреляционная матрица |
| **7** | Хронологический split Train 70% / Valid 15% / Calib 15% |
| **8** | **XGBoost classifier** — обучение с early stopping на Valid |
| **9** | Platt scaling (CalibratedClassifierCV, cv='prefit') на Calib |
| **10** | Сводная таблица метрик: XGB vs XGB+Calib |
| **11** | **Feature Importance**: built-in Gain + Permutation Importance (Valid AUC) |

---

## Признаки

### Геометрия свечи
- `candle_body`, `body_abs`, `candle_range`, `upper_wick`, `lower_wick`
- `body_to_range`, `upper_wick_to_range`, `lower_wick_to_range`
- `close_pos_in_range`, `open_pos_in_range`
- `direction`, `is_green`, `is_red`, `is_doji_like`

### Returns & momentum
- `ret_1`, `ret_6`, `ret_12`, `ret_24` — pct_change за 5m / 30m / 60m / 120m
- `gap_from_prev_close`, `close_to_prev_high`, `close_to_prev_low`

### Rolling context (windows=6, 12, 24)
- `range_mean_W`, `range_std_W`, `range_zscore_W`
- `close_ma_W`, `close_vs_ma_W`, `close_zscore_W`
- `vol_ma_W`, `vol_ratio_W`

### Calendar / session
- `hour`, `dayofweek`, `hour_sin/cos`, `dow_sin/cos`
- `minutes_from_session_open`, `is_opening_30m`, `is_first_hour`, `bar_in_day`

### Технические индикаторы (pandas_ta)
- EMA 10/20, SMA 20, RSI 14, ATR 14, NATR 14
- MACD (12/26/9), Bollinger Bands (20), Stochastic, OBV
- `price_vs_ema_10/20`, `close_pos_in_bbands`

### Price × Volume
- `body_x_vol`, `signed_vol`, `money_flow_proxy`

---

## Модель

### XGBoost (Cell 8)

```python
XGBClassifier(
    n_estimators=500,
    max_depth=4,              # неглубокие деревья → меньше overfit
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=20,      # минимум сэмплов в листе
    reg_alpha=0.1,            # L1
    reg_lambda=1.0,           # L2
    scale_pos_weight=neg/pos, # компенсация дисбаланса классов
    early_stopping_rounds=30, # останавливаемся при деградации LogLoss на Valid
)
```

> **StandardScaler не нужен** — деревья инвариантны к масштабу признаков.

### Calibration (Cell 9)

Platt scaling через `CalibratedClassifierCV(cv='prefit')` на отдельном calib-сегменте.  
Калибровка улучшает Log Loss и Brier Score, не меняя AUC.

---

## Метрики оценки

| Метрика | Почему важна |
|---------|-------------|
| **AUC-ROC** | Дискриминирующая способность, инвариантна к порогу |
| **Log Loss** | Качество вероятностей — ключевая метрика для calibration |
| **Brier Score** | Вероятностная точность, интерпретируется как MSE по вероятностям |
| **Accuracy** | Ориентир, но не основная метрика при дисбалансе классов |

> **Baseline:** случайная модель при 50/50 классах → AUC=0.50, LogLoss=ln(2)≈0.693, Brier=0.25

---

## Feature Importance (Cell 11)

**Два типа важности признаков:**

- **Gain (built-in)** — средний прирост качества на сплитах по признаку во всех деревьях
- **Permutation Importance (Valid)** — насколько снижается AUC при случайном перемешивании признака на `X_valid`

Permutation importance честнее для оценки реальной предсказательной силы, так как не зависит от структуры дерева.  
Признаки с importance ≤ 0 — кандидаты на удаление из feature set.

---

## Быстрый старт

```bash
git clone https://github.com/jekamalyshev/sber-intraday-alpha.git
cd sber-intraday-alpha
pip install -r requirements.txt
```

1. Скачать 5-минутные данные SBER с [Finam Export](https://www.finam.ru/profile/moex-akcii/sberbank/export/)
2. Положить CSV в `data/Сбербанк/year_result.csv`
3. Открыть `notebooks/sber_intraday_pipeline.ipynb` и запустить все ячейки

---

## Методологические гарантии

- ✅ **No look-ahead bias** — все признаки используют только данные текущего и прошлых баров
- ✅ **Хронологический split** — Train / Valid / Calib идут строго по времени, без перемешивания
- ✅ **Early stopping только на Valid** — веса модели обновляются только по Train
- ✅ **Калибровка на отдельном Calib** — sigmoid обучается на данных, не виденных при обучении XGB
- ✅ **min_periods=w** в rolling — защита от частичных окон на разогреве
- ⚠️ **Survivorship bias** — в данных только один тикер (SBER), выжившая компания
- ⚠️ **Статичный split** — следующий шаг: walk-forward validation

---

## Следующие шаги

- [ ] Walk-forward validation (expanding window)
- [ ] Убрать признаки с permutation importance ≤ 0 → переобучить
- [ ] Hyperparameter tuning (Optuna)
- [ ] Симуляция транзакционных издержек (комиссия MOEX + slippage)
- [ ] Regime detection (HMM / rolling volatility)
- [ ] Meta-labeling для фильтрации сигналов с низким confidence

---

## Зависимости

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.4
xgboost >= 2.0
pandas_ta >= 0.3.14b
matplotlib >= 3.7
seaborn >= 0.13
```

---

## Disclaimer

Данный репозиторий является **исследовательским проектом**. Результаты бэктеста не гарантируют доходности в реальной торговле. Все стратегии требуют дополнительной валидации, учёта транзакционных издержек и тестирования на out-of-sample данных.
