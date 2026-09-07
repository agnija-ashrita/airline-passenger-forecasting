# Airline Passenger Demand Forecasting (Time Series)

Forecasting monthly airline passenger volume from historical totals — the kind of forecast an airline or airport uses for capacity planning, staffing, and revenue management.

## Problem

Accurate short-horizon demand forecasts let an airline plan fleet capacity, staffing, and pricing ahead of time. This project forecasts the next 24 months of passenger volume from 10 years of monthly history, comparing a classical statistical approach against a deep learning approach.

## Dataset

[Airline Passengers](https://github.com/jbrownlee/Datasets) — the classic monthly international airline passenger totals dataset, January 1949 to December 1960 (144 months), in thousands of passengers.

The raw CSV is **not committed to this repo**. The notebook downloads it automatically at runtime (with a fallback mirror if the primary source is unreachable) and caches it locally under `data/`, which is git-ignored. This keeps the repo lightweight and ensures anyone who clones it gets a working pipeline with zero manual data-download steps.

## Approach

1. **EDA** — trend/seasonality visualization, multiplicative seasonal decomposition, ACF/PACF plots to justify model choices.
2. **Time-based train/test split** — the last 24 months (2 full seasonal cycles) are held out; time series are never split randomly, since that would leak future information into training.
3. **Modeling** — two fundamentally different approaches trained and compared:
   - **SARIMA** `(1,1,1)x(1,1,1,12)` on the log-transformed series — a classical statistical model that directly encodes trend and 12-month seasonality.
   - **LSTM** — a windowed (12-month lookback) recurrent neural network, evaluated with honest walk-forward one-step-ahead forecasting (each prediction uses the true preceding months, not the model's own prior predictions).
4. **Evaluation** — RMSE, MAE, and MAPE on the 24 held-out months, plus visual forecast-vs-actual comparison.

## Results

See the notebook's Section 7 (`results_df`) for the full metrics table generated on each run — this file intentionally doesn't hardcode numbers here so the README never drifts out of sync with the code.

## Reproducibility

- Fixed random seeds (`RANDOM_SEED = 42`) for both NumPy and TensorFlow.
- Dependency versions pinned in `requirements.txt`.
- No absolute file paths — all paths are relative to the repo root.
- Data loading is a documented function (`load_airline_data`) with an explicit fallback source, rather than a manual download step.
- The MinMaxScaler used for the LSTM is fit only on the training set, avoiding test-set leakage into the transform.

Re-running the notebook top to bottom reproduces the same split, the same SARIMA fit, and the same LSTM training run.

## How to run

**Option A — Google Colab (recommended, no local setup):**
1. Upload `Airline_Passenger_Forecasting.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Colab has pandas, numpy, statsmodels, scikit-learn, and TensorFlow pre-installed — just run all cells.
3. Run all cells (`Runtime` → `Run all`). Data downloads automatically. Full run takes under 2 minutes (the dataset is small).

**Option B — Local:**
```bash
git clone https://github.com/<your-username>/airline-passenger-forecasting.git
cd airline-passenger-forecasting
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Airline_Passenger_Forecasting.ipynb
```

## Repository structure

```
airline-passenger-forecasting/
├── Airline_Passenger_Forecasting.ipynb   # Full analysis, both models, evaluation
├── requirements.txt                      # Pinned dependencies
└── README.md
```

## Possible extensions

- Grid search over SARIMA orders (`pmdarima.auto_arima`)
- A stacked or bidirectional LSTM
- Exogenous regressors (e.g. holidays) via SARIMAX
- Extending to a multi-series dataset, where deep learning has a clearer data-volume advantage over classical models
