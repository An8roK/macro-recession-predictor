# Macro Recession Predictor — Phase 1 Architecture (US Vertical Slice)

## 1. System Overview & $0 GCP Architecture
Decoupled batch-ETL + read-only dashboard hosted in GCP project `an8rok-labs` (`us-west1`), designed to stay strictly within Google Cloud Always Free tiers:
- **Batch Pipeline (`src/pipeline.py`):** Fetches FRED macro series via REST API, constructs point-in-time monthly features, runs expanding-window walk-forward ML across 5 horizons, and writes normalized tables to BigQuery. Locally executed via `.venv`; deployed later as a weekly **Cloud Run Job** triggered by **Cloud Scheduler**.
- **Data Warehouse (BigQuery):** Dataset `an8rok-labs.macro_recession` in region `us-west1` (co-located with Cloud Run to avoid egress). **CRITICAL COST RULE:** Always write using free batch load jobs (`client.load_table_from_dataframe` with `WRITE_TRUNCATE`) — NEVER use paid streaming inserts (`insert_rows_json`).
- **Web Dashboard (`main.py`):** Stateless Streamlit + Plotly UI reading precomputed BigQuery tables (cached via `@st.cache_data(ttl=3600)`). Zero API calls or model training on page load.
- **Multi-Region Readiness:** Every BigQuery table includes a `region` column (`'US'` in Phase 1) so Phase 2 (Euro Area, Japan, China, India, Global) appends cleanly with zero schema changes.

## 2. Data Ingestion & Point-in-Time Alignment (`src/ingest.py`)
Fetch directly from the FRED REST API (`https://api.stlouisfed.org/fred/series/observations`) using `requests` and `FRED_API_KEY` from `.env` / environment variables (with retry + timeout). Do NOT use third-party `fredapi`.

| Series ID | Description | Monthly Alignment | Publication Lag Shift |
| :--- | :--- | :--- | :--- |
| `USREC` | NBER Recession Indicator (Target, 0/1) | Month-end (`ME`) | 0 months (Target) |
| `GS10` | 10-Year Treasury Constant Maturity Rate | Month-end (`ME`) | 0 months (market) |
| `TB3MS` | 3-Month Treasury Bill Secondary Market Rate | Month-end (`ME`) | 0 months (market) |
| `FEDFUNDS` | Effective Federal Funds Rate | Month-end (`ME`) | 0 months (market) |
| `BAA` | Moody's Seasoned Baa Corporate Bond Yield | Month-end (`ME`) | 0 months (market) |
| `CPIAUCSL` | Consumer Price Index (All Urban Consumers) | Month-end (`ME`) | 1 month (`shift(1)`) |
| `INDPRO` | Industrial Production Index | Month-end (`ME`) | 1 month (`shift(1)`) |
| `UNRATE` | Unemployment Rate | Month-end (`ME`) | 1 month (`shift(1)`) |
| `PAYEMS` | All Employees, Total Nonfarm Payrolls | Month-end (`ME`) | 1 month (`shift(1)`) |

- Use pandas 3.x-safe conventions: `resample("ME").last()` (never `"M"`), `.ffill()` (never `fillna(method="ffill")`).
- History window: `1954-01-01` to latest available observation.

## 3. Feature Engineering (`src/features.py`)
All lagged macro series (`CPIAUCSL`, `INDPRO`, `UNRATE`, `PAYEMS`) are shifted by 1 month **after** computing monthly levels/changes so features at `as_of_date = t` reflect only information genuinely published by month-end `t`:
1. `term_spread_10y3m`: `GS10 - TB3MS`
2. `term_spread_3m_ma`: 3-month rolling mean of `term_spread_10y3m`
3. `fedfunds_level`: `FEDFUNDS`
4. `fedfunds_12m_chg`: `FEDFUNDS - FEDFUNDS.shift(12)`
5. `credit_spread_baa`: `BAA - GS10`
6. `credit_spread_6m_chg`: `credit_spread_baa - credit_spread_baa.shift(6)`
7. `cpi_yoy`: 12-month % change of `CPIAUCSL` (lagged 1M)
8. `cpi_3m_ann`: annualized 3-month % change `((CPI / CPI.shift(3))**4 - 1) * 100` (lagged 1M)
9. `indpro_yoy`: 12-month % change of `INDPRO` (lagged 1M)
10. `indpro_6m_chg`: 6-month % change of `INDPRO` (lagged 1M)
11. `sahm_rule_signal`: 3-month rolling mean of `UNRATE` minus trailing 12-month minimum of that 3-month rolling mean (lagged 1M)
12. `payems_yoy`: 12-month % change of `PAYEMS` (lagged 1M)

Drop initial burn-in rows with NaNs (starting cleanly ~1955–1956). Forward-fill up to 1 month at the trailing edge only if a newly released month has partial market/macro availability, preserving the exact latest `as_of_date`.

## 4. Walk-Forward Multi-Horizon Modeling (`src/modeling.py`)
- **Horizons:** `h in [0, 3, 6, 12, 18]` months. Target at date `t` is `y(t, h) = USREC(t + h)`.
- **Strict Leakage Embargo:** When scoring out-of-sample date `t_eval` for horizon `h`, the expanding training set MUST only include rows `t_train` where `t_train + h <= t_eval` (ensuring no future label overlap). Fit `StandardScaler` strictly inside each training window.
- **OOS Evaluation Start:** `1975-01-31` (provides ~20 years initial training history and ~50 years OOS across 7 US recessions).
- **Models (pure `scikit-learn`, no heavy C++ tree libraries):**
  1. `logistic_l2` (Primary calibrated model): `Pipeline([StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)])`, refit every month on expanding window.
  2. `random_forest` (Nonlinear benchmark): `RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)`, refit every 12 months on expanding window for runtime speed.
- **Live Forecast Rows:** For the most recent dates where `t + h` is in the future (`actual_recession` is `NULL`), fit on all labeled rows where `t_train + h <= latest_labeled_date` and output live probabilities (`is_oos = True`, `actual_recession = NULL`).
- **Driver Attribution:** From the final live `logistic_l2` model per horizon, compute exact feature log-odds contributions `coef_i * z_i` for the latest `as_of_date`.
- **OOS Metrics (`model_metrics`):** Compute strictly on `is_oos = True` and `actual_recession IS NOT NULL`: `roc_auc`, `pr_auc` (average precision), `brier_score`, `base_rate`, `n_obs`, `n_recessions`.

## 5. BigQuery Schema (`an8rok-labs.macro_recession`, location `us-west1`)
Auto-create dataset if missing. Overwrite tables via `WRITE_TRUNCATE` load jobs:
1. `raw_macro_series`: `region` (STRING), `series_id` (STRING), `observation_date` (DATE), `value` (FLOAT64), `ingested_at` (TIMESTAMP)
2. `monthly_features`: `region` (STRING), `as_of_date` (DATE), `target_recession` (INT64), 12 feature FLOAT64 columns, `updated_at` (TIMESTAMP)
3. `recession_predictions`: `region` (STRING), `as_of_date` (DATE), `horizon_months` (INT64), `model_name` (STRING), `recession_prob` (FLOAT64), `is_oos` (BOOL), `actual_recession` (INT64 nullable), `updated_at` (TIMESTAMP)
4. `model_metrics`: `region` (STRING), `horizon_months` (INT64), `model_name` (STRING), `eval_start_date` (DATE), `eval_end_date` (DATE), `n_obs` (INT64), `n_recessions` (INT64), `roc_auc` (FLOAT64), `pr_auc` (FLOAT64), `brier_score` (FLOAT64), `base_rate` (FLOAT64), `updated_at` (TIMESTAMP)
5. `feature_contributions`: `region` (STRING), `as_of_date` (DATE), `horizon_months` (INT64), `feature_name` (STRING), `raw_value` (FLOAT64), `standardized_value` (FLOAT64), `coef` (FLOAT64), `log_odds_contrib` (FLOAT64), `updated_at` (TIMESTAMP)

## 6. Streamlit Dashboard (`main.py` & `src/charts.py`)
Clean institutional dark/light-compatible Plotly UI with sidebar filters (Region selector defaulting to `US`, Model selector `logistic_l2` vs `random_forest`, Horizon selector) and 4 tabs:
1. **Recession Risk Curve:** Latest `as_of_date` banner, KPI cards for 0M/3M/6M/12M/18M probabilities, and an interactive term-structure bar/line chart across all 5 horizons.
2. **Historical Timeline:** Out-of-sample predicted recession probability over time (1975–present) with NBER recession shading (`USREC == 1`) and toggleable horizon/model comparison.
3. **Signal Drivers:** Latest macro indicator readings table + horizontal bar chart of `log_odds_contrib` explaining which signals push recession risk up or down for the selected horizon.
4. **Model Validation:** OOS metrics table (`ROC-AUC`, `PR-AUC`, `Brier Score` vs `Base Rate`) comparing `logistic_l2` vs `random_forest` across all 5 horizons, plus a binned calibration (reliability) curve.

## 7. Repository Layout
```text
macro-recession-predictor/
├── main.py                  # Streamlit dashboard reading BigQuery
├── requirements.txt         # Lean Python 3.14-compatible deps
├── docs/ARCHITECTURE.md     # This specification
├── src/
│   ├── config.py            # Constants, GCP project/dataset, FRED series map
│   ├── ingest.py            # FRED REST API fetcher
│   ├── features.py          # Point-in-time lag shifts & 12 macro features
│   ├── modeling.py          # Expanding-window walk-forward ML + attribution
│   ├── warehouse.py         # BigQuery dataset init, batch load, & read queries
│   ├── pipeline.py          # CLI entry point: ingest -> features -> model -> BQ
│   └── charts.py            # Plotly figure builders for all 4 tabs
└── tests/
    └── verify_pipeline.py   # Fast unit + leakage + BigQuery smoke checks