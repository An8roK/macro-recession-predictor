# Macro Recession Predictor — Phase 2 Global Multi-Region Architecture

## 1. System Overview & $0 GCP Architecture
Decoupled batch-ETL + read-only dashboard hosted in GCP project `an8rok-labs` (`us-west1`), built as a single shared container image (`python:3.14-slim`) to stay strictly within Google Cloud Always Free tiers:
- **Batch Pipeline (`src/pipeline.py`):** Fetches verified macroeconomic and market time series from three official free APIs — **FRED**, **BIS SDMX** (`stats.bis.org`), and **OECD SDMX** (`sdmx.oecd.org`) — across 5 major economies (**US**, **EU**, **JP**, **CN**, **IN**), constructs point-in-time monthly features, runs expanding-window walk-forward ML across 5 horizons, computes the GDP-weighted **GLOBAL** composite, and batch-loads normalized tables to BigQuery. Executed locally via `.venv` or as a weekly **Cloud Run Job** triggered by **Cloud Scheduler**.
- **Data Warehouse (BigQuery):** Dataset `an8rok-labs.macro_recession` in region `us-west1` (co-located with Cloud Run to avoid egress). **CRITICAL COST RULE:** Always write using free batch load jobs (`client.load_table_from_dataframe` with `WRITE_TRUNCATE`) — NEVER use paid streaming inserts (`insert_rows_json`).
- **Web Dashboard (`main.py`):** Stateless Streamlit + Plotly UI running as a **Cloud Run Service**, reading precomputed BigQuery tables (cached via `@st.cache_data(ttl=3600)`). Zero API calls or model training on page load.
- **Multi-Region Schema:** Every BigQuery table includes a `region` column (`'GLOBAL'`, `'US'`, `'EU'`, `'JP'`, `'CN'`, `'IN'`).

---

## 2. Option A Hybrid Target Definitions
Each economy uses the highest-signal, institutionally standard recession/downturn regime definition available from live public feeds:

| Region | Economy | Regime Badge | Target Series | Definition | Settlement Buffer | OOS Start |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| `US` | United States | Classical Recession (NBER) | `FRED:USREC` | Monthly NBER recession indicator (`0`/`1`) | `0M` | `1975-01-31` |
| `EU` | Euro Area (EA20) | Technical Recession (2Q Neg GDP) | `FRED:CLV10MEURB1GQSCAEA20Q` | Quarterly SA Real GDP: `1` for every quarter in a run of ≥2 consecutive negative QoQ quarters, expanded to its 3 calendar months; trailing months after latest GDP quarter stay `NULL` | `+3M` | `2010-01-31` |
| `JP` | Japan | Technical Recession (2Q Neg GDP) | `FRED:JPNRGDPEXP` | Quarterly SA Real GDP: `1` for every quarter in a run of ≥2 consecutive negative QoQ quarters, expanded to its 3 calendar months; trailing months stay `NULL` | `+3M` | `2005-01-31` |
| `CN` | China | Growth-Cycle Downturn (OECD CLI < 99.5) | `OECD:DF_CLI:CHN` | Monthly amplitude-adjusted OECD CLI: `1` if `CLI < 99.5`, else `0` | `0M` | `2008-01-31` |
| `IN` | India | Growth-Cycle Downturn (OECD CLI < 99.5) | `OECD:DF_CLI:IND` | Monthly amplitude-adjusted OECD CLI: `1` if `CLI < 99.5`, else `0` | `0M` | `2005-01-31` |
| `GLOBAL` | Global Composite | GDP-Weighted Global Composite | Derived | GDP-weighted probability across available modeled regions (`US: 0.40, CN: 0.26, EU: 0.22, JP: 0.06, IN: 0.06`), dynamically re-normalized when fewer than 5 regions exist | — | `1975-01-31` |

---

## 3. Multi-Source Data Ingestion (`src/ingest.py` & `src/config.py`)
Direct REST/CSV clients built solely on `requests` (with retry/backoff, timeouts, and secret-safe logging) — no third-party `fredapi` and no DBnomics mirrors:
- **FRED REST API:** `https://api.stlouisfed.org/fred/series/observations` (`FRED_API_KEY` from `.env`).
- **BIS SDMX API v1 (CSV):** `https://stats.bis.org/api/v1/data/{dataflow}/{key}?format=csv` (`WS_CBPOL` policy rates, `WS_LONG_CPI` monthly CPI).
- **OECD SDMX REST API (CSV):** `https://sdmx.oecd.org/public/rest/data/{agency_flow}/{key}?format=csvfilewithlabels` (`DSD_STES@DF_CLI`, `DSD_STES@DF_FINMARK`).

### Verified Series Map by Region
- **US (Phase 1 — intact, `1954-01-01` onward):**
  - Target: `USREC` (0M lag)
  - Market (0M lag): `GS10`, `TB3MS`, `FEDFUNDS`, `BAA`
  - Macro (1M lag): `CPIAUCSL`, `INDPRO`, `UNRATE`, `PAYEMS`
- **EU (Euro Area EA20):**
  - Target: `FRED:CLV10MEURB1GQSCAEA20Q` (Quarterly Real GDP, SA)
  - Market/Policy (0M lag): `OECD:DF_FINMARK:IRLT:EA20` (10Y), `OECD:DF_FINMARK:IR3TIB:EA20` (3M), `BIS:WS_CBPOL/M.XM` (ECB Policy), `OECD:DF_FINMARK:SHARE:EA20` (Broad Equities)
  - Macro/FX (1M lag): `BIS:WS_LONG_CPI/M.XM` (Headline CPI), `OECD:DF_CLI:G4E` (4-Major-Europe CLI), `FRED:RBXMBIS` (REER)
- **JP (Japan):**
  - Target: `FRED:JPNRGDPEXP` (Quarterly Real GDP, SA)
  - Market/Policy (0M lag): `FRED:IRLTLT01JPM156N` (10Y), `BIS:WS_CBPOL/M.JP` (BoJ Policy), `FRED:SPASTT01JPM661N` (Broad Equities)
  - Macro/FX (1M lag): `BIS:WS_LONG_CPI/M.JP` (Headline CPI), `OECD:DF_CLI:JPN` (Japan CLI), `FRED:RBJPBIS` (REER)
- **CN (China):**
  - Target: `OECD:DF_CLI:CHN` (Monthly Amplitude-Adjusted CLI)
  - Market/Policy (0M lag): `FRED:IR3TIB01CNM156N` (3M Short Rate), `BIS:WS_CBPOL/M.CN` (PBoC Policy), `FRED:SPASTT01CNM661N` (Broad Equities)
  - Macro/FX (1M lag): `BIS:WS_LONG_CPI/M.CN` (Headline CPI), `FRED:RBCNBIS` (REER)
- **IN (India):**
  - Target: `OECD:DF_CLI:IND` (Monthly Amplitude-Adjusted CLI)
  - Market/Policy (0M lag): `BIS:WS_CBPOL/M.IN` (RBI Policy), `FRED:SPASTT01INM661N` (Broad Equities)
  - Macro/FX (1M lag): `BIS:WS_LONG_CPI/M.IN` (Headline CPI), `FRED:RBINBIS` (REER)

All monthly series use pandas 3.x-safe `resample("ME").last()` and `.ffill(limit=1)`.

---

## 4. Point-in-Time Feature Engineering (`src/features.py`)
On resampled monthly raw feature series (strictly excluding target series), apply **strictly causal `.ffill(limit=1)`** before computing YoY changes, drawdowns, and rolling windows so a single unpublished print (e.g., the Oct 2025 US shutdown gap in `CPIAUCSL`/`UNRATE` or a 1-month trailing release lag) carries forward the prior published print without cascading NaNs or introducing look-ahead interpolation.
- **Japan Policy Rate Exception (`BIS:WS_CBPOL/M.JP` only):** Because BIS omits `WS_CBPOL/M.JP` during Bank of Japan ZIRP/QE/QQE monetary-base targeting regimes (1999–2016), apply strictly causal unlimited `.ffill()` on `BIS:WS_CBPOL/M.JP` (only after its first valid print) so Japan retains uninterrupted 1995–2026 history while all other series remain strictly bounded at `.ffill(limit=1)`.

### US Feature Matrix (12 Features — unchanged from Phase 1)
1. `term_spread_10y3m`: `GS10 - TB3MS` (0M)
2. `term_spread_3m_ma`: 3M rolling mean of `term_spread_10y3m` (0M)
3. `fedfunds_level`: `FEDFUNDS` (0M)
4. `fedfunds_12m_chg`: `FEDFUNDS - FEDFUNDS.shift(12)` (0M)
5. `credit_spread_baa`: `BAA - GS10` (0M)
6. `credit_spread_6m_chg`: `credit_spread_baa - credit_spread_baa.shift(6)` (0M)
7. `cpi_yoy`: 12M % change of `CPIAUCSL` (1M lag)
8. `cpi_3m_ann`: `((CPI / CPI.shift(3))**4 - 1) * 100` (1M lag)
9. `indpro_yoy`: 12M % change of `INDPRO` (1M lag)
10. `indpro_6m_chg`: 6M % change of `INDPRO` (1M lag)
11. `sahm_rule_signal`: 3M rolling mean of `UNRATE` minus trailing 12M min of that 3M mean (1M lag)
12. `payems_yoy`: 12M % change of `PAYEMS` (1M lag)

### International Feature Matrix (EU & JP: 11 Features; CN & IN: 9 Features)
1. `rate_spread_level`: EU `10Y - 3M`; JP `10Y - Policy`; CN `3M - Policy`; IN `Policy - cpi_yoy` (real policy rate using already-lagged `cpi_yoy`) (0M)
2. `rate_spread_3m_ma`: 3M rolling mean of `rate_spread_level` (0M)
3. `policy_rate_level`: Central bank policy rate (0M)
4. `policy_rate_12m_chg`: 12M level change in policy rate (0M)
5. `cpi_yoy`: 12M % change of headline CPI (1M lag)
6. `cpi_3m_ann`: Annualized 3M % change of headline CPI (1M lag)
7. `equity_12m_ret`: 12M % return of broad equity index (0M)
8. `equity_12m_dd`: % drawdown from trailing 12M peak `(eq / eq.rolling(12).max() - 1) * 100` (0M)
9. `reer_12m_chg`: 12M % change of BIS Real Broad Effective Exchange Rate (1M lag)
10. `cli_level_dev`: `CLI - 100.0` (1M lag — **EU & JP only; NULL and excluded for CN & IN to prevent target tautology**)
11. `cli_6m_chg`: `CLI - CLI.shift(6)` (1M lag — **EU & JP only; NULL and excluded for CN & IN**)

---

## 5. Multi-Region Walk-Forward Modeling & Global Aggregate (`src/modeling.py`)
- **Horizons:** `h in [0, 3, 6, 12, 18]` months. Target at date `t` is `y(t, h) = target_recession(t + h)`.
- **Strict Leakage Embargo:**
  - `US`, `CN`, `IN`: `t_train + h <= t_eval` (and `t_train < t_eval`).
  - `EU`, `JP` (quarterly GDP settlement buffer): `t_train + h + 3 <= t_eval` (and `t_train < t_eval`), guaranteeing both GDP publication settlement and confirmation of 2-consecutive-quarter contractions.
  - Fit `StandardScaler` strictly inside each expanding training window; skip early dates if training window does not yet contain both classes `{0, 1}`.
- **Models (pure `scikit-learn`):**
  1. `logistic_l2` (Primary calibrated model): `Pipeline([StandardScaler(), LogisticRegression(C=c_region, max_iter=1000)])` with per-region L2 regularization (**`C=0.5`** for `US` and tighter **`C=0.05`** for `EU`, `JP`, `CN`, `IN` to prevent small-sample overfitting in early expanding windows), refit every **3 months** (quarterly) on the expanding window for fast Cloud Run execution.
  2. `random_forest` (Nonlinear benchmark): `RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)`, refit every **12 months** on the expanding window.
- **Live Forecast Rows:** For trailing dates where `t + h` is in the future (or after the latest GDP quarter for EU/JP, so `actual_recession` is `NULL`), fit on all labeled rows satisfying `t_train + h + settlement_lag <= latest_labeled_date` and output live probabilities (`is_oos = True`, `actual_recession = NULL`).
- **Driver Attribution:** From the final live `logistic_l2` model per region & horizon, compute exact log-odds contributions `coef_i * z_i` for the latest `as_of_date`.
- **Global Aggregate (`region = 'GLOBAL'`):**
  - Static GDP weights: `{'US': 0.40, 'CN': 0.26, 'EU': 0.22, 'JP': 0.06, 'IN': 0.06}`.
  - For each `(as_of_date, horizon_months, model_name)`, dynamically re-normalize weights across available economies (e.g., pre-2010 before EU OOS begins) and compute `sum(w_norm_r * prob_r)`.
  - Store `GLOBAL` rows in `recession_predictions` (`is_oos = True`, `actual_recession = NULL`) and latest per-economy risk contributions (`weight * regional_prob`) in `feature_contributions`.
- **OOS Metrics (`model_metrics`):** Computed strictly on `is_oos = True` and `actual_recession IS NOT NULL` per modeled region, horizon, and model: `roc_auc`, `pr_auc`, `brier_score`, `base_rate`, `n_obs`, `n_recessions`.

---

## 6. BigQuery Schema (`an8rok-labs.macro_recession`, location `us-west1`)
Auto-create dataset if missing. Overwrite all 5 tables atomically via `WRITE_TRUNCATE` batch load jobs:
1. `raw_macro_series`: `region` (STRING), `series_id` (STRING), `observation_date` (DATE), `value` (FLOAT64), `ingested_at` (TIMESTAMP)
2. `monthly_features`: `region` (STRING), `as_of_date` (DATE), `target_recession` (INT64 nullable), 21 unified nullable FLOAT64 feature columns (12 US + 9 international-specific; `cpi_yoy` & `cpi_3m_ann` shared), `updated_at` (TIMESTAMP)
3. `recession_predictions`: `region` (STRING), `as_of_date` (DATE), `horizon_months` (INT64), `model_name` (STRING), `recession_prob` (FLOAT64), `is_oos` (BOOL), `actual_recession` (INT64 nullable), `updated_at` (TIMESTAMP)
4. `model_metrics`: `region` (STRING), `horizon_months` (INT64), `model_name` (STRING), `eval_start_date` (DATE), `eval_end_date` (DATE), `n_obs` (INT64), `n_recessions` (INT64), `roc_auc` (FLOAT64), `pr_auc` (FLOAT64), `brier_score` (FLOAT64), `base_rate` (FLOAT64), `updated_at` (TIMESTAMP)
5. `feature_contributions`: `region` (STRING), `as_of_date` (DATE), `horizon_months` (INT64), `feature_name` (STRING), `raw_value` (FLOAT64), `standardized_value` (FLOAT64), `coef` (FLOAT64), `log_odds_contrib` (FLOAT64), `updated_at` (TIMESTAMP)

---

## 7. Streamlit Dashboard (`main.py` & `src/charts.py`)
Clean institutional dark/light-compatible Plotly UI (`width="stretch"` everywhere) with sidebar filters (Region selector with `GLOBAL` first + `US`, `EU`, `JP`, `CN`, `IN`; Model selector; Horizon selector), top regime-definition badge, and 4 tabs:
1. **Recession Risk Curve:** Latest `as_of_date` & regime badge banner, 0M–18M KPI cards, interactive term-structure chart, plus a **Global Risk Matrix** table & multi-region term-structure chart when `GLOBAL` is selected.
2. **Historical Timeline:** Out-of-sample predicted probability over time with region-specific recession/downturn shading (or NBER shading + regional overlay toggles for `GLOBAL`).
3. **Signal Drivers:** Per-feature log-odds contributions (`coef_i * z_i`) + latest indicator table for modeled economies, or weighted per-economy risk contributions (`weight * regional_prob`) for `GLOBAL`.
4. **Model Validation:** Per-region OOS metrics + binned reliability curve, or a **Cross-Region OOS Validation Summary** across all 5 modeled economies when `GLOBAL` is selected.
