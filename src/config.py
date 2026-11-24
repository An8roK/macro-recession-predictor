"""Project constants, GCP/BigQuery configuration, and FRED series definitions."""

from __future__ import annotations

import os
from dataclasses import dataclass

# GCP & BigQuery settings (co-located in us-west1 to stay within Always Free tier)
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "an8rok-labs")
BQ_DATASET: str = os.getenv("BQ_DATASET", "macro_recession")
BQ_LOCATION: str = os.getenv("BQ_LOCATION", "us-west1")

# Multi-region identifier (Phase 1 = US vertical slice)
DEFAULT_REGION: str = "US"

# FRED REST API settings
FRED_API_BASE_URL: str = "https://api.stlouisfed.org/fred/series/observations"
HISTORY_START_DATE: str = "1954-01-01"
OOS_START_DATE: str = "1975-01-31"

# Forecast horizons (months ahead)
HORIZONS: tuple[int, ...] = (0, 3, 6, 12, 18)

# Model identifiers
MODEL_LOGISTIC: str = "logistic_l2"
MODEL_RF: str = "random_forest"
MODEL_NAMES: tuple[str, ...] = (MODEL_LOGISTIC, MODEL_RF)

# BigQuery table names
TABLE_RAW_SERIES: str = "raw_macro_series"
TABLE_MONTHLY_FEATURES: str = "monthly_features"
TABLE_PREDICTIONS: str = "recession_predictions"
TABLE_METRICS: str = "model_metrics"
TABLE_CONTRIBUTIONS: str = "feature_contributions"

ALL_TABLES: tuple[str, ...] = (
    TABLE_RAW_SERIES,
    TABLE_MONTHLY_FEATURES,
    TABLE_PREDICTIONS,
    TABLE_METRICS,
    TABLE_CONTRIBUTIONS,
)


@dataclass(frozen=True)
class SeriesSpec:
    """Metadata for a single FRED macroeconomic series."""

    series_id: str
    description: str
    lag_months: int
    is_target: bool = False


FRED_SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("USREC", "NBER Recession Indicator (Target, 0/1)", lag_months=0, is_target=True),
    SeriesSpec("GS10", "10-Year Treasury Constant Maturity Rate", lag_months=0),
    SeriesSpec("TB3MS", "3-Month Treasury Bill Secondary Market Rate", lag_months=0),
    SeriesSpec("FEDFUNDS", "Effective Federal Funds Rate", lag_months=0),
    SeriesSpec("BAA", "Moody's Seasoned Baa Corporate Bond Yield", lag_months=0),
    SeriesSpec("CPIAUCSL", "Consumer Price Index (All Urban Consumers)", lag_months=1),
    SeriesSpec("INDPRO", "Industrial Production Index", lag_months=1),
    SeriesSpec("UNRATE", "Unemployment Rate", lag_months=1),
    SeriesSpec("PAYEMS", "All Employees, Total Nonfarm Payrolls", lag_months=1),
)

FRED_SERIES_IDS: tuple[str, ...] = tuple(spec.series_id for spec in FRED_SERIES)

# Exact ordered list of the 12 model features defined in docs/ARCHITECTURE.md
FEATURE_COLUMNS: tuple[str, ...] = (
    "term_spread_10y3m",
    "term_spread_3m_ma",
    "fedfunds_level",
    "fedfunds_12m_chg",
    "credit_spread_baa",
    "credit_spread_6m_chg",
    "cpi_yoy",
    "cpi_3m_ann",
    "indpro_yoy",
    "indpro_6m_chg",
    "sahm_rule_signal",
    "payems_yoy",
)

FEATURE_LABELS: dict[str, dict[str, str]] = {
    "term_spread_10y3m": {"label": "10Y–3M Term Spread", "unit": "pp"},
    "term_spread_3m_ma": {"label": "10Y–3M Term Spread (3M MA)", "unit": "pp"},
    "fedfunds_level": {"label": "Fed Funds Rate", "unit": "%"},
    "fedfunds_12m_chg": {"label": "Fed Funds 12M Change", "unit": "pp"},
    "credit_spread_baa": {"label": "Baa–10Y Credit Spread", "unit": "pp"},
    "credit_spread_6m_chg": {"label": "Baa Credit Spread 6M Change", "unit": "pp"},
    "cpi_yoy": {"label": "CPI Inflation (YoY, 1M lag)", "unit": "%"},
    "cpi_3m_ann": {"label": "CPI Inflation (3M Ann., 1M lag)", "unit": "%"},
    "indpro_yoy": {"label": "Industrial Production (YoY, 1M lag)", "unit": "%"},
    "indpro_6m_chg": {"label": "Industrial Production (6M %, 1M lag)", "unit": "%"},
    "sahm_rule_signal": {"label": "Sahm Rule Signal (1M lag)", "unit": "pp"},
    "payems_yoy": {"label": "Nonfarm Payrolls (YoY, 1M lag)", "unit": "%"},
}
