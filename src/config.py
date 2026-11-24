"""Project constants, GCP/BigQuery configuration, and Phase 2 multi-region definitions."""

from __future__ import annotations

import os
from dataclasses import dataclass

# GCP & BigQuery settings (co-located in us-west1 to stay within Always Free tier)
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "an8rok-labs")
BQ_DATASET: str = os.getenv("BQ_DATASET", "macro_recession")
BQ_LOCATION: str = os.getenv("BQ_LOCATION", "us-west1")

# Multi-region identifiers
DEFAULT_REGION: str = "US"
GLOBAL_REGION: str = "GLOBAL"
MODEL_REGIONS: tuple[str, ...] = ("US", "EU", "JP", "CN", "IN")
ALL_REGIONS: tuple[str, ...] = ("GLOBAL", "US", "EU", "JP", "CN", "IN")

# Static GDP weights for the GLOBAL composite (dynamically re-normalized when < 5 regions exist)
GLOBAL_WEIGHTS: dict[str, float] = {
    "US": 0.40,
    "CN": 0.26,
    "EU": 0.22,
    "JP": 0.06,
    "IN": 0.06,
}

# External API endpoints (zero DBnomics; official FRED, BIS SDMX, OECD SDMX only)
FRED_API_BASE_URL: str = "https://api.stlouisfed.org/fred/series/observations"
BIS_SDMX_BASE_URL: str = "https://stats.bis.org/api/v1/data"
OECD_SDMX_BASE_URL: str = "https://sdmx.oecd.org/public/rest/data"

HISTORY_START_DATE: str = "1954-01-01"
INTL_HISTORY_START_DATE: str = "1980-01-01"
OOS_START_DATE: str = "1975-01-31"
CLI_DOWNTURN_THRESHOLD: float = 99.5

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
    """Metadata for a single macroeconomic series."""

    series_id: str
    description: str
    lag_months: int
    is_target: bool = False


# Phase 1 US FRED catalog (kept completely intact)
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

# Phase 1 US feature columns (exact 12 columns; alias FEATURE_COLUMNS kept for compatibility)
US_FEATURE_COLUMNS: tuple[str, ...] = (
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
FEATURE_COLUMNS: tuple[str, ...] = US_FEATURE_COLUMNS

# Phase 2 international-specific feature columns (cpi_yoy and cpi_3m_ann are shared with US)
INTL_ONLY_FEATURE_COLUMNS: tuple[str, ...] = (
    "rate_spread_level",
    "rate_spread_3m_ma",
    "policy_rate_level",
    "policy_rate_12m_chg",
    "equity_12m_ret",
    "equity_12m_dd",
    "reer_12m_chg",
    "cli_level_dev",
    "cli_6m_chg",
)

# Unified 21 nullable feature columns stored in BigQuery monthly_features
ALL_FEATURE_COLUMNS: tuple[str, ...] = US_FEATURE_COLUMNS + INTL_ONLY_FEATURE_COLUMNS

# Active feature sets per international group
INTL_WITH_CLI_FEATURES: tuple[str, ...] = (
    "rate_spread_level",
    "rate_spread_3m_ma",
    "policy_rate_level",
    "policy_rate_12m_chg",
    "cpi_yoy",
    "cpi_3m_ann",
    "equity_12m_ret",
    "equity_12m_dd",
    "reer_12m_chg",
    "cli_level_dev",
    "cli_6m_chg",
)

# CLI strictly excluded for CN & IN to prevent target tautology
INTL_NO_CLI_FEATURES: tuple[str, ...] = (
    "rate_spread_level",
    "rate_spread_3m_ma",
    "policy_rate_level",
    "policy_rate_12m_chg",
    "cpi_yoy",
    "cpi_3m_ann",
    "equity_12m_ret",
    "equity_12m_dd",
    "reer_12m_chg",
)


@dataclass(frozen=True)
class RegionSpec:
    """Declarative specification for a modeled economy or composite region."""

    code: str
    name: str
    regime_badge: str
    target_type: str
    oos_start_date: str
    settlement_lag_months: int
    feature_columns: tuple[str, ...]
    series_roles: dict[str, str]
    spread_label: str = "Rate Spread"
    logistic_c: float = 0.5


REGION_SPECS: dict[str, RegionSpec] = {
    "GLOBAL": RegionSpec(
        code="GLOBAL",
        name="Global (GDP-Weighted)",
        regime_badge="GDP-Weighted Global Composite",
        target_type="global_composite",
        oos_start_date="1975-01-31",
        settlement_lag_months=0,
        feature_columns=(),
        series_roles={},
    ),
    "US": RegionSpec(
        code="US",
        name="United States",
        regime_badge="Classical Recession (NBER)",
        target_type="nber",
        oos_start_date="1975-01-31",
        settlement_lag_months=0,
        feature_columns=US_FEATURE_COLUMNS,
        series_roles={
            "target": "USREC",
            "yield_10y": "GS10",
            "rate_3m": "TB3MS",
            "policy_rate": "FEDFUNDS",
            "baa": "BAA",
            "cpi": "CPIAUCSL",
            "indpro": "INDPRO",
            "unrate": "UNRATE",
            "payems": "PAYEMS",
        },
    ),
    "EU": RegionSpec(
        code="EU",
        name="Euro Area (EA20)",
        regime_badge="Technical Recession (2Q Neg GDP)",
        target_type="gdp_2q",
        oos_start_date="2010-01-31",
        settlement_lag_months=3,
        feature_columns=INTL_WITH_CLI_FEATURES,
        series_roles={
            "target": "CLV10MEURB1GQSCAEA20Q",
            "yield_10y": "OECD:DF_FINMARK:IRLT:EA20",
            "rate_3m": "OECD:DF_FINMARK:IR3TIB:EA20",
            "policy_rate": "BIS:WS_CBPOL/M.XM",
            "cpi": "BIS:WS_LONG_CPI/M.XM",
            "cli": "OECD:DF_CLI:G4E",
            "equity": "OECD:DF_FINMARK:SHARE:EA20",
            "reer": "RBXMBIS",
        },
        spread_label="10Y–3M Rate Spread",
        logistic_c=0.05,
    ),
    "JP": RegionSpec(
        code="JP",
        name="Japan",
        regime_badge="Technical Recession (2Q Neg GDP)",
        target_type="gdp_2q",
        oos_start_date="2005-01-31",
        settlement_lag_months=3,
        feature_columns=INTL_WITH_CLI_FEATURES,
        series_roles={
            "target": "JPNRGDPEXP",
            "yield_10y": "IRLTLT01JPM156N",
            "policy_rate": "BIS:WS_CBPOL/M.JP",
            "cpi": "BIS:WS_LONG_CPI/M.JP",
            "cli": "OECD:DF_CLI:JPN",
            "equity": "SPASTT01JPM661N",
            "reer": "RBJPBIS",
        },
        spread_label="10Y–Policy Rate Spread",
        logistic_c=0.05,
    ),
    "CN": RegionSpec(
        code="CN",
        name="China",
        regime_badge="Growth-Cycle Downturn (OECD CLI < 99.5)",
        target_type="cli_995",
        oos_start_date="2008-01-31",
        settlement_lag_months=0,
        feature_columns=INTL_NO_CLI_FEATURES,
        series_roles={
            "target": "OECD:DF_CLI:CHN",
            "rate_3m": "IR3TIB01CNM156N",
            "policy_rate": "BIS:WS_CBPOL/M.CN",
            "cpi": "BIS:WS_LONG_CPI/M.CN",
            "equity": "SPASTT01CNM661N",
            "reer": "RBCNBIS",
        },
        spread_label="3M–Policy Rate Spread",
        logistic_c=0.05,
    ),
    "IN": RegionSpec(
        code="IN",
        name="India",
        regime_badge="Growth-Cycle Downturn (OECD CLI < 99.5)",
        target_type="cli_995",
        oos_start_date="2005-01-31",
        settlement_lag_months=0,
        feature_columns=INTL_NO_CLI_FEATURES,
        series_roles={
            "target": "OECD:DF_CLI:IND",
            "policy_rate": "BIS:WS_CBPOL/M.IN",
            "cpi": "BIS:WS_LONG_CPI/M.IN",
            "equity": "SPASTT01INM661N",
            "reer": "RBINBIS",
        },
        spread_label="Real Policy Rate (Policy − CPI YoY)",
        logistic_c=0.05,
    ),
}

FEATURE_LABELS: dict[str, dict[str, str]] = {
    # Phase 1 US features
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
    # Phase 2 international features
    "rate_spread_level": {"label": "Rate Spread", "unit": "pp"},
    "rate_spread_3m_ma": {"label": "Rate Spread (3M MA)", "unit": "pp"},
    "policy_rate_level": {"label": "Central Bank Policy Rate", "unit": "%"},
    "policy_rate_12m_chg": {"label": "Policy Rate 12M Change", "unit": "pp"},
    "equity_12m_ret": {"label": "Broad Equities (12M Return)", "unit": "%"},
    "equity_12m_dd": {"label": "Broad Equities (12M Drawdown)", "unit": "%"},
    "reer_12m_chg": {"label": "Real Effective FX (12M %, 1M lag)", "unit": "%"},
    "cli_level_dev": {"label": "OECD CLI Deviation (1M lag)", "unit": "pts"},
    "cli_6m_chg": {"label": "OECD CLI 6M Change (1M lag)", "unit": "pts"},
}


def get_feature_label(feature_name: str, region: str = DEFAULT_REGION) -> str:
    """Return a clean, region-aware display label for a feature or GLOBAL constituent."""
    if region == GLOBAL_REGION and feature_name in REGION_SPECS:
        return f"{REGION_SPECS[feature_name].name} ({feature_name})"
    spec = REGION_SPECS.get(region)
    if spec is not None:
        if feature_name == "rate_spread_level":
            return spec.spread_label
        if feature_name == "rate_spread_3m_ma":
            return f"{spec.spread_label} (3M MA)"
    return FEATURE_LABELS.get(feature_name, {}).get("label", feature_name)
