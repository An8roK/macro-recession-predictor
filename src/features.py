"""Point-in-time feature engineering with explicit macro publication lag shifts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEFAULT_REGION, FEATURE_COLUMNS
from src.ingest import pivot_to_month_end

RAW_FEATURE_SERIES: tuple[str, ...] = (
    "GS10",
    "TB3MS",
    "FEDFUNDS",
    "BAA",
    "CPIAUCSL",
    "INDPRO",
    "UNRATE",
    "PAYEMS",
)

REQUIRED_SERIES: tuple[str, ...] = ("USREC",) + RAW_FEATURE_SERIES


def compute_features_from_monthly(
    monthly_wide: pd.DataFrame,
    region: str = DEFAULT_REGION,
) -> pd.DataFrame:
    """Compute the 12 point-in-time macro features from a month-end wide DataFrame.

    Publication lag & missing-release discipline:
    - On the resampled monthly raw feature series (strictly excluding `USREC`),
      apply strictly causal `.ffill(limit=1)` BEFORE computing YoY changes and
      rolling windows so a single unpublished release (e.g., the Oct 2025 federal
      shutdown gap in CPIAUCSL/UNRATE) or a 1-month trailing release lag carries
      forward the prior published print rather than cascading NaNs across 12+ months.
      Linear interpolation is intentionally never used to prevent look-ahead bias.
    - Market yields (GS10, TB3MS, FEDFUNDS, BAA) are observed by month-end `t` (0M shift).
    - Macro releases (CPIAUCSL, INDPRO, UNRATE, PAYEMS) are published in month `t+1`,
      so their monthly transformations are computed first and then shifted by 1 month
      (`shift(1)`) to eliminate look-ahead bias at `as_of_date = t`.
    - `target_recession` (`USREC`) is never forward-filled and remains nullable on
      trailing live rows when market features lead `USREC` availability.
    """
    missing = [col for col in REQUIRED_SERIES if col not in monthly_wide.columns]
    if missing:
        raise ValueError(f"Missing required FRED series for feature engineering: {missing}")

    df = monthly_wide.sort_index().copy()
    # Ensure a contiguous month-end index so .shift(k) and rolling(k) strictly represent k months.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="ME")
    df = df.reindex(full_index)
    df.index.name = "as_of_date"

    # Strictly causal 1-month carry-forward on raw feature series only (never USREC).
    # Leading burn-in NaNs (e.g., FEDFUNDS prior to Jul 1954) stay NaN because ffill only looks backward.
    raw_feats = df[list(RAW_FEATURE_SERIES)].ffill(limit=1)

    feats = pd.DataFrame(index=df.index)

    # 1-2. Yield curve (0M publication lag)
    term_spread = raw_feats["GS10"] - raw_feats["TB3MS"]
    feats["term_spread_10y3m"] = term_spread
    feats["term_spread_3m_ma"] = term_spread.rolling(window=3, min_periods=3).mean()

    # 3-4. Monetary policy stance (0M publication lag)
    feats["fedfunds_level"] = raw_feats["FEDFUNDS"]
    feats["fedfunds_12m_chg"] = raw_feats["FEDFUNDS"] - raw_feats["FEDFUNDS"].shift(12)

    # 5-6. Corporate credit conditions (0M publication lag)
    credit_spread = raw_feats["BAA"] - raw_feats["GS10"]
    feats["credit_spread_baa"] = credit_spread
    feats["credit_spread_6m_chg"] = credit_spread - credit_spread.shift(6)

    # 7-8. Inflation (1M publication lag applied after monthly transforms)
    cpi = raw_feats["CPIAUCSL"]
    cpi_yoy_raw = (cpi / cpi.shift(12) - 1.0) * 100.0
    cpi_3m_ann_raw = ((cpi / cpi.shift(3)) ** 4 - 1.0) * 100.0
    feats["cpi_yoy"] = cpi_yoy_raw.shift(1)
    feats["cpi_3m_ann"] = cpi_3m_ann_raw.shift(1)

    # 9-10. Real activity — Industrial Production (1M publication lag)
    indpro = raw_feats["INDPRO"]
    indpro_yoy_raw = (indpro / indpro.shift(12) - 1.0) * 100.0
    indpro_6m_chg_raw = (indpro / indpro.shift(6) - 1.0) * 100.0
    feats["indpro_yoy"] = indpro_yoy_raw.shift(1)
    feats["indpro_6m_chg"] = indpro_6m_chg_raw.shift(1)

    # 11. Labor market — Sahm Rule signal (1M publication lag)
    # 3M rolling mean of UNRATE minus its trailing 12M minimum, then lagged 1M.
    unrate_3m = raw_feats["UNRATE"].rolling(window=3, min_periods=3).mean()
    unrate_3m_min12 = unrate_3m.rolling(window=12, min_periods=12).min()
    sahm_raw = unrate_3m - unrate_3m_min12
    feats["sahm_rule_signal"] = sahm_raw.shift(1)

    # 12. Labor market — Nonfarm Payrolls YoY (1M publication lag)
    payems = raw_feats["PAYEMS"]
    payems_yoy_raw = (payems / payems.shift(12) - 1.0) * 100.0
    feats["payems_yoy"] = payems_yoy_raw.shift(1)

    # Drop initial burn-in rows (and any row where a 2+ month gap leaves NaNs in features).
    feats = feats.dropna(subset=list(FEATURE_COLUMNS)).copy()
    if feats.empty:
        raise ValueError("No rows have complete feature coverage after burn-in.")

    # Attach actual NBER target without ever forward-filling missing target labels.
    target_series = df["USREC"].reindex(feats.index)
    target_int = pd.Series(
        [int(round(v)) if pd.notna(v) else pd.NA for v in target_series],
        index=feats.index,
        dtype="Int64",
    )

    out = pd.DataFrame(index=feats.index)
    out["region"] = region
    out["as_of_date"] = feats.index
    out["target_recession"] = target_int
    for col in FEATURE_COLUMNS:
        out[col] = feats[col].astype("float64")
    out["updated_at"] = pd.Timestamp.now(tz="UTC")

    # Sanity check against infinite values from unexpected zero denominators
    numeric_vals = out[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric_vals).all():
        raise ValueError("Non-finite values detected in engineered feature matrix.")

    return out.reset_index(drop=True)


def build_monthly_features(
    raw_df: pd.DataFrame,
    region: str = DEFAULT_REGION,
) -> pd.DataFrame:
    """End-to-end helper: pivot raw FRED observations and engineer monthly features."""
    monthly_wide = pivot_to_month_end(raw_df)
    return compute_features_from_monthly(monthly_wide, region=region)
