"""Point-in-time feature & target engineering for US (Phase 1) and EU/JP/CN/IN (Phase 2)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURE_COLUMNS,
    CLI_DOWNTURN_THRESHOLD,
    DEFAULT_REGION,
    MODEL_REGIONS,
    REGION_SPECS,
    US_FEATURE_COLUMNS,
)
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


def _finalize_unified_features(
    feats: pd.DataFrame,
    target_int: pd.Series,
    active_columns: tuple[str, ...],
    region: str,
) -> pd.DataFrame:
    """Pack regional features into the unified 21-feature BigQuery monthly_features schema."""
    numeric_vals = feats[list(active_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric_vals).all():
        raise ValueError(f"Non-finite values detected in engineered features for region={region}.")

    out = pd.DataFrame(index=feats.index)
    out["region"] = region
    out["as_of_date"] = feats.index
    out["target_recession"] = target_int
    for col in ALL_FEATURE_COLUMNS:
        if col in feats.columns:
            out[col] = feats[col].astype("float64")
        else:
            out[col] = np.nan
    out["updated_at"] = pd.Timestamp.now(tz="UTC")
    return out.reset_index(drop=True)


def compute_features_from_monthly(
    monthly_wide: pd.DataFrame,
    region: str = DEFAULT_REGION,
) -> pd.DataFrame:
    """Compute the 12 Phase 1 US point-in-time macro features from a month-end wide DataFrame.

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
        raise ValueError(f"Missing required FRED series for US feature engineering: {missing}")

    df = monthly_wide.sort_index().copy()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="ME")
    df = df.reindex(full_index)
    df.index.name = "as_of_date"

    # Strictly causal 1-month carry-forward on raw feature series only (never USREC).
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
    unrate_3m = raw_feats["UNRATE"].rolling(window=3, min_periods=3).mean()
    unrate_3m_min12 = unrate_3m.rolling(window=12, min_periods=12).min()
    sahm_raw = unrate_3m - unrate_3m_min12
    feats["sahm_rule_signal"] = sahm_raw.shift(1)

    # 12. Labor market — Nonfarm Payrolls YoY (1M publication lag)
    payems = raw_feats["PAYEMS"]
    payems_yoy_raw = (payems / payems.shift(12) - 1.0) * 100.0
    feats["payems_yoy"] = payems_yoy_raw.shift(1)

    feats = feats.dropna(subset=list(US_FEATURE_COLUMNS)).copy()
    if feats.empty:
        raise ValueError("No rows have complete feature coverage after burn-in for US.")

    target_series = df["USREC"].reindex(feats.index)
    target_int = pd.Series(
        [int(round(v)) if pd.notna(v) else pd.NA for v in target_series],
        index=feats.index,
        dtype="Int64",
    )
    return _finalize_unified_features(feats, target_int, US_FEATURE_COLUMNS, region=region)


def build_monthly_features(
    raw_df: pd.DataFrame,
    region: str = DEFAULT_REGION,
) -> pd.DataFrame:
    """Build monthly features for a single region (dispatches US vs international cleanly)."""
    reg_df = raw_df[raw_df["region"] == region].copy() if "region" in raw_df.columns else raw_df
    if region == "US":
        monthly_wide = pivot_to_month_end(reg_df)
        return compute_features_from_monthly(monthly_wide, region=region)
    return build_international_features(reg_df, region=region)


def build_gdp_recession_target(gdp_series: pd.Series) -> pd.Series:
    """Label technical recessions: >=2 consecutive negative QoQ Real GDP quarters.

    Every quarter belonging to a run of at least 2 consecutive quarters with negative
    QoQ growth is flagged `1` and expanded to its 3 calendar month-ends. Months after
    the latest reported GDP quarter are not generated (staying NULL when reindexed).
    Note: The walk-forward model adds a +3M GDP settlement embargo (`t_train + h + 3 <= t_eval`)
    so the 2nd-quarter confirmation never leaks into training.
    """
    s = gdp_series.dropna().copy()
    s.index = pd.to_datetime(s.index)
    q = s.sort_index().resample("QE").last().dropna()
    if len(q) < 3:
        raise ValueError("Need at least 3 quarterly GDP observations to compute 2Q contractions.")

    qoq = q.pct_change()
    neg = qoq < 0.0
    in_rec_q = (neg & neg.shift(1, fill_value=False)) | (neg & neg.shift(-1, fill_value=False))
    # Drop the first quarter where QoQ growth is undefined.
    in_rec_q = in_rec_q.iloc[1:]

    monthly_labels: dict[pd.Timestamp, int] = {}
    for q_end, flag in in_rec_q.items():
        q_start = (q_end - pd.offsets.QuarterBegin(startingMonth=1)).normalize()
        for m_end in pd.date_range(q_start, q_end, freq="ME"):
            monthly_labels[m_end] = int(bool(flag))

    return pd.Series(monthly_labels, dtype="Int64").sort_index()


def build_cli_downturn_target(
    cli_series: pd.Series,
    threshold: float = CLI_DOWNTURN_THRESHOLD,
) -> pd.Series:
    """Label growth-cycle downturns: 1 if monthly OECD CLI < threshold (99.5), else 0."""
    s = cli_series.copy()
    s.index = pd.to_datetime(s.index)
    monthly = s.sort_index().resample("ME").last()
    labels = [
        (1 if float(v) < threshold else 0) if pd.notna(v) else pd.NA for v in monthly
    ]
    return pd.Series(labels, index=monthly.index, dtype="Int64")


def build_international_features(
    raw_region_df: pd.DataFrame,
    region: str,
) -> pd.DataFrame:
    """Compute Option A target and point-in-time features for EU, JP, CN, or IN."""
    if region not in REGION_SPECS or region in ("US", "GLOBAL"):
        raise ValueError(f"Unsupported international region: {region}")

    spec = REGION_SPECS[region]
    roles = spec.series_roles
    target_sid = roles["target"]

    work = raw_region_df.copy()
    work["observation_date"] = pd.to_datetime(work["observation_date"])

    # 1. Extract target series BEFORE monthly feature resampling so quarterly GDP dates stay exact
    target_raw = work[work["series_id"] == target_sid].set_index("observation_date")["value"]
    if target_raw.empty:
        raise ValueError(f"Missing target series '{target_sid}' for region={region}.")

    if spec.target_type == "gdp_2q":
        target_monthly = build_gdp_recession_target(target_raw)
    elif spec.target_type == "cli_995":
        target_monthly = build_cli_downturn_target(target_raw, threshold=CLI_DOWNTURN_THRESHOLD)
    else:
        raise ValueError(f"Unsupported target_type '{spec.target_type}' for region={region}.")

    # 2. Pivot feature series to contiguous monthly 'ME' grid (excluding target for CN/IN tautology safety)
    feature_sids = {sid for role, sid in roles.items() if role != "target"}
    feat_raw_df = work[work["series_id"].isin(feature_sids)]
    monthly_wide = pivot_to_month_end(feat_raw_df)

    missing_sids = [sid for sid in feature_sids if sid not in monthly_wide.columns]
    if missing_sids:
        raise ValueError(f"Missing required feature series for {region}: {missing_sids}")

    full_index = pd.date_range(monthly_wide.index.min(), monthly_wide.index.max(), freq="ME")
    monthly_wide = monthly_wide.reindex(full_index)
    monthly_wide.index.name = "as_of_date"

    # Strictly causal 1-month carry-forward on raw monthly feature levels before rolling/diff.
    # Exception for JP policy rate only: BIS omits WS_CBPOL/M.JP during BoJ's monetary-base
    # targeting regimes (ZIRP/QE/QQE), so carry forward the last announced BoJ rate causally
    # (after the first valid observation) to preserve uninterrupted 1995-2026 JP history.
    raw_feats = monthly_wide.ffill(limit=1)
    if region == "JP":
        jp_policy_sid = roles["policy_rate"]
        raw_feats[jp_policy_sid] = monthly_wide[jp_policy_sid].ffill()

    feats = pd.DataFrame(index=raw_feats.index)

    # Headline CPI (1M lag)
    cpi = raw_feats[roles["cpi"]]
    cpi_yoy_raw = (cpi / cpi.shift(12) - 1.0) * 100.0
    cpi_3m_ann_raw = ((cpi / cpi.shift(3)) ** 4 - 1.0) * 100.0
    feats["cpi_yoy"] = cpi_yoy_raw.shift(1)
    feats["cpi_3m_ann"] = cpi_3m_ann_raw.shift(1)

    # Central Bank Policy Rate (0M lag)
    policy = raw_feats[roles["policy_rate"]]
    feats["policy_rate_level"] = policy
    feats["policy_rate_12m_chg"] = policy - policy.shift(12)

    # Region-specific Rate Spread (0M lag; IN uses already-1M-lagged cpi_yoy for point-in-time real rate)
    if region == "EU":
        rate_spread = raw_feats[roles["yield_10y"]] - raw_feats[roles["rate_3m"]]
    elif region == "JP":
        rate_spread = raw_feats[roles["yield_10y"]] - policy
    elif region == "CN":
        rate_spread = raw_feats[roles["rate_3m"]] - policy
    elif region == "IN":
        rate_spread = policy - feats["cpi_yoy"]
    else:
        raise ValueError(f"No rate spread rule configured for region={region}")

    feats["rate_spread_level"] = rate_spread
    feats["rate_spread_3m_ma"] = rate_spread.rolling(window=3, min_periods=3).mean()

    # Broad Equity Index (0M lag)
    equity = raw_feats[roles["equity"]]
    feats["equity_12m_ret"] = (equity / equity.shift(12) - 1.0) * 100.0
    equity_peak_12m = equity.rolling(window=12, min_periods=12).max()
    feats["equity_12m_dd"] = (equity / equity_peak_12m - 1.0) * 100.0

    # Real Broad Effective Exchange Rate (1M lag)
    reer = raw_feats[roles["reer"]]
    feats["reer_12m_chg"] = ((reer / reer.shift(12) - 1.0) * 100.0).shift(1)

    # OECD CLI features (1M lag) — ONLY for EU and JP; strictly NULL & excluded for CN and IN
    if "cli" in roles:
        cli = raw_feats[roles["cli"]]
        feats["cli_level_dev"] = (cli - 100.0).shift(1)
        feats["cli_6m_chg"] = (cli - cli.shift(6)).shift(1)

    # Drop leading burn-in rows using strictly this region's active features
    active_cols = spec.feature_columns
    feats = feats.dropna(subset=list(active_cols)).copy()
    if feats.empty:
        raise ValueError(f"No rows have complete feature coverage after burn-in for {region}.")

    # Align target without forward-filling (trailing months beyond latest GDP quarter stay NULL)
    target_aligned = target_monthly.reindex(feats.index).astype("Int64")
    return _finalize_unified_features(feats, target_aligned, active_cols, region=region)


def build_all_regions_features(
    raw_df: pd.DataFrame,
    regions: Iterable[str] = MODEL_REGIONS,
) -> pd.DataFrame:
    """Build and concatenate unified monthly_features across all configured modeled regions."""
    frames = [build_monthly_features(raw_df, region=reg) for reg in regions]
    return pd.concat(frames, ignore_index=True)
