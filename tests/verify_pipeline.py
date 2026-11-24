"""Fast unit, feature-math, strict walk-forward embargo, Phase 2 multi-region, and BQ smoke checks.

Usage:
    python tests/verify_pipeline.py            # Run offline synthetic unit + leakage tests
    python tests/verify_pipeline.py --bq-smoke # Also run live BigQuery read & row-count smoke checks
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure workspace root is importable when executed directly as `python tests/verify_pipeline.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import warehouse as warehouse_mod
from src.config import (
    ALL_REGIONS,
    ALL_TABLES,
    FEATURE_COLUMNS,
    GLOBAL_REGION,
    GLOBAL_WEIGHTS,
    HORIZONS,
    MODEL_LOGISTIC,
    MODEL_REGIONS,
    REGION_SPECS,
)
from src.features import (
    build_cli_downturn_target,
    build_gdp_recession_target,
    build_international_features,
    build_monthly_features,
    compute_features_from_monthly,
)
from src.ingest import pivot_to_month_end
from src.modeling import (
    build_global_aggregate,
    build_horizon_target,
    get_embargoed_train_mask,
    run_walk_forward_models,
)


def _make_synthetic_monthly_wide(
    start: str = "1954-01-31",
    periods: int = 440,
    seed: int = 7,
) -> pd.DataFrame:
    """Create deterministic synthetic US macro series with periodic recession regimes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq="ME")
    t = np.arange(periods, dtype=float)

    cycle = np.sin(2.0 * np.pi * t / 72.0)
    usrec = (cycle > 0.80).astype(float)

    lead_stress = np.sin(2.0 * np.pi * (t + 9.0) / 72.0)
    gs10 = 5.5 - 0.8 * lead_stress + rng.normal(0.0, 0.15, periods)
    tb3ms = 4.0 + 1.1 * lead_stress + rng.normal(0.0, 0.15, periods)
    fedfunds = tb3ms + 0.3 + rng.normal(0.0, 0.1, periods)
    baa = gs10 + 1.8 + 0.9 * np.maximum(cycle, 0.0) + rng.normal(0.0, 0.1, periods)

    cpi = 100.0 * np.exp(np.cumsum(0.0025 + rng.normal(0.0, 0.0005, periods)))
    indpro = 100.0 * np.exp(np.cumsum(0.002 - 0.008 * usrec + rng.normal(0.0, 0.001, periods)))
    payems = 50000.0 * np.exp(np.cumsum(0.0015 - 0.004 * usrec + rng.normal(0.0, 0.0005, periods)))
    unrate = np.clip(4.5 + 2.5 * np.maximum(cycle, 0.0) + rng.normal(0.0, 0.1, periods), 3.0, 12.0)

    return pd.DataFrame(
        {
            "USREC": usrec,
            "GS10": gs10,
            "TB3MS": tb3ms,
            "FEDFUNDS": fedfunds,
            "BAA": baa,
            "CPIAUCSL": cpi,
            "INDPRO": indpro,
            "UNRATE": unrate,
            "PAYEMS": payems,
        },
        index=idx,
    )


def _make_synthetic_intl_raw(
    region: str,
    start: str = "1995-01-31",
    periods: int = 300,
    seed: int = 21,
) -> pd.DataFrame:
    """Create deterministic synthetic raw_macro_series for EU, JP, CN, or IN."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq="ME")
    t = np.arange(periods, dtype=float)
    spec = REGION_SPECS[region]
    roles = spec.series_roles
    ts = pd.Timestamp.now(tz="UTC")

    cycle = np.sin(2.0 * np.pi * t / 60.0)
    rows: list[dict] = []

    def add_series(sid: str, dates: pd.DatetimeIndex, vals: np.ndarray) -> None:
        for dt, v in zip(dates, vals):
            rows.append(
                {
                    "region": region,
                    "series_id": sid,
                    "observation_date": dt,
                    "value": float(v),
                    "ingested_at": ts,
                }
            )

    policy = 3.0 + 1.5 * np.sin(2.0 * np.pi * (t + 6.0) / 60.0) + rng.normal(0, 0.05, periods)
    cpi = 100.0 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.0004, periods)))
    equity = 100.0 * np.exp(np.cumsum(0.005 - 0.015 * np.maximum(cycle, 0) + rng.normal(0, 0.01, periods)))
    reer = 100.0 + 5.0 * np.cos(2.0 * np.pi * t / 48.0) + rng.normal(0, 0.3, periods)
    cli = 100.0 - 1.5 * cycle + rng.normal(0, 0.1, periods)

    add_series(roles["policy_rate"], idx, policy)
    add_series(roles["cpi"], idx, cpi)
    add_series(roles["equity"], idx, equity)
    add_series(roles["reer"], idx, reer)

    if "yield_10y" in roles:
        y10 = 4.0 - 0.6 * cycle + rng.normal(0, 0.08, periods)
        add_series(roles["yield_10y"], idx, y10)
    if "rate_3m" in roles:
        r3m = policy + 0.25 + rng.normal(0, 0.05, periods)
        add_series(roles["rate_3m"], idx, r3m)
    if "cli" in roles:
        add_series(roles["cli"], idx, cli)

    if spec.target_type == "gdp_2q":
        # Quarterly GDP ending at the prior quarter so trailing month(s) after latest GDP stay NULL
        q_idx = pd.date_range(start=start, end=idx[-4], freq="QS")
        q_t = np.arange(len(q_idx), dtype=float)
        # Create clear 2-quarter negative growth contractions periodically
        q_growth = np.where((q_t % 16 == 5) | (q_t % 16 == 6), -0.012, 0.006)
        gdp = 1000.0 * np.cumprod(1.0 + q_growth)
        add_series(roles["target"], q_idx, gdp)
    else:
        add_series(roles["target"], idx, cli)

    return pd.DataFrame(rows)


def test_pandas3_pivot_and_feature_math() -> None:
    """Verify month-end alignment, exact 12-feature formulas, 1M lag shifts, and trailing ffill."""
    wide = _make_synthetic_monthly_wide(periods=48, seed=11)

    rows = []
    for dt, row in wide.iterrows():
        obs_start = dt.replace(day=1)
        for sid, val in row.items():
            rows.append(
                {
                    "region": "US",
                    "series_id": sid,
                    "observation_date": obs_start,
                    "value": float(val),
                    "ingested_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    raw_df = pd.DataFrame(rows)
    pivoted = pivot_to_month_end(raw_df)
    assert (pivoted.index == wide.index).all(), "pivot_to_month_end must align to month-end ('ME')."

    feats = build_monthly_features(raw_df, region="US")
    assert list(feats.columns[3:15]) == list(FEATURE_COLUMNS), "Exact 12 US feature order mismatch."
    assert feats[list(FEATURE_COLUMNS)].notna().all().all(), "US features must have no NaNs after burn-in."

    last_date = feats["as_of_date"].iloc[-1]
    t_pos = wide.index.get_loc(last_date)
    last_row = feats.iloc[-1]

    expected_term = wide["GS10"].iloc[t_pos] - wide["TB3MS"].iloc[t_pos]
    expected_term_3m = (wide["GS10"] - wide["TB3MS"]).iloc[t_pos - 2 : t_pos + 1].mean()
    expected_ff_12m = wide["FEDFUNDS"].iloc[t_pos] - wide["FEDFUNDS"].iloc[t_pos - 12]
    expected_credit = wide["BAA"].iloc[t_pos] - wide["GS10"].iloc[t_pos]
    expected_credit_6m = expected_credit - (wide["BAA"].iloc[t_pos - 6] - wide["GS10"].iloc[t_pos - 6])

    lag = t_pos - 1
    expected_cpi_yoy = (wide["CPIAUCSL"].iloc[lag] / wide["CPIAUCSL"].iloc[lag - 12] - 1.0) * 100.0
    expected_cpi_3m = ((wide["CPIAUCSL"].iloc[lag] / wide["CPIAUCSL"].iloc[lag - 3]) ** 4 - 1.0) * 100.0
    expected_indpro_yoy = (wide["INDPRO"].iloc[lag] / wide["INDPRO"].iloc[lag - 12] - 1.0) * 100.0
    expected_indpro_6m = (wide["INDPRO"].iloc[lag] / wide["INDPRO"].iloc[lag - 6] - 1.0) * 100.0
    expected_payems_yoy = (wide["PAYEMS"].iloc[lag] / wide["PAYEMS"].iloc[lag - 12] - 1.0) * 100.0

    unrate_3m = wide["UNRATE"].rolling(3).mean()
    expected_sahm = unrate_3m.iloc[lag] - unrate_3m.iloc[lag - 11 : lag + 1].min()

    checks = {
        "term_spread_10y3m": expected_term,
        "term_spread_3m_ma": expected_term_3m,
        "fedfunds_level": wide["FEDFUNDS"].iloc[t_pos],
        "fedfunds_12m_chg": expected_ff_12m,
        "credit_spread_baa": expected_credit,
        "credit_spread_6m_chg": expected_credit_6m,
        "cpi_yoy": expected_cpi_yoy,
        "cpi_3m_ann": expected_cpi_3m,
        "indpro_yoy": expected_indpro_yoy,
        "indpro_6m_chg": expected_indpro_6m,
        "sahm_rule_signal": expected_sahm,
        "payems_yoy": expected_payems_yoy,
    }
    for col, expected in checks.items():
        actual = float(last_row[col])
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-10), (
            f"Formula mismatch for {col}: got {actual}, expected {expected}"
        )

    shocked_wide = wide.copy()
    for lagged_series in ("CPIAUCSL", "INDPRO", "UNRATE", "PAYEMS"):
        shocked_wide.iloc[-1, shocked_wide.columns.get_loc(lagged_series)] *= 2.0
    feats_shocked = compute_features_from_monthly(shocked_wide, region="US")
    np.testing.assert_allclose(
        feats[list(FEATURE_COLUMNS)].to_numpy(),
        feats_shocked[list(FEATURE_COLUMNS)].to_numpy(),
        atol=1e-12,
        err_msg="Lagged macro shock at month T leaked into month T features!",
    )

    partial_wide = wide.copy()
    partial_wide.iloc[-1, partial_wide.columns.get_loc("BAA")] = np.nan
    partial_wide.iloc[-1, partial_wide.columns.get_loc("USREC")] = np.nan
    feats_partial = compute_features_from_monthly(partial_wide, region="US")
    assert feats_partial["as_of_date"].iloc[-1] == wide.index[-1]
    assert pd.isna(feats_partial["target_recession"].iloc[-1])
    print("[PASS] test_pandas3_pivot_and_feature_math")


def test_strict_walk_forward_embargo_and_zero_leakage() -> None:
    """Prove mathematically that no label after t_eval can ever influence predictions at t_eval."""
    wide = _make_synthetic_monthly_wide(periods=440, seed=42)
    feats = compute_features_from_monthly(wide, region="US")

    probe_date = pd.Timestamp("1982-06-30")
    probe_m = probe_date.year * 12 + (probe_date.month - 1)
    dates = feats["as_of_date"]
    months = dates.dt.year * 12 + (dates.dt.month - 1)

    for h in HORIZONS:
        y_h = build_horizon_target(dates, feats["target_recession"], h)
        mask = get_embargoed_train_mask(dates, y_h, eval_date=probe_date, horizon_months=h)
        train_months = months[mask]
        assert len(train_months) > 0
        assert (train_months + h <= probe_m).all()
        assert (train_months < probe_m).all()

    preds_base, metrics_base, contribs_base = run_walk_forward_models(
        feats, horizons=HORIZONS, oos_start_date="1975-01-31", region="US"
    )

    feats_flipped = feats.copy()
    future_mask = feats_flipped["as_of_date"] > probe_date
    feats_flipped.loc[future_mask, "target_recession"] = (
        1 - feats_flipped.loc[future_mask, "target_recession"].astype(int)
    ).astype("Int64")

    preds_flipped, _, _ = run_walk_forward_models(
        feats_flipped, horizons=HORIZONS, oos_start_date="1975-01-31", region="US"
    )

    base_pre = preds_base[preds_base["as_of_date"] <= probe_date].sort_values(
        ["horizon_months", "model_name", "as_of_date"]
    )
    flip_pre = preds_flipped[preds_flipped["as_of_date"] <= probe_date].sort_values(
        ["horizon_months", "model_name", "as_of_date"]
    )

    assert len(base_pre) == len(flip_pre) and len(base_pre) > 0
    np.testing.assert_allclose(
        base_pre["recession_prob"].to_numpy(),
        flip_pre["recession_prob"].to_numpy(),
        atol=1e-12,
        err_msg="Future label leakage detected!",
    )

    for h in HORIZONS:
        h_preds = preds_base[
            (preds_base["horizon_months"] == h) & (preds_base["model_name"] == MODEL_LOGISTIC)
        ]
        assert int(h_preds["actual_recession"].isna().sum()) == h
        m_row = metrics_base[
            (metrics_base["horizon_months"] == h) & (metrics_base["model_name"] == MODEL_LOGISTIC)
        ].iloc[0]
        assert int(m_row["n_obs"]) == int(h_preds["actual_recession"].notna().sum())

    assert len(contribs_base) == len(HORIZONS) * len(FEATURE_COLUMNS)
    np.testing.assert_allclose(
        contribs_base["log_odds_contrib"].to_numpy(),
        (contribs_base["coef"] * contribs_base["standardized_value"]).to_numpy(),
        atol=1e-12,
    )
    print("[PASS] test_strict_walk_forward_embargo_and_zero_leakage")


def test_mid_series_shutdown_gap_and_trailing_usrec_lag() -> None:
    """Simulate a single missing mid-series month (like Oct 2025) and a 1M trailing USREC lag."""
    wide = _make_synthetic_monthly_wide(periods=360, seed=99)
    gap_idx = -13
    gap_date = wide.index[gap_idx]
    post_gap_date = wide.index[gap_idx + 1]

    wide_with_gap = wide.copy()
    wide_with_gap.loc[gap_date, "CPIAUCSL"] = np.nan
    wide_with_gap.loc[gap_date, "UNRATE"] = np.nan
    wide_with_gap.iloc[-1, wide_with_gap.columns.get_loc("USREC")] = np.nan

    feats = compute_features_from_monthly(wide_with_gap, region="US")
    assert feats["as_of_date"].iloc[-1] == wide.index[-1]
    assert len(feats[feats["as_of_date"] > gap_date]) == 12
    assert pd.isna(feats["target_recession"].iloc[-1])

    # Prove causal carry-forward (not linear interpolation)
    wide_shocked = wide_with_gap.copy()
    wide_shocked.loc[post_gap_date, "CPIAUCSL"] *= 1.5
    wide_shocked.loc[post_gap_date, "UNRATE"] += 5.0
    feats_shocked = compute_features_from_monthly(wide_shocked, region="US")

    row_base = feats.loc[feats["as_of_date"] == post_gap_date, list(FEATURE_COLUMNS)].to_numpy()
    row_shocked = feats_shocked.loc[
        feats_shocked["as_of_date"] == post_gap_date, list(FEATURE_COLUMNS)
    ].to_numpy()
    np.testing.assert_allclose(row_base, row_shocked, atol=1e-12)

    preds, metrics, contribs = run_walk_forward_models(
        feats, horizons=HORIZONS, oos_start_date="1975-01-31", region="US"
    )
    latest_preds = preds[preds["as_of_date"] == wide.index[-1]]
    assert len(latest_preds) == len(HORIZONS) * 2
    assert latest_preds["actual_recession"].isna().all()
    assert len(contribs) == len(HORIZONS) * len(FEATURE_COLUMNS)
    assert len(metrics) == len(HORIZONS) * 2
    print("[PASS] test_mid_series_shutdown_gap_and_trailing_usrec_lag")


def test_gdp_2q_target_labeler() -> None:
    """Verify 2-consecutive-negative-quarter GDP labeling, single-dip rejection, and monthly expansion."""
    # Q1=100, Q2=102 (+), Q3=101 (-1 single dip!), Q4=103 (+), Q5=102 (-), Q6=100 (-), Q7=101 (+)
    q_dates = pd.date_range("2020-01-01", periods=7, freq="QS")
    gdp = pd.Series([100.0, 102.0, 101.0, 103.0, 102.0, 100.0, 101.0], index=q_dates)
    monthly = build_gdp_recession_target(gdp)

    # First quarter (2020 Q1) has undefined QoQ -> 6 labeled quarters * 3 months = 18 monthly labels
    assert len(monthly) == 18
    # Single-quarter dip in 2020 Q3 must NOT be flagged as recession
    assert int(monthly.loc["2020-09-30"]) == 0
    # 2 consecutive negative quarters: 2021 Q1 & 2021 Q2 (6 months) MUST all be flagged 1
    for m_str in ("2021-01-31", "2021-02-28", "2021-03-31", "2021-04-30", "2021-05-31", "2021-06-30"):
        assert int(monthly.loc[m_str]) == 1, f"Expected recession=1 on {m_str}"
    assert int(monthly.sum()) == 6
    # Recovery quarter 2021 Q3 must be 0
    assert int(monthly.loc["2021-09-30"]) == 0
    print("[PASS] test_gdp_2q_target_labeler")


def test_cli_downturn_target_labeler_and_tautology_guard() -> None:
    """Verify OECD CLI < 99.5 labeling and strict CLI exclusion from CN/IN feature matrices."""
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    cli = pd.Series([100.2, 99.50, 99.49, 98.0, np.nan], index=dates)
    labels = build_cli_downturn_target(cli, threshold=99.5)

    assert int(labels.iloc[0]) == 0
    assert int(labels.iloc[1]) == 0  # strictly < 99.5
    assert int(labels.iloc[2]) == 1
    assert int(labels.iloc[3]) == 1
    assert pd.isna(labels.iloc[4])

    # Confirm CN and IN exclude cli_level_dev and cli_6m_chg from active features & leave them NULL
    for reg in ("CN", "IN"):
        assert "cli_level_dev" not in REGION_SPECS[reg].feature_columns
        assert "cli_6m_chg" not in REGION_SPECS[reg].feature_columns
        raw_intl = _make_synthetic_intl_raw(reg, periods=120, seed=31)
        feats_intl = build_international_features(raw_intl, region=reg)
        assert feats_intl["cli_level_dev"].isna().all(), f"CLI feature must be NULL for {reg}"
        assert feats_intl["cli_6m_chg"].isna().all(), f"CLI feature must be NULL for {reg}"
        assert feats_intl[list(REGION_SPECS[reg].feature_columns)].notna().all().all()
    print("[PASS] test_cli_downturn_target_labeler_and_tautology_guard")


def test_eu_jp_gdp_settlement_embargo() -> None:
    """Prove EU/JP enforce t_train + h + 3 <= t_eval and zero leakage on quarterly GDP targets."""
    raw_eu = _make_synthetic_intl_raw("EU", start="1995-01-31", periods=300, seed=55)
    feats_eu = build_international_features(raw_eu, region="EU")
    # Last month after final GDP quarter must stay NULL (live forecast row)
    assert pd.isna(feats_eu["target_recession"].iloc[-1])

    probe_date = pd.Timestamp("2015-06-30")
    probe_m = probe_date.year * 12 + (probe_date.month - 1)
    dates = feats_eu["as_of_date"]
    months = dates.dt.year * 12 + (dates.dt.month - 1)

    for h in HORIZONS:
        y_h = build_horizon_target(dates, feats_eu["target_recession"], h)
        mask = get_embargoed_train_mask(
            dates, y_h, eval_date=probe_date, horizon_months=h, settlement_lag_months=3
        )
        train_months = months[mask]
        assert len(train_months) > 0
        assert (train_months + h + 3 <= probe_m).all(), (
            f"EU +3M settlement embargo violated for h={h}!"
        )

    # End-to-end future-label perturbation proof on EU with +3M settlement embargo
    preds_base, metrics_base, contribs_base = run_walk_forward_models(
        feats_eu, horizons=HORIZONS, oos_start_date="2010-01-31", region="EU"
    )
    feats_flipped = feats_eu.copy()
    future_labeled = (feats_flipped["as_of_date"] > probe_date) & feats_flipped["target_recession"].notna()
    feats_flipped.loc[future_labeled, "target_recession"] = (
        1 - feats_flipped.loc[future_labeled, "target_recession"].astype(int)
    ).astype("Int64")
    preds_flipped, _, _ = run_walk_forward_models(
        feats_flipped, horizons=HORIZONS, oos_start_date="2010-01-31", region="EU"
    )

    base_pre = preds_base[preds_base["as_of_date"] <= probe_date].sort_values(
        ["horizon_months", "model_name", "as_of_date"]
    )
    flip_pre = preds_flipped[preds_flipped["as_of_date"] <= probe_date].sort_values(
        ["horizon_months", "model_name", "as_of_date"]
    )
    np.testing.assert_allclose(
        base_pre["recession_prob"].to_numpy(),
        flip_pre["recession_prob"].to_numpy(),
        atol=1e-12,
        err_msg="Future label leaked past EU +3M settlement embargo!",
    )
    assert len(contribs_base) == len(HORIZONS) * len(REGION_SPECS["EU"].feature_columns)
    assert len(metrics_base) == len(HORIZONS) * 2

    # Verify JP-only causal unlimited ffill on BIS:WS_CBPOL/M.JP (and per-region logistic_c config)
    assert REGION_SPECS["US"].logistic_c == 0.5
    for reg in ("EU", "JP", "CN", "IN"):
        assert REGION_SPECS[reg].logistic_c == 0.05

    raw_jp = _make_synthetic_intl_raw("JP", start="1995-01-31", periods=120, seed=77)
    jp_pol_sid = REGION_SPECS["JP"].series_roles["policy_rate"]
    # Blank out 18 consecutive months of JP policy rate (simulating BoJ ZIRP/QQE gap)
    gap_mask = (raw_jp["series_id"] == jp_pol_sid) & (
        raw_jp["observation_date"].between("1999-03-01", "2000-08-31")
    )
    raw_jp_gap = raw_jp[~gap_mask].copy()
    feats_jp_gap = build_international_features(raw_jp_gap, region="JP")
    assert len(feats_jp_gap) == 107, f"Expected full uninterrupted JP history (107 rows), got {len(feats_jp_gap)}"
    print("[PASS] test_eu_jp_gdp_settlement_embargo")


def test_global_gdp_weight_renormalization() -> None:
    """Verify exact GDP-weighted composite, pre-2010 dynamic re-normalization, and Tab 3 sums."""
    ts = pd.Timestamp.now(tz="UTC")
    # Date 1 (pre-2010): only US (0.40) and JP (0.06) exist -> norm weights 0.40/0.46 and 0.06/0.46
    # Date 2 (post-2010): all 5 regions exist -> exact weights 0.40, 0.26, 0.22, 0.06, 0.06
    d1 = pd.Timestamp("2006-01-31")
    d2 = pd.Timestamp("2026-10-31")
    records = [
        {"region": "US", "as_of_date": d1, "horizon_months": 12, "model_name": MODEL_LOGISTIC, "recession_prob": 0.20, "is_oos": True, "actual_recession": 0, "updated_at": ts},
        {"region": "JP", "as_of_date": d1, "horizon_months": 12, "model_name": MODEL_LOGISTIC, "recession_prob": 0.50, "is_oos": True, "actual_recession": 0, "updated_at": ts},
        {"region": "US", "as_of_date": d2, "horizon_months": 12, "model_name": MODEL_LOGISTIC, "recession_prob": 0.10, "is_oos": True, "actual_recession": pd.NA, "updated_at": ts},
        {"region": "CN", "as_of_date": d2, "horizon_months": 12, "model_name": MODEL_LOGISTIC, "recession_prob": 0.30, "is_oos": True, "actual_recession": pd.NA, "updated_at": ts},
        {"region": "EU", "as_of_date": d2, "horizon_months": 12, "model_name": MODEL_LOGISTIC, "recession_prob": 0.20, "is_oos": True, "actual_recession": pd.NA, "updated_at": ts},
        {"region": "JP", "as_of_date": d2, "horizon_months": 12, "model_name": MODEL_LOGISTIC, "recession_prob": 0.40, "is_oos": True, "actual_recession": pd.NA, "updated_at": ts},
        {"region": "IN", "as_of_date": d2, "horizon_months": 12, "model_name": MODEL_LOGISTIC, "recession_prob": 0.50, "is_oos": True, "actual_recession": pd.NA, "updated_at": ts},
    ]
    preds_df = pd.DataFrame(records)
    g_preds, g_contribs = build_global_aggregate(preds_df, weights=GLOBAL_WEIGHTS)

    expected_d1 = (0.40 * 0.20 + 0.06 * 0.50) / (0.40 + 0.06)
    expected_d2 = 0.40 * 0.10 + 0.26 * 0.30 + 0.22 * 0.20 + 0.06 * 0.40 + 0.06 * 0.50

    actual_d1 = float(g_preds.loc[g_preds["as_of_date"] == d1, "recession_prob"].iloc[0])
    actual_d2 = float(g_preds.loc[g_preds["as_of_date"] == d2, "recession_prob"].iloc[0])

    assert np.isclose(actual_d1, expected_d1, atol=1e-12), f"Pre-2010 renorm mismatch: {actual_d1} vs {expected_d1}"
    assert np.isclose(actual_d2, expected_d2, atol=1e-12), f"Full 5-region weight mismatch: {actual_d2} vs {expected_d2}"
    assert g_preds["actual_recession"].isna().all()
    assert g_preds["is_oos"].all()

    # Latest GLOBAL contributions must sum exactly to GLOBAL recession_prob on d2
    assert len(g_contribs) == 5
    assert np.isclose(float(g_contribs["log_odds_contrib"].sum()), expected_d2, atol=1e-12)
    print("[PASS] test_global_gdp_weight_renormalization")


def test_warehouse_no_streaming_inserts_guard() -> None:
    """Static guard ensuring warehouse.py uses batch WRITE_TRUNCATE and never insert_rows_json."""
    source = inspect.getsource(warehouse_mod)
    assert "insert_rows_json" not in source, "Forbidden paid streaming insert API referenced in warehouse.py!"
    assert "load_table_from_dataframe" in source, "warehouse.py must use load_table_from_dataframe."
    assert "WRITE_TRUNCATE" in source, "warehouse.py must use WRITE_TRUNCATE."
    print("[PASS] test_warehouse_no_streaming_inserts_guard")


def run_bigquery_smoke_check() -> None:
    """Verify live BigQuery dataset, all 6 regions, non-empty tables, and read query contracts."""
    wh = warehouse_mod.BigQueryWarehouse()
    counts = wh.get_table_row_counts(ALL_TABLES)
    for table_name, count in counts.items():
        assert count > 0, f"BigQuery table {table_name} is empty!"

    regions = wh.read_available_regions()
    for reg in ALL_REGIONS:
        assert reg in regions, f"Missing expected region '{reg}' in BigQuery recession_predictions."

    all_preds = wh.read_all_predictions()
    all_metrics = wh.read_all_metrics()
    g_contribs = wh.read_contributions(GLOBAL_REGION)

    assert not all_preds.empty and all_preds["recession_prob"].between(0.0, 1.0).all()
    assert len(all_metrics) == len(MODEL_REGIONS) * len(HORIZONS) * 2
    assert len(g_contribs) == len(HORIZONS) * len(MODEL_REGIONS)
    print("[PASS] run_bigquery_smoke_check")
    for table_name, count in counts.items():
        print(f"       {table_name}: {count:,} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Macro Recession Predictor pipeline.")
    parser.add_argument(
        "--bq-smoke",
        action="store_true",
        help="Run live BigQuery read & row-count smoke checks against an8rok-labs.macro_recession.",
    )
    args = parser.parse_args()

    test_pandas3_pivot_and_feature_math()
    test_strict_walk_forward_embargo_and_zero_leakage()
    test_mid_series_shutdown_gap_and_trailing_usrec_lag()
    test_gdp_2q_target_labeler()
    test_cli_downturn_target_labeler_and_tautology_guard()
    test_eu_jp_gdp_settlement_embargo()
    test_global_gdp_weight_renormalization()
    test_warehouse_no_streaming_inserts_guard()
    if args.bq_smoke:
        run_bigquery_smoke_check()
    print("\nAll verification checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
