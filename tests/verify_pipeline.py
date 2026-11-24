"""Fast unit, feature-math, strict walk-forward embargo, and BigQuery smoke checks.

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
from src.config import ALL_TABLES, FEATURE_COLUMNS, HORIZONS, MODEL_LOGISTIC, MODEL_RF
from src.features import build_monthly_features, compute_features_from_monthly
from src.ingest import pivot_to_month_end
from src.modeling import (
    build_horizon_target,
    get_embargoed_train_mask,
    run_walk_forward_models,
)


def _make_synthetic_monthly_wide(
    start: str = "1954-01-31",
    periods: int = 440,
    seed: int = 7,
) -> pd.DataFrame:
    """Create deterministic synthetic macro series with periodic recession regimes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq="ME")
    t = np.arange(periods, dtype=float)

    # Periodic synthetic recessions (~14% base rate across multiple cycles)
    cycle = np.sin(2.0 * np.pi * t / 72.0)
    usrec = (cycle > 0.80).astype(float)

    # Leading yield-curve & credit stress before recessions
    lead_stress = np.sin(2.0 * np.pi * (t + 9.0) / 72.0)
    gs10 = 5.5 - 0.8 * lead_stress + rng.normal(0.0, 0.15, periods)
    tb3ms = 4.0 + 1.1 * lead_stress + rng.normal(0.0, 0.15, periods)
    fedfunds = tb3ms + 0.3 + rng.normal(0.0, 0.1, periods)
    baa = gs10 + 1.8 + 0.9 * np.maximum(cycle, 0.0) + rng.normal(0.0, 0.1, periods)

    # Smooth macro indices with recession slowdowns
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


def test_pandas3_pivot_and_feature_math() -> None:
    """Verify month-end alignment, exact 12-feature formulas, 1M lag shifts, and trailing ffill."""
    wide = _make_synthetic_monthly_wide(periods=48, seed=11)

    # Build raw long format with month-start observation dates (just like FRED monthly series)
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
    assert list(feats.columns[3:15]) == list(FEATURE_COLUMNS), "Exact 12 feature column order mismatch."
    assert feats[list(FEATURE_COLUMNS)].notna().all().all(), "Features must have no NaNs after burn-in."

    # Verify exact math on the last row (index -1 -> calendar month T)
    last_date = feats["as_of_date"].iloc[-1]
    t_pos = wide.index.get_loc(last_date)
    last_row = feats.iloc[-1]

    expected_term = wide["GS10"].iloc[t_pos] - wide["TB3MS"].iloc[t_pos]
    expected_term_3m = (wide["GS10"] - wide["TB3MS"]).iloc[t_pos - 2 : t_pos + 1].mean()
    expected_ff_12m = wide["FEDFUNDS"].iloc[t_pos] - wide["FEDFUNDS"].iloc[t_pos - 12]
    expected_credit = wide["BAA"].iloc[t_pos] - wide["GS10"].iloc[t_pos]
    expected_credit_6m = expected_credit - (wide["BAA"].iloc[t_pos - 6] - wide["GS10"].iloc[t_pos - 6])

    # Lagged 1M macro features at t_pos must reflect calculations at (t_pos - 1)
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

    # Publication-lag invariance proof: shocking lagged macro series at T must NOT affect features at T
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

    # Trailing-edge 1M ffill check: missing BAA and missing USREC at T
    partial_wide = wide.copy()
    partial_wide.iloc[-1, partial_wide.columns.get_loc("BAA")] = np.nan
    partial_wide.iloc[-1, partial_wide.columns.get_loc("USREC")] = np.nan
    feats_partial = compute_features_from_monthly(partial_wide, region="US")
    assert feats_partial["as_of_date"].iloc[-1] == wide.index[-1], "Latest as_of_date must be preserved via 1M ffill."
    assert pd.isna(feats_partial["target_recession"].iloc[-1]), "Missing USREC at T must stay NULL (never ffilled)."
    print("[PASS] test_pandas3_pivot_and_feature_math")


def test_strict_walk_forward_embargo_and_zero_leakage() -> None:
    """Prove mathematically that no label after t_eval can ever influence predictions at t_eval."""
    wide = _make_synthetic_monthly_wide(periods=440, seed=42)
    feats = compute_features_from_monthly(wide, region="US")

    # 1. Direct mask boundary checks for every horizon
    probe_date = pd.Timestamp("1982-06-30")
    probe_m = probe_date.year * 12 + (probe_date.month - 1)
    dates = feats["as_of_date"]
    months = dates.dt.year * 12 + (dates.dt.month - 1)

    for h in HORIZONS:
        y_h = build_horizon_target(dates, feats["target_recession"], h)
        mask = get_embargoed_train_mask(dates, y_h, eval_date=probe_date, horizon_months=h)
        train_months = months[mask]
        assert len(train_months) > 0, f"Empty training window for h={h}"
        assert (train_months + h <= probe_m).all(), f"Embargo violated for h={h}: t_train + h > t_eval!"
        assert (train_months < probe_m).all(), f"Eval row leaked into its own training set for h={h}!"

    # 2. End-to-end future-label perturbation test:
    # Flip EVERY USREC label strictly after probe_date and confirm ALL predictions <= probe_date are bit-for-bit identical.
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
        err_msg="Future label leakage detected! Predictions <= probe_date changed when future labels flipped.",
    )

    # 3. Verify live rows (NULL actuals), OOS metrics count, and exact attribution identity
    for h in HORIZONS:
        h_preds = preds_base[
            (preds_base["horizon_months"] == h) & (preds_base["model_name"] == MODEL_LOGISTIC)
        ]
        null_count = int(h_preds["actual_recession"].isna().sum())
        assert null_count == h, f"Expected {h} live NULL rows for h={h} when USREC is full, got {null_count}."

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
    """Simulate a single missing mid-series month (like Oct 2025) and a 1M trailing USREC lag.

    Verifies:
    1. Strictly causal `.ffill(limit=1)` on raw feature series prevents cascading NaNs across
       rolling Sahm/YoY windows so all subsequent months are preserved.
    2. Zero look-ahead (no interpolation): shocking month `T_gap + 1` leaves features at
       `T_gap + 1` (which consume 1M-lagged `T_gap`) completely unchanged.
    3. A trailing row where `USREC` lags market data by 1 month stays nullable and is scored
       as a live row (`is_oos=True`, `actual_recession=NA`) across all 5 horizons including 0M.
    """
    wide = _make_synthetic_monthly_wide(periods=360, seed=99)
    gap_idx = -13  # 12 months before the final observation (mirrors Oct 2025 -> Oct 2026)
    gap_date = wide.index[gap_idx]
    post_gap_date = wide.index[gap_idx + 1]

    wide_with_gap = wide.copy()
    wide_with_gap.loc[gap_date, "CPIAUCSL"] = np.nan
    wide_with_gap.loc[gap_date, "UNRATE"] = np.nan
    # Also simulate USREC lagging market data by 1 month at the trailing edge
    wide_with_gap.iloc[-1, wide_with_gap.columns.get_loc("USREC")] = np.nan

    feats = compute_features_from_monthly(wide_with_gap, region="US")

    # 1. No subsequent months dropped after the mid-series gap, and trailing row preserved
    assert feats["as_of_date"].iloc[-1] == wide.index[-1], (
        f"Expected pipeline to recover through {wide.index[-1]}, stopped at {feats['as_of_date'].iloc[-1]}"
    )
    post_gap_rows = feats[feats["as_of_date"] > gap_date]
    assert len(post_gap_rows) == 12, f"Expected all 12 post-gap months preserved, got {len(post_gap_rows)}"
    assert pd.isna(feats["target_recession"].iloc[-1]), "Trailing missing USREC must remain NULL."

    # 2. Prove causal carry-forward (not linear interpolation):
    # Shocking CPIAUCSL/UNRATE at `post_gap_date` must NOT alter features at `post_gap_date`
    # because `post_gap_date` uses the 1M-lagged `gap_date` print, which must depend only on `gap_date - 1`.
    wide_shocked = wide_with_gap.copy()
    wide_shocked.loc[post_gap_date, "CPIAUCSL"] *= 1.5
    wide_shocked.loc[post_gap_date, "UNRATE"] += 5.0
    feats_shocked = compute_features_from_monthly(wide_shocked, region="US")

    row_base = feats.loc[feats["as_of_date"] == post_gap_date, list(FEATURE_COLUMNS)].to_numpy()
    row_shocked = feats_shocked.loc[
        feats_shocked["as_of_date"] == post_gap_date, list(FEATURE_COLUMNS)
    ].to_numpy()
    np.testing.assert_allclose(
        row_base,
        row_shocked,
        atol=1e-12,
        err_msg="Future value at T_gap+1 leaked into T_gap carry-forward (interpolation leak)!",
    )

    # 3. Verify walk-forward modeling scores the trailing nullable-USREC row cleanly for all horizons
    preds, metrics, contribs = run_walk_forward_models(
        feats, horizons=HORIZONS, oos_start_date="1975-01-31", region="US"
    )
    latest_date = wide.index[-1]
    latest_preds = preds[preds["as_of_date"] == latest_date]
    assert len(latest_preds) == len(HORIZONS) * 2
    assert latest_preds["actual_recession"].isna().all(), (
        "All horizons (including 0M) must have actual_recession=NA on trailing row when USREC lags."
    )
    assert latest_preds["is_oos"].all()
    assert len(contribs) == len(HORIZONS) * len(FEATURE_COLUMNS)
    assert (contribs["as_of_date"] == latest_date).all()
    assert len(metrics) == len(HORIZONS) * 2
    print("[PASS] test_mid_series_shutdown_gap_and_trailing_usrec_lag")


def test_warehouse_no_streaming_inserts_guard() -> None:
    """Static guard ensuring warehouse.py uses batch WRITE_TRUNCATE and never insert_rows_json."""
    source = inspect.getsource(warehouse_mod)
    assert "insert_rows_json" not in source, "Forbidden paid streaming insert API referenced in warehouse.py!"
    assert "load_table_from_dataframe" in source, "warehouse.py must use load_table_from_dataframe."
    assert "WRITE_TRUNCATE" in source, "warehouse.py must use WRITE_TRUNCATE."
    print("[PASS] test_warehouse_no_streaming_inserts_guard")


def run_bigquery_smoke_check() -> None:
    """Verify live BigQuery dataset, non-empty tables, and read query contracts."""
    wh = warehouse_mod.BigQueryWarehouse()
    counts = wh.get_table_row_counts(ALL_TABLES)
    for table_name, count in counts.items():
        assert count > 0, f"BigQuery table {table_name} is empty!"

    preds = wh.read_predictions("US")
    metrics = wh.read_metrics("US")
    contribs = wh.read_contributions("US")
    feats = wh.read_monthly_features("US")

    assert not preds.empty and preds["recession_prob"].between(0.0, 1.0).all()
    assert len(metrics) == len(HORIZONS) * 2
    assert len(contribs) == len(HORIZONS) * len(FEATURE_COLUMNS)
    assert not feats.empty
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
    test_warehouse_no_streaming_inserts_guard()
    if args.bq_smoke:
        run_bigquery_smoke_check()
    print("\nAll verification checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
