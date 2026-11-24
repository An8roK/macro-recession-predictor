"""Stateless Streamlit dashboard reading precomputed multi-region forecasts from BigQuery."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import (
    MODEL_DISPLAY,
    build_calibration_chart,
    build_cross_region_validation_chart,
    build_drivers_chart,
    build_global_drivers_chart,
    build_global_risk_curve_chart,
    build_global_timeline_chart,
    build_historical_timeline_chart,
    build_risk_curve_chart,
)
from src.config import (
    FEATURE_LABELS,
    GLOBAL_REGION,
    GLOBAL_WEIGHTS,
    HORIZONS,
    MODEL_LOGISTIC,
    MODEL_NAMES,
    MODEL_REGIONS,
    MODEL_RF,
    REGION_SPECS,
    get_feature_label,
)
from src.warehouse import BigQueryWarehouse


@st.cache_data(ttl=3600, show_spinner=False)
def _load_regions() -> list[str]:
    return BigQueryWarehouse().read_available_regions()


@st.cache_data(ttl=3600, show_spinner="Loading BigQuery forecasts...")
def _load_region_data(region: str) -> dict[str, pd.DataFrame]:
    wh = BigQueryWarehouse()
    if region == GLOBAL_REGION:
        return {
            "all_predictions": wh.read_all_predictions(),
            "all_metrics": wh.read_all_metrics(),
            "contributions": wh.read_contributions(region=GLOBAL_REGION),
            "us_features": wh.read_monthly_features(region="US"),
        }
    return {
        "predictions": wh.read_predictions(region=region),
        "metrics": wh.read_metrics(region=region),
        "contributions": wh.read_contributions(region=region),
        "features": wh.read_monthly_features(region=region),
    }


def _render_sidebar(regions: list[str]) -> tuple[str, str, int]:
    st.sidebar.title("Macro Recession Predictor")
    st.sidebar.caption("Global walk-forward macro risk monitor (BigQuery-backed)")

    default_idx = regions.index(GLOBAL_REGION) if GLOBAL_REGION in regions else 0
    selected_region = st.sidebar.selectbox(
        "Region",
        options=regions,
        index=default_idx,
        format_func=lambda r: f"{r} — {REGION_SPECS[r].name}" if r in REGION_SPECS else r,
    )

    selected_model = st.sidebar.selectbox(
        "Model",
        options=list(MODEL_NAMES),
        format_func=lambda m: MODEL_DISPLAY.get(m, m),
        index=0,
    )

    default_horizon_idx = list(HORIZONS).index(12) if 12 in HORIZONS else 0
    selected_horizon = st.sidebar.selectbox(
        "Forecast Horizon",
        options=list(HORIZONS),
        format_func=lambda h: f"{h}M ({'Nowcast' if h == 0 else f'{h}-Month Ahead'})",
        index=default_horizon_idx,
    )

    st.sidebar.divider()
    if st.sidebar.button("Refresh BigQuery Cache", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    return selected_region, selected_model, int(selected_horizon)


def _render_top_banner(
    region: str,
    latest_date: pd.Timestamp,
    updated_at: pd.Timestamp,
    selected_model: str,
) -> None:
    spec = REGION_SPECS.get(region)
    badge = spec.regime_badge if spec else region
    name = spec.name if spec else region
    st.info(
        f"**Economy:** `{name} ({region})`  |  "
        f"**Regime Definition:** `{badge}`  |  "
        f"**Latest As-Of Date:** `{latest_date.strftime('%Y-%m-%d')}`  |  "
        f"**Model:** `{MODEL_DISPLAY.get(selected_model, selected_model)}`  |  "
        f"**Updated:** `{updated_at.strftime('%Y-%m-%d %H:%M UTC')}`"
    )


def _render_global_risk_curve_tab(
    all_preds: pd.DataFrame,
    selected_model: str,
) -> None:
    glob_preds = all_preds[all_preds["region"] == GLOBAL_REGION]
    latest_date = pd.Timestamp(glob_preds["as_of_date"].max())
    updated_at = pd.Timestamp(glob_preds["updated_at"].max())
    _render_top_banner(GLOBAL_REGION, latest_date, updated_at, selected_model)

    latest_glob = glob_preds[
        (glob_preds["as_of_date"] == latest_date) & (glob_preds["model_name"] == selected_model)
    ].set_index("horizon_months")

    cols = st.columns(len(HORIZONS))
    for col, h in zip(cols, HORIZONS):
        prob = float(latest_glob.loc[h, "recession_prob"]) * 100.0 if h in latest_glob.index else 0.0
        label = "0M Global Nowcast" if h == 0 else f"{h}M Global Risk"
        col.metric(label=label, value=f"{prob:.1f}%")

    fig = build_global_risk_curve_chart(all_preds=all_preds, selected_model=selected_model)
    st.plotly_chart(fig, width="stretch")

    # Global Risk Matrix table across all 5 economies + GLOBAL
    st.subheader("Global Risk Matrix (Latest As-Of Curve by Economy)")
    matrix_rows = []
    model_preds = all_preds[all_preds["model_name"] == selected_model]
    for reg in (GLOBAL_REGION,) + MODEL_REGIONS:
        sub = model_preds[model_preds["region"] == reg]
        if sub.empty:
            continue
        dt = pd.Timestamp(sub["as_of_date"].max())
        cur = sub[sub["as_of_date"] == dt].set_index("horizon_months")["recession_prob"]
        row = {
            "Region": reg,
            "Economy": REGION_SPECS[reg].name,
            "GDP Weight": "100%" if reg == GLOBAL_REGION else f"{GLOBAL_WEIGHTS[reg] * 100:.0f}%",
            "Regime Definition": REGION_SPECS[reg].regime_badge,
            "As-Of Date": dt.strftime("%Y-%m-%d"),
        }
        for h in HORIZONS:
            row[f"{h}M"] = float(cur[h]) * 100.0 if h in cur.index else float("nan")
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    fmt = {f"{h}M": "{:.1f}%" for h in HORIZONS}
    st.dataframe(matrix_df.style.format(fmt), width="stretch", hide_index=True)


def _render_regional_risk_curve_tab(
    preds_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selected_model: str,
    selected_region: str,
) -> None:
    latest_date = pd.Timestamp(preds_df["as_of_date"].max())
    latest_preds = preds_df[preds_df["as_of_date"] == latest_date].copy()
    updated_at = pd.Timestamp(latest_preds["updated_at"].max())
    _render_top_banner(selected_region, latest_date, updated_at, selected_model)

    model_latest = latest_preds[latest_preds["model_name"] == selected_model].set_index(
        "horizon_months"
    )
    model_metrics = metrics_df[metrics_df["model_name"] == selected_model].set_index(
        "horizon_months"
    )

    cols = st.columns(len(HORIZONS))
    for col, h in zip(cols, HORIZONS):
        prob = float(model_latest.loc[h, "recession_prob"]) * 100.0 if h in model_latest.index else 0.0
        base = (
            float(model_metrics.loc[h, "base_rate"]) * 100.0
            if h in model_metrics.index
            else float("nan")
        )
        delta_str = f"{prob - base:+.1f} pp vs base" if pd.notna(base) else None
        label = "0M Nowcast" if h == 0 else f"{h}M Horizon"
        col.metric(label=label, value=f"{prob:.1f}%", delta=delta_str, delta_color="inverse")

    fig = build_risk_curve_chart(
        latest_preds=latest_preds,
        metrics_df=metrics_df,
        selected_model=selected_model,
        region=selected_region,
    )
    st.plotly_chart(fig, width="stretch")


def _render_global_timeline_tab(
    all_preds: pd.DataFrame,
    us_features_df: pd.DataFrame,
    selected_model: str,
    selected_horizon: int,
) -> None:
    overlay_regions = st.multiselect(
        "Overlay Constituent Economies",
        options=list(MODEL_REGIONS),
        default=["US", "EU", "CN"],
        format_func=lambda r: f"{r} — {REGION_SPECS[r].name}",
    )
    fig = build_global_timeline_chart(
        all_preds=all_preds,
        us_features_df=us_features_df,
        selected_model=selected_model,
        selected_horizon=selected_horizon,
        overlay_regions=overlay_regions,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Global curve is dynamically GDP-weighted across available economies at each historical "
        "date (US 40%, CN 26%, EU 22%, JP 6%, IN 6%, re-normalized pre-2010)."
    )


def _render_regional_timeline_tab(
    preds_df: pd.DataFrame,
    features_df: pd.DataFrame,
    selected_model: str,
    selected_horizon: int,
    selected_region: str,
) -> None:
    ctrl1, ctrl2 = st.columns([1, 2])
    mode = ctrl1.radio(
        "Comparison Mode",
        options=["Compare Models", "Compare Horizons"],
        horizontal=True,
    )

    if mode == "Compare Horizons":
        overlay_horizons = ctrl2.multiselect(
            "Horizons to overlay",
            options=list(HORIZONS),
            default=[0, 6, 12],
            format_func=lambda h: f"{h}M",
        )
        if not overlay_horizons:
            overlay_horizons = [selected_horizon]
        fig = build_historical_timeline_chart(
            preds_df=preds_df,
            features_df=features_df,
            selected_model=selected_model,
            selected_horizon=selected_horizon,
            region=selected_region,
            compare_models=False,
            overlay_horizons=overlay_horizons,
        )
    else:
        compare_models = ctrl2.checkbox(
            f"Overlay {MODEL_DISPLAY[MODEL_RF] if selected_model == MODEL_LOGISTIC else MODEL_DISPLAY[MODEL_LOGISTIC]}",
            value=True,
        )
        fig = build_historical_timeline_chart(
            preds_df=preds_df,
            features_df=features_df,
            selected_model=selected_model,
            selected_horizon=selected_horizon,
            region=selected_region,
            compare_models=compare_models,
            overlay_horizons=None,
        )

    st.plotly_chart(fig, width="stretch")
    spec = REGION_SPECS[selected_region]
    embargo_note = (
        "`t_train + h + 3 <= t_eval` (quarterly GDP settlement buffer)"
        if spec.settlement_lag_months > 0
        else "`t_train + h <= t_eval`"
    )
    st.caption(
        f"Strictly out-of-sample expanding-window forecasts from {spec.oos_start_date[:4]} onward "
        f"with embargo {embargo_note}. Shaded bars mark `{spec.regime_badge}` episodes."
    )


def _render_global_drivers_tab(
    contribs_df: pd.DataFrame,
    selected_horizon: int,
    selected_model: str,
) -> None:
    if selected_model != MODEL_LOGISTIC:
        st.caption(
            "Note: Global regional risk attributions are computed from the primary calibrated "
            "`logistic_l2` composite probabilities."
        )

    fig = build_global_drivers_chart(contribs_df=contribs_df, horizon_months=selected_horizon)
    st.plotly_chart(fig, width="stretch")

    sub = contribs_df[contribs_df["horizon_months"] == selected_horizon].copy()
    total_prob = float(sub["log_odds_contrib"].sum())
    sub["Economy"] = sub["feature_name"].map(lambda k: get_feature_label(k, GLOBAL_REGION))
    sub["GDP Weight (%)"] = sub["coef"] * 100.0
    sub["Regional Prob (%)"] = sub["raw_value"] * 100.0
    sub["Weighted Contrib (pp)"] = sub["log_odds_contrib"] * 100.0
    sub["Share of Global Risk (%)"] = (
        (sub["log_odds_contrib"] / total_prob * 100.0) if total_prob > 0 else 0.0
    )
    sub = sub.sort_values("Weighted Contrib (pp)", ascending=False)

    st.subheader(f"Global Composite Breakdown ({selected_horizon}M Horizon — Total: {total_prob * 100:.1f}%)")
    st.dataframe(
        sub[
            [
                "Economy",
                "GDP Weight (%)",
                "Regional Prob (%)",
                "Weighted Contrib (pp)",
                "Share of Global Risk (%)",
            ]
        ].style.format(
            {
                "GDP Weight (%)": "{:.0f}%",
                "Regional Prob (%)": "{:.1f}%",
                "Weighted Contrib (pp)": "{:.2f} pp",
                "Share of Global Risk (%)": "{:.1f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def _render_regional_drivers_tab(
    contribs_df: pd.DataFrame,
    selected_horizon: int,
    selected_model: str,
    selected_region: str,
) -> None:
    if selected_model != MODEL_LOGISTIC:
        st.caption(
            "Note: Exact linear log-odds attributions (`coef * z-score`) are derived from the "
            "primary calibrated `logistic_l2` model for interpretability."
        )

    fig = build_drivers_chart(
        contribs_df=contribs_df,
        horizon_months=selected_horizon,
        region=selected_region,
    )
    st.plotly_chart(fig, width="stretch")

    sub = contribs_df[contribs_df["horizon_months"] == selected_horizon].copy()
    sub["Indicator"] = sub["feature_name"].map(lambda k: get_feature_label(k, selected_region))
    sub["Unit"] = sub["feature_name"].map(lambda k: FEATURE_LABELS.get(k, {}).get("unit", ""))
    sub["Direction"] = sub["log_odds_contrib"].map(
        lambda v: "Raises Risk" if v > 0 else "Lowers Risk"
    )
    sub = sub.reindex(sub["log_odds_contrib"].abs().sort_values(ascending=False).index)

    table_df = sub[
        [
            "Indicator",
            "raw_value",
            "Unit",
            "standardized_value",
            "coef",
            "log_odds_contrib",
            "Direction",
        ]
    ].rename(
        columns={
            "raw_value": "Latest Reading",
            "standardized_value": "Z-Score",
            "coef": "L2 Coef",
            "log_odds_contrib": "Log-Odds Contrib",
        }
    )

    st.subheader(f"Latest Macro Signal Readings ({selected_region}, {selected_horizon}M Horizon)")
    st.dataframe(
        table_df.style.format(
            {
                "Latest Reading": "{:.2f}",
                "Z-Score": "{:+.2f}",
                "L2 Coef": "{:+.3f}",
                "Log-Odds Contrib": "{:+.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def _render_global_validation_tab(
    all_metrics: pd.DataFrame,
    selected_model: str,
) -> None:
    st.subheader("Cross-Region Out-of-Sample Model Validation Summary")
    fig = build_cross_region_validation_chart(all_metrics=all_metrics, selected_model=selected_model)
    st.plotly_chart(fig, width="stretch")

    display = all_metrics.copy()
    display["_reg_order"] = display["region"].map({r: i for i, r in enumerate(MODEL_REGIONS)})
    display = display.sort_values(["_reg_order", "horizon_months", "model_name"])
    display["Regime"] = display["region"].map(
        lambda r: REGION_SPECS[r].regime_badge if r in REGION_SPECS else r
    )
    display["Horizon"] = display["horizon_months"].map(lambda h: f"{int(h)}M")
    display["Model"] = display["model_name"].map(lambda m: MODEL_DISPLAY.get(m, m))
    display["Eval Window"] = (
        display["eval_start_date"].dt.strftime("%Y-%m")
        + " → "
        + display["eval_end_date"].dt.strftime("%Y-%m")
    )

    cols = [
        "region",
        "Regime",
        "Horizon",
        "Model",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "base_rate",
        "n_obs",
        "n_recessions",
        "Eval Window",
    ]
    renamed = display[cols].rename(
        columns={
            "region": "Region",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "brier_score": "Brier Score",
            "base_rate": "Base Rate",
            "n_obs": "OOS Months",
            "n_recessions": "Downturn Months",
        }
    )
    st.dataframe(
        renamed.style.format(
            {
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
                "Brier Score": "{:.4f}",
                "Base Rate": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def _render_regional_validation_tab(
    preds_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selected_horizon: int,
    selected_region: str,
) -> None:
    spec = REGION_SPECS[selected_region]
    st.subheader(f"Out-of-Sample Model Validation — {spec.name} ({spec.regime_badge})")
    display = metrics_df.copy()
    display["Horizon"] = display["horizon_months"].map(lambda h: f"{int(h)}M")
    display["Model"] = display["model_name"].map(lambda m: MODEL_DISPLAY.get(m, m))
    display["Eval Window"] = (
        display["eval_start_date"].dt.strftime("%Y-%m")
        + " → "
        + display["eval_end_date"].dt.strftime("%Y-%m")
    )

    cols = [
        "Horizon",
        "Model",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "base_rate",
        "n_obs",
        "n_recessions",
        "Eval Window",
    ]
    renamed = display[cols].rename(
        columns={
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "brier_score": "Brier Score",
            "base_rate": "Base Rate",
            "n_obs": "OOS Months",
            "n_recessions": "Downturn Months",
        }
    )
    st.dataframe(
        renamed.style.format(
            {
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
                "Brier Score": "{:.4f}",
                "Base Rate": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    fig = build_calibration_chart(preds_df=preds_df, horizon_months=selected_horizon)
    st.plotly_chart(fig, width="stretch")


def render_dashboard() -> None:
    """Main Streamlit UI entry point (stateless, reads cached BigQuery tables only)."""
    st.set_page_config(
        page_title="Global Macro Recession Predictor",
        page_icon="📉",
        layout="wide",
    )

    try:
        regions = _load_regions()
    except Exception as exc:
        st.error(
            f"Unable to connect to BigQuery warehouse: {exc}. "
            "Verify GCP authentication and run `python -m src.pipeline` first."
        )
        return

    selected_region, selected_model, selected_horizon = _render_sidebar(regions)
    data = _load_region_data(selected_region)

    st.title("Global Macro Recession Risk Monitor")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Recession Risk Curve",
            "Historical Timeline",
            "Signal Drivers",
            "Model Validation",
        ]
    )

    if selected_region == GLOBAL_REGION:
        all_preds = data["all_predictions"]
        all_metrics = data["all_metrics"]
        contribs_df = data["contributions"]
        us_features_df = data["us_features"]

        if all_preds.empty:
            st.warning("No precomputed predictions found in BigQuery. Run `python -m src.pipeline`.")
            return

        with tab1:
            _render_global_risk_curve_tab(all_preds, selected_model)
        with tab2:
            _render_global_timeline_tab(all_preds, us_features_df, selected_model, selected_horizon)
        with tab3:
            _render_global_drivers_tab(contribs_df, selected_horizon, selected_model)
        with tab4:
            _render_global_validation_tab(all_metrics, selected_model)
    else:
        preds_df = data["predictions"]
        metrics_df = data["metrics"]
        contribs_df = data["contributions"]
        features_df = data["features"]

        if preds_df.empty:
            st.warning(
                f"No precomputed predictions found in BigQuery for region `{selected_region}`. "
                "Run `python -m src.pipeline` to populate tables."
            )
            return

        with tab1:
            _render_regional_risk_curve_tab(preds_df, metrics_df, selected_model, selected_region)
        with tab2:
            _render_regional_timeline_tab(
                preds_df, features_df, selected_model, selected_horizon, selected_region
            )
        with tab3:
            _render_regional_drivers_tab(
                contribs_df, selected_horizon, selected_model, selected_region
            )
        with tab4:
            _render_regional_validation_tab(preds_df, metrics_df, selected_horizon, selected_region)


if __name__ == "__main__":
    render_dashboard()
