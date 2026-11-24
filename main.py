"""Stateless Streamlit dashboard reading precomputed recession forecasts from BigQuery."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import (
    MODEL_DISPLAY,
    build_calibration_chart,
    build_drivers_chart,
    build_historical_timeline_chart,
    build_risk_curve_chart,
)
from src.config import (
    DEFAULT_REGION,
    FEATURE_LABELS,
    HORIZONS,
    MODEL_LOGISTIC,
    MODEL_NAMES,
    MODEL_RF,
)
from src.warehouse import BigQueryWarehouse


@st.cache_data(ttl=3600, show_spinner=False)
def _load_regions() -> list[str]:
    return BigQueryWarehouse().read_available_regions()


@st.cache_data(ttl=3600, show_spinner="Loading BigQuery forecasts...")
def _load_dashboard_data(region: str) -> dict[str, pd.DataFrame]:
    wh = BigQueryWarehouse()
    return {
        "predictions": wh.read_predictions(region=region),
        "metrics": wh.read_metrics(region=region),
        "contributions": wh.read_contributions(region=region),
        "features": wh.read_monthly_features(region=region),
    }


def _render_sidebar(regions: list[str]) -> tuple[str, str, int]:
    st.sidebar.title("Macro Recession Predictor")
    st.sidebar.caption("Walk-forward macro regime risk monitor (BigQuery-backed)")

    default_region_idx = regions.index(DEFAULT_REGION) if DEFAULT_REGION in regions else 0
    selected_region = st.sidebar.selectbox("Region", options=regions, index=default_region_idx)

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


def _render_risk_curve_tab(
    preds_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selected_model: str,
    selected_region: str,
) -> None:
    latest_date = pd.Timestamp(preds_df["as_of_date"].max())
    latest_preds = preds_df[preds_df["as_of_date"] == latest_date].copy()
    updated_at = pd.Timestamp(latest_preds["updated_at"].max())

    st.info(
        f"**Region:** `{selected_region}`  |  "
        f"**Latest As-Of Date:** `{latest_date.strftime('%Y-%m-%d')}`  |  "
        f"**Model:** `{MODEL_DISPLAY.get(selected_model, selected_model)}`  |  "
        f"**Warehouse Updated:** `{updated_at.strftime('%Y-%m-%d %H:%M UTC')}`"
    )

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
    )
    st.plotly_chart(fig, width="stretch")


def _render_timeline_tab(
    preds_df: pd.DataFrame,
    features_df: pd.DataFrame,
    selected_model: str,
    selected_horizon: int,
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
            compare_models=compare_models,
            overlay_horizons=None,
        )

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "All probabilities shown from 1975 onward are strictly out-of-sample expanding-window "
        "forecasts with horizon embargo (`t_train + h <= t_eval`). Shaded bars mark NBER recessions."
    )


def _render_drivers_tab(
    contribs_df: pd.DataFrame,
    selected_horizon: int,
    selected_model: str,
) -> None:
    if selected_model != MODEL_LOGISTIC:
        st.caption(
            "Note: Exact linear log-odds attributions (`coef * z-score`) are derived from the "
            "primary calibrated `logistic_l2` model for interpretability."
        )

    fig = build_drivers_chart(contribs_df=contribs_df, horizon_months=selected_horizon)
    st.plotly_chart(fig, width="stretch")

    sub = contribs_df[contribs_df["horizon_months"] == selected_horizon].copy()
    sub["Indicator"] = sub["feature_name"].map(
        lambda k: FEATURE_LABELS.get(k, {}).get("label", k)
    )
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

    st.subheader(f"Latest Macro Signal Readings ({selected_horizon}M Horizon)")
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


def _render_validation_tab(
    preds_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selected_horizon: int,
) -> None:
    st.subheader("Out-of-Sample Model Validation (1975–Present)")
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
            "n_recessions": "Recession Months",
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
        page_title="Macro Recession Predictor",
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
    data = _load_dashboard_data(selected_region)

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

    st.title("Macro Recession Risk Monitor")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Recession Risk Curve",
            "Historical Timeline",
            "Signal Drivers",
            "Model Validation",
        ]
    )

    with tab1:
        _render_risk_curve_tab(preds_df, metrics_df, selected_model, selected_region)
    with tab2:
        _render_timeline_tab(preds_df, features_df, selected_model, selected_horizon)
    with tab3:
        _render_drivers_tab(contribs_df, selected_horizon, selected_model)
    with tab4:
        _render_validation_tab(preds_df, metrics_df, selected_horizon)


if __name__ == "__main__":
    render_dashboard()
