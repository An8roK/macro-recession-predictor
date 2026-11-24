"""Plotly figure builders for all 4 dashboard tabs (dark/light theme compatible)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import FEATURE_LABELS, MODEL_LOGISTIC, MODEL_RF

MODEL_DISPLAY: dict[str, str] = {
    MODEL_LOGISTIC: "Logistic L2 (Primary)",
    MODEL_RF: "Random Forest (Benchmark)",
}

MODEL_COLORS: dict[str, str] = {
    MODEL_LOGISTIC: "#2563EB",
    MODEL_RF: "#F59E0B",
}

HORIZON_PALETTE: dict[int, str] = {
    0: "#10B981",
    3: "#06B6D4",
    6: "#2563EB",
    12: "#8B5CF6",
    18: "#EC4899",
}


def _apply_institutional_layout(fig: go.Figure, title: str, yaxis_title: str = "") -> go.Figure:
    """Apply transparent, theme-agnostic institutional styling for dark & light Streamlit modes."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=48, r=24, t=56, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(title=yaxis_title, showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


def _extract_recession_spans(features_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert binary USREC indicator series into contiguous (start, end) shading intervals."""
    if features_df.empty or "target_recession" not in features_df.columns:
        return []

    work = features_df[["as_of_date", "target_recession"]].dropna().sort_values("as_of_date")
    if work.empty:
        return []

    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_recession = False
    start_dt: pd.Timestamp | None = None
    prev_dt: pd.Timestamp | None = None

    for dt, val in zip(work["as_of_date"], work["target_recession"].astype(int)):
        ts = pd.Timestamp(dt)
        if val == 1 and not in_recession:
            start_dt = ts
            in_recession = True
        elif val == 0 and in_recession and start_dt is not None and prev_dt is not None:
            spans.append((start_dt, prev_dt + pd.offsets.MonthEnd(1)))
            in_recession = False
            start_dt = None
        prev_dt = ts

    if in_recession and start_dt is not None and prev_dt is not None:
        spans.append((start_dt, prev_dt + pd.offsets.MonthEnd(1)))
    return spans


def build_risk_curve_chart(
    latest_preds: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selected_model: str,
) -> go.Figure:
    """Build Tab 1 term-structure bar + line chart across 0M/3M/6M/12M/18M horizons."""
    fig = go.Figure()

    primary_df = latest_preds[latest_preds["model_name"] == selected_model].sort_values(
        "horizon_months"
    )
    other_model = MODEL_RF if selected_model == MODEL_LOGISTIC else MODEL_LOGISTIC
    other_df = latest_preds[latest_preds["model_name"] == other_model].sort_values("horizon_months")

    horizon_labels = [f"{int(h)}M" for h in primary_df["horizon_months"]]

    fig.add_trace(
        go.Bar(
            x=horizon_labels,
            y=primary_df["recession_prob"] * 100.0,
            name=MODEL_DISPLAY.get(selected_model, selected_model),
            marker_color=MODEL_COLORS.get(selected_model, "#2563EB"),
            opacity=0.85,
            text=[f"{p * 100:.1f}%" for p in primary_df["recession_prob"]],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )

    if not other_df.empty:
        other_labels = [f"{int(h)}M" for h in other_df["horizon_months"]]
        fig.add_trace(
            go.Scatter(
                x=other_labels,
                y=other_df["recession_prob"] * 100.0,
                name=MODEL_DISPLAY.get(other_model, other_model),
                mode="lines+markers",
                line=dict(color=MODEL_COLORS.get(other_model, "#F59E0B"), width=2.5),
                marker=dict(size=8),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        )

    base_rates = (
        metrics_df[metrics_df["model_name"] == selected_model]
        .sort_values("horizon_months")
    )
    if not base_rates.empty:
        fig.add_trace(
            go.Scatter(
                x=[f"{int(h)}M" for h in base_rates["horizon_months"]],
                y=base_rates["base_rate"] * 100.0,
                name="Historical Base Rate",
                mode="lines",
                line=dict(color="rgba(148,163,184,0.9)", width=2, dash="dot"),
                hovertemplate="%{x} base rate: %{y:.1f}%<extra></extra>",
            )
        )

    _apply_institutional_layout(
        fig,
        title="Recession Probability Term Structure",
        yaxis_title="Probability (%)",
    )
    fig.update_yaxes(range=[0, max(100.0, float(latest_preds["recession_prob"].max() * 115.0))])
    return fig


def build_historical_timeline_chart(
    preds_df: pd.DataFrame,
    features_df: pd.DataFrame,
    selected_model: str,
    selected_horizon: int,
    compare_models: bool = True,
    overlay_horizons: list[int] | None = None,
) -> go.Figure:
    """Build Tab 2 OOS recession probability timeline with NBER recession shading."""
    fig = go.Figure()

    oos_df = preds_df[preds_df["is_oos"]].copy()
    if overlay_horizons:
        for h in sorted(overlay_horizons):
            sub = oos_df[
                (oos_df["model_name"] == selected_model) & (oos_df["horizon_months"] == h)
            ].sort_values("as_of_date")
            fig.add_trace(
                go.Scatter(
                    x=sub["as_of_date"],
                    y=sub["recession_prob"] * 100.0,
                    name=f"{MODEL_DISPLAY.get(selected_model, selected_model)} ({h}M)",
                    mode="lines",
                    line=dict(color=HORIZON_PALETTE.get(h, "#2563EB"), width=2),
                    hovertemplate="%{x|%Y-%m}: %{y:.1f}%<extra></extra>",
                )
            )
    else:
        models_to_plot = [selected_model]
        if compare_models:
            other = MODEL_RF if selected_model == MODEL_LOGISTIC else MODEL_LOGISTIC
            models_to_plot.append(other)

        for model_name in models_to_plot:
            sub = oos_df[
                (oos_df["model_name"] == model_name)
                & (oos_df["horizon_months"] == selected_horizon)
            ].sort_values("as_of_date")
            is_primary = model_name == selected_model
            fig.add_trace(
                go.Scatter(
                    x=sub["as_of_date"],
                    y=sub["recession_prob"] * 100.0,
                    name=f"{MODEL_DISPLAY.get(model_name, model_name)} ({selected_horizon}M)",
                    mode="lines",
                    line=dict(
                        color=MODEL_COLORS.get(model_name, "#2563EB"),
                        width=2.4 if is_primary else 1.8,
                        dash="solid" if is_primary else "dash",
                    ),
                    hovertemplate="%{x|%Y-%m}: %{y:.1f}%<extra></extra>",
                )
            )

    min_date = oos_df["as_of_date"].min()
    max_date = oos_df["as_of_date"].max()
    recession_spans = _extract_recession_spans(
        features_df[features_df["as_of_date"] >= min_date]
    )
    for start_dt, end_dt in recession_spans:
        fig.add_vrect(
            x0=max(start_dt, min_date),
            x1=min(end_dt, max_date),
            fillcolor="rgba(148, 163, 184, 0.25)",
            line_width=0,
            layer="below",
        )

    _apply_institutional_layout(
        fig,
        title="Out-of-Sample Recession Probability Timeline (Gray = NBER Recessions)",
        yaxis_title="Recession Probability (%)",
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def build_drivers_chart(contribs_df: pd.DataFrame, horizon_months: int) -> go.Figure:
    """Build Tab 3 horizontal diverging bar chart of log-odds contributions."""
    sub = contribs_df[contribs_df["horizon_months"] == horizon_months].copy()
    sub["abs_contrib"] = sub["log_odds_contrib"].abs()
    sub = sub.sort_values("abs_contrib", ascending=True)
    sub["display_name"] = sub["feature_name"].map(
        lambda k: FEATURE_LABELS.get(k, {}).get("label", k)
    )

    colors = [
        "#EF4444" if val > 0 else "#14B8A6" for val in sub["log_odds_contrib"]
    ]

    fig = go.Figure(
        go.Bar(
            x=sub["log_odds_contrib"],
            y=sub["display_name"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in sub["log_odds_contrib"]],
            textposition="outside",
            customdata=np.stack(
                [sub["raw_value"], sub["standardized_value"], sub["coef"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Log-Odds Contrib: %{x:+.3f}<br>"
                "Raw Value: %{customdata[0]:.2f}<br>"
                "Z-Score: %{customdata[1]:+.2f}<br>"
                "Coef: %{customdata[2]:+.3f}<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_width=1.5, line_color="rgba(148,163,184,0.8)")
    _apply_institutional_layout(
        fig,
        title=f"Log-Odds Risk Drivers — {horizon_months}M Horizon (Logistic L2)",
        yaxis_title="",
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Log-Odds Contribution (Red = Higher Risk, Teal = Lower Risk)")
    return fig


def build_calibration_chart(
    preds_df: pd.DataFrame,
    horizon_months: int,
    n_bins: int = 8,
) -> go.Figure:
    """Build Tab 4 binned calibration (reliability) curve comparing both models."""
    fig = go.Figure()

    # Perfect calibration reference diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="rgba(148,163,184,0.8)", width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )

    scored = preds_df[
        preds_df["is_oos"]
        & preds_df["actual_recession"].notna()
        & (preds_df["horizon_months"] == horizon_months)
    ].copy()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for model_name in (MODEL_LOGISTIC, MODEL_RF):
        sub = scored[scored["model_name"] == model_name].copy()
        if sub.empty:
            continue
        sub["bin"] = pd.cut(
            sub["recession_prob"].clip(0.0, 1.0),
            bins=bin_edges,
            include_lowest=True,
        )
        grouped = (
            sub.groupby("bin", observed=True)
            .agg(
                mean_pred=("recession_prob", "mean"),
                obs_rate=("actual_recession", "mean"),
                count=("actual_recession", "count"),
            )
            .reset_index()
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["mean_pred"] * 100.0,
                y=grouped["obs_rate"] * 100.0,
                mode="lines+markers",
                name=MODEL_DISPLAY.get(model_name, model_name),
                line=dict(color=MODEL_COLORS.get(model_name, "#2563EB"), width=2.4),
                marker=dict(size=8),
                customdata=grouped["count"],
                hovertemplate=(
                    "Predicted: %{x:.1f}%<br>"
                    "Realized: %{y:.1f}%<br>"
                    "Months in bin: %{customdata}<extra></extra>"
                ),
            )
        )

    _apply_institutional_layout(
        fig,
        title=f"OOS Calibration (Reliability) Curve — {horizon_months}M Horizon",
        yaxis_title="Observed Recession Frequency (%)",
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Mean Predicted Probability (%)", range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    return fig
