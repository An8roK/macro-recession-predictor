"""Plotly figure builders for all 4 dashboard tabs across GLOBAL and 5 modeled economies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import (
    GLOBAL_REGION,
    MODEL_LOGISTIC,
    MODEL_REGIONS,
    MODEL_RF,
    REGION_SPECS,
    get_feature_label,
)

MODEL_DISPLAY: dict[str, str] = {
    MODEL_LOGISTIC: "Logistic L2 (Primary)",
    MODEL_RF: "Random Forest (Benchmark)",
}

MODEL_COLORS: dict[str, str] = {
    MODEL_LOGISTIC: "#2563EB",
    MODEL_RF: "#F59E0B",
}

REGION_COLORS: dict[str, str] = {
    "GLOBAL": "#111827",
    "US": "#2563EB",
    "EU": "#10B981",
    "JP": "#8B5CF6",
    "CN": "#EF4444",
    "IN": "#F59E0B",
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
    """Convert binary target_recession series into contiguous (start, end) shading intervals."""
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
    region: str = "US",
) -> go.Figure:
    """Build Tab 1 term-structure bar + line chart for an individual economy."""
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

    region_name = REGION_SPECS[region].name if region in REGION_SPECS else region
    _apply_institutional_layout(
        fig,
        title=f"{region_name} — Recession Probability Term Structure",
        yaxis_title="Probability (%)",
    )
    fig.update_yaxes(range=[0, max(100.0, float(latest_preds["recession_prob"].max() * 115.0))])
    return fig


def build_global_risk_curve_chart(
    all_preds: pd.DataFrame,
    selected_model: str,
) -> go.Figure:
    """Build Tab 1 multi-region term-structure comparison chart for GLOBAL view."""
    fig = go.Figure()
    model_preds = all_preds[all_preds["model_name"] == selected_model].copy()

    # Plot constituent economies first, then bold GLOBAL composite on top
    for reg in MODEL_REGIONS:
        sub = model_preds[model_preds["region"] == reg]
        if sub.empty:
            continue
        latest_dt = sub["as_of_date"].max()
        cur = sub[sub["as_of_date"] == latest_dt].sort_values("horizon_months")
        fig.add_trace(
            go.Scatter(
                x=[f"{int(h)}M" for h in cur["horizon_months"]],
                y=cur["recession_prob"] * 100.0,
                name=f"{reg} ({REGION_SPECS[reg].name})",
                mode="lines+markers",
                line=dict(color=REGION_COLORS.get(reg, "#64748B"), width=2, dash="dot"),
                marker=dict(size=7),
                hovertemplate=f"{reg} %{{x}}: %{{y:.1f}}%<extra></extra>",
            )
        )

    glob = model_preds[model_preds["region"] == GLOBAL_REGION]
    if not glob.empty:
        latest_dt = glob["as_of_date"].max()
        cur_g = glob[glob["as_of_date"] == latest_dt].sort_values("horizon_months")
        fig.add_trace(
            go.Scatter(
                x=[f"{int(h)}M" for h in cur_g["horizon_months"]],
                y=cur_g["recession_prob"] * 100.0,
                name="GLOBAL (GDP-Weighted)",
                mode="lines+markers+text",
                line=dict(color="#06B6D4", width=3.5),
                marker=dict(size=10, symbol="diamond"),
                text=[f"{p * 100:.1f}%" for p in cur_g["recession_prob"]],
                textposition="top center",
                hovertemplate="GLOBAL %{x}: %{y:.1f}%<extra></extra>",
            )
        )

    _apply_institutional_layout(
        fig,
        title=f"Global & Regional Risk Term Structure ({MODEL_DISPLAY.get(selected_model, selected_model)})",
        yaxis_title="Recession / Downturn Probability (%)",
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def build_historical_timeline_chart(
    preds_df: pd.DataFrame,
    features_df: pd.DataFrame,
    selected_model: str,
    selected_horizon: int,
    region: str = "US",
    compare_models: bool = True,
    overlay_horizons: list[int] | None = None,
) -> go.Figure:
    """Build Tab 2 OOS probability timeline with region-specific regime shading."""
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

    shade_label = REGION_SPECS[region].regime_badge if region in REGION_SPECS else "Recessions"
    _apply_institutional_layout(
        fig,
        title=f"Out-of-Sample Risk Timeline (Gray = {shade_label})",
        yaxis_title="Probability (%)",
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def build_global_timeline_chart(
    all_preds: pd.DataFrame,
    us_features_df: pd.DataFrame,
    selected_model: str,
    selected_horizon: int,
    overlay_regions: list[str] | None = None,
) -> go.Figure:
    """Build Tab 2 historical GDP-weighted GLOBAL curve with optional regional overlays."""
    fig = go.Figure()
    sub = all_preds[
        all_preds["is_oos"]
        & (all_preds["model_name"] == selected_model)
        & (all_preds["horizon_months"] == selected_horizon)
    ].copy()

    for reg in overlay_regions or []:
        r_df = sub[sub["region"] == reg].sort_values("as_of_date")
        if r_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=r_df["as_of_date"],
                y=r_df["recession_prob"] * 100.0,
                name=f"{reg} ({selected_horizon}M)",
                mode="lines",
                line=dict(color=REGION_COLORS.get(reg, "#64748B"), width=1.6, dash="dot"),
                opacity=0.85,
                hovertemplate=f"{reg} %{{x|%Y-%m}}: %{{y:.1f}}%<extra></extra>",
            )
        )

    g_df = sub[sub["region"] == GLOBAL_REGION].sort_values("as_of_date")
    if not g_df.empty:
        fig.add_trace(
            go.Scatter(
                x=g_df["as_of_date"],
                y=g_df["recession_prob"] * 100.0,
                name=f"GLOBAL GDP-Weighted ({selected_horizon}M)",
                mode="lines",
                line=dict(color="#06B6D4", width=2.8),
                hovertemplate="GLOBAL %{x|%Y-%m}: %{y:.1f}%<extra></extra>",
            )
        )

    if not g_df.empty and not us_features_df.empty:
        min_date = g_df["as_of_date"].min()
        max_date = g_df["as_of_date"].max()
        for start_dt, end_dt in _extract_recession_spans(
            us_features_df[us_features_df["as_of_date"] >= min_date]
        ):
            fig.add_vrect(
                x0=max(start_dt, min_date),
                x1=min(end_dt, max_date),
                fillcolor="rgba(148, 163, 184, 0.22)",
                line_width=0,
                layer="below",
            )

    _apply_institutional_layout(
        fig,
        title=f"Global GDP-Weighted Risk Timeline — {selected_horizon}M Horizon (Gray = US NBER Reference)",
        yaxis_title="Probability (%)",
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def build_drivers_chart(
    contribs_df: pd.DataFrame,
    horizon_months: int,
    region: str = "US",
) -> go.Figure:
    """Build Tab 3 horizontal diverging bar chart of log-odds contributions for an economy."""
    sub = contribs_df[contribs_df["horizon_months"] == horizon_months].copy()
    sub["abs_contrib"] = sub["log_odds_contrib"].abs()
    sub = sub.sort_values("abs_contrib", ascending=True)
    sub["display_name"] = sub["feature_name"].map(lambda k: get_feature_label(k, region))

    colors = ["#EF4444" if val > 0 else "#14B8A6" for val in sub["log_odds_contrib"]]

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
        title=f"{region} Log-Odds Risk Drivers — {horizon_months}M Horizon (Logistic L2)",
        yaxis_title="",
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Log-Odds Contribution (Red = Higher Risk, Teal = Lower Risk)")
    return fig


def build_global_drivers_chart(contribs_df: pd.DataFrame, horizon_months: int) -> go.Figure:
    """Build Tab 3 horizontal bar chart of weighted regional contributions to GLOBAL risk."""
    sub = contribs_df[contribs_df["horizon_months"] == horizon_months].copy()
    sub = sub.sort_values("log_odds_contrib", ascending=True)
    sub["display_name"] = sub["feature_name"].map(lambda k: get_feature_label(k, GLOBAL_REGION))
    sub["contrib_pp"] = sub["log_odds_contrib"] * 100.0
    colors = [REGION_COLORS.get(code, "#2563EB") for code in sub["feature_name"]]

    fig = go.Figure(
        go.Bar(
            x=sub["contrib_pp"],
            y=sub["display_name"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f} pp" for v in sub["contrib_pp"]],
            textposition="outside",
            customdata=np.stack(
                [sub["coef"] * 100.0, sub["raw_value"] * 100.0], axis=-1
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Weighted Contribution: %{x:.2f} pp<br>"
                "GDP Weight: %{customdata[0]:.1f}%<br>"
                "Regional Probability: %{customdata[1]:.1f}%<extra></extra>"
            ),
        )
    )

    _apply_institutional_layout(
        fig,
        title=f"Regional Contributions to Global Composite Risk — {horizon_months}M Horizon",
        yaxis_title="",
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Weighted Probability Contribution (percentage points)")
    return fig


def build_calibration_chart(
    preds_df: pd.DataFrame,
    horizon_months: int,
    n_bins: int = 8,
) -> go.Figure:
    """Build Tab 4 binned calibration (reliability) curve comparing both models."""
    fig = go.Figure()

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
        yaxis_title="Observed Frequency (%)",
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Mean Predicted Probability (%)", range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    return fig


def build_cross_region_validation_chart(
    all_metrics: pd.DataFrame,
    selected_model: str,
) -> go.Figure:
    """Build Tab 4 cross-region OOS ROC-AUC comparison chart across all 5 modeled economies."""
    fig = go.Figure()
    sub = all_metrics[all_metrics["model_name"] == selected_model].copy()

    for reg in MODEL_REGIONS:
        r_df = sub[sub["region"] == reg].sort_values("horizon_months")
        if r_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=[f"{int(h)}M" for h in r_df["horizon_months"]],
                y=r_df["roc_auc"],
                name=f"{reg} ({REGION_SPECS[reg].name})",
                mode="lines+markers",
                line=dict(color=REGION_COLORS.get(reg, "#2563EB"), width=2.4),
                marker=dict(size=8),
                hovertemplate=f"{reg} %{{x}} ROC-AUC: %{{y:.3f}}<extra></extra>",
            )
        )

    fig.add_hline(y=0.5, line_width=1.2, line_dash="dot", line_color="rgba(148,163,184,0.8)")
    _apply_institutional_layout(
        fig,
        title=f"Cross-Region OOS ROC-AUC by Horizon ({MODEL_DISPLAY.get(selected_model, selected_model)})",
        yaxis_title="ROC-AUC",
    )
    fig.update_yaxes(range=[0.4, 1.0])
    return fig
