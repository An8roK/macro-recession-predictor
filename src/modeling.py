"""Expanding-window walk-forward multi-horizon modeling with strict embargo."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DEFAULT_REGION,
    FEATURE_COLUMNS,
    HORIZONS,
    MODEL_LOGISTIC,
    MODEL_RF,
    OOS_START_DATE,
)

logger = logging.getLogger(__name__)


def _month_index(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Convert dates to integer month ordinals (year * 12 + month - 1) for exact arithmetic."""
    dt = pd.to_datetime(dates)
    return (dt.dt.year.to_numpy(dtype=np.int64) * 12) + (dt.dt.month.to_numpy(dtype=np.int64) - 1)


def build_horizon_target(
    dates: pd.Series,
    target_recession: pd.Series,
    horizon_months: int,
) -> pd.Series:
    """Construct y(t, h) = USREC(t + h) using exact calendar-month lookup."""
    months = _month_index(dates)
    lookup: dict[int, int] = {}
    for m, val in zip(months, target_recession):
        if pd.notna(val):
            lookup[int(m)] = int(val)

    shifted = [lookup.get(int(m) + horizon_months, pd.NA) for m in months]
    return pd.Series(shifted, index=dates.index, dtype="Int64")


def get_embargoed_train_mask(
    dates: pd.Series,
    y_horizon: pd.Series,
    eval_date: pd.Timestamp,
    horizon_months: int,
    latest_labeled_date: pd.Timestamp | None = None,
) -> np.ndarray:
    """Return boolean mask of training rows strictly satisfying the walk-forward embargo.

    Rules:
    - Training row must have a known label (`y_horizon` not NA).
    - Strict leakage embargo: `t_train + h <= cutoff`, where `cutoff` is `eval_date`
      for historical OOS evaluation, or `min(eval_date, latest_labeled_date)` for live
      rows where `t_eval + h` lies in the future.
    - Never include `t_eval` itself in its own training set (`t_train < eval_date`),
      which matters specifically when `h == 0`.
    """
    train_months = _month_index(dates)
    eval_ts = pd.Timestamp(eval_date)
    eval_month = eval_ts.year * 12 + (eval_ts.month - 1)

    cutoff_month = eval_month
    if latest_labeled_date is not None:
        latest_ts = pd.Timestamp(latest_labeled_date)
        latest_month = latest_ts.year * 12 + (latest_ts.month - 1)
        cutoff_month = min(cutoff_month, latest_month)

    has_label = y_horizon.notna().to_numpy(dtype=bool)
    embargo_ok = (train_months + horizon_months) <= cutoff_month
    strictly_prior = train_months < eval_month
    return has_label & embargo_ok & strictly_prior


def _make_logistic_pipeline() -> Pipeline:
    """Primary calibrated linear model: StandardScaler + L2 Logistic Regression."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=1000)),
        ]
    )


def _make_random_forest() -> RandomForestClassifier:
    """Nonlinear benchmark model: shallow regularized Random Forest."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=10,
        random_state=42,
    )


def _fit_and_predict_prob(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[object | None, float]:
    """Fit estimator on training window and return (fitted_model, P(y=1))."""
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        # Degenerate single-class window fallback (should not occur in full US history)
        return None, float(unique_classes[0])

    estimator.fit(X_train, y_train)
    prob = float(estimator.predict_proba(X_eval)[0, 1])
    return estimator, prob


def _compute_logistic_contributions(
    fitted_pipeline: Pipeline,
    x_row: np.ndarray,
    as_of_date: pd.Timestamp,
    horizon_months: int,
    region: str,
    updated_at: pd.Timestamp,
) -> list[dict]:
    """Compute exact per-feature log-odds contributions `coef_i * z_i`."""
    scaler: StandardScaler = fitted_pipeline.named_steps["scaler"]
    clf: LogisticRegression = fitted_pipeline.named_steps["clf"]

    z_vals = scaler.transform(x_row.reshape(1, -1))[0]
    coefs = clf.coef_[0]
    contribs = coefs * z_vals

    rows: list[dict] = []
    for feat_name, raw_v, z_v, c_v, contrib_v in zip(
        FEATURE_COLUMNS, x_row, z_vals, coefs, contribs
    ):
        rows.append(
            {
                "region": region,
                "as_of_date": pd.Timestamp(as_of_date),
                "horizon_months": int(horizon_months),
                "feature_name": feat_name,
                "raw_value": float(raw_v),
                "standardized_value": float(z_v),
                "coef": float(c_v),
                "log_odds_contrib": float(contrib_v),
                "updated_at": updated_at,
            }
        )
    return rows


def run_walk_forward_models(
    features_df: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS,
    oos_start_date: str = OOS_START_DATE,
    region: str = DEFAULT_REGION,
    rf_refit_interval: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute expanding-window walk-forward training, live scoring, attribution, and OOS metrics.

    Returns:
        (predictions_df, metrics_df, contributions_df) matching BigQuery table schemas.
    """
    df = features_df.sort_values("as_of_date").reset_index(drop=True).copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    labeled_dates = df.loc[df["target_recession"].notna(), "as_of_date"]
    if labeled_dates.empty:
        raise ValueError("No labeled target_recession observations found in features_df.")
    latest_labeled_date = pd.Timestamp(labeled_dates.max())
    latest_as_of_date = pd.Timestamp(df["as_of_date"].max())
    oos_start_ts = pd.Timestamp(oos_start_date)

    X_all = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    dates_series = df["as_of_date"]
    updated_at = pd.Timestamp.now(tz="UTC")

    oos_indices = df.index[df["as_of_date"] >= oos_start_ts].to_list()
    if not oos_indices:
        raise ValueError(f"No rows found on or after OOS start date {oos_start_date}.")

    prediction_records: list[dict] = []
    contribution_records: list[dict] = []

    for h in horizons:
        logger.info("Running walk-forward models for horizon h=%dM...", h)
        y_h = build_horizon_target(dates_series, df["target_recession"], h)

        # Fit live models once on all labeled rows satisfying t_train + h <= latest_labeled_date
        live_mask = get_embargoed_train_mask(
            dates=dates_series,
            y_horizon=y_h,
            eval_date=latest_as_of_date + pd.offsets.MonthEnd(1),
            horizon_months=h,
            latest_labeled_date=latest_labeled_date,
        )
        X_live_train = X_all[live_mask]
        y_live_train = y_h.to_numpy()[live_mask].astype(np.int64)

        live_logistic = _make_logistic_pipeline()
        live_logistic, _ = _fit_and_predict_prob(
            live_logistic, X_live_train, y_live_train, X_all[-1:].copy()
        )
        live_rf = _make_random_forest()
        live_rf, _ = _fit_and_predict_prob(
            live_rf, X_live_train, y_live_train, X_all[-1:].copy()
        )

        # Track the exact logistic pipeline that scores latest_as_of_date for exact attribution
        latest_logistic_model: Pipeline | None = None
        rf_model_cache = None
        rf_steps_since_fit = rf_refit_interval

        for idx in oos_indices:
            eval_date = pd.Timestamp(df.at[idx, "as_of_date"])
            actual_val = y_h.iat[idx]
            is_live_row = pd.isna(actual_val)
            x_eval = X_all[idx : idx + 1]

            if is_live_row:
                # Live forecast row: actual_recession is NULL, use full embargoed live model
                prob_log = (
                    float(live_logistic.predict_proba(x_eval)[0, 1])
                    if live_logistic is not None
                    else float(np.mean(y_live_train))
                )
                prob_rf = (
                    float(live_rf.predict_proba(x_eval)[0, 1])
                    if live_rf is not None
                    else float(np.mean(y_live_train))
                )
                if eval_date == latest_as_of_date:
                    latest_logistic_model = live_logistic
            else:
                train_mask = get_embargoed_train_mask(
                    dates=dates_series,
                    y_horizon=y_h,
                    eval_date=eval_date,
                    horizon_months=h,
                    latest_labeled_date=latest_labeled_date,
                )
                X_train = X_all[train_mask]
                y_train = y_h.to_numpy()[train_mask].astype(np.int64)
                if len(y_train) == 0:
                    continue

                # 1. logistic_l2: refit every month on expanding window
                log_model, prob_log = _fit_and_predict_prob(
                    _make_logistic_pipeline(), X_train, y_train, x_eval
                )
                if eval_date == latest_as_of_date:
                    latest_logistic_model = log_model

                # 2. random_forest: refit every 12 months on expanding window
                if rf_model_cache is None or rf_steps_since_fit >= rf_refit_interval:
                    rf_model_cache, prob_rf = _fit_and_predict_prob(
                        _make_random_forest(), X_train, y_train, x_eval
                    )
                    rf_steps_since_fit = 1
                else:
                    prob_rf = (
                        float(rf_model_cache.predict_proba(x_eval)[0, 1])
                        if rf_model_cache is not None
                        else float(np.mean(y_train))
                    )
                    rf_steps_since_fit += 1

            actual_int = pd.NA if is_live_row else int(actual_val)
            prediction_records.append(
                {
                    "region": region,
                    "as_of_date": eval_date,
                    "horizon_months": int(h),
                    "model_name": MODEL_LOGISTIC,
                    "recession_prob": prob_log,
                    "is_oos": True,
                    "actual_recession": actual_int,
                    "updated_at": updated_at,
                }
            )
            prediction_records.append(
                {
                    "region": region,
                    "as_of_date": eval_date,
                    "horizon_months": int(h),
                    "model_name": MODEL_RF,
                    "recession_prob": prob_rf,
                    "is_oos": True,
                    "actual_recession": actual_int,
                    "updated_at": updated_at,
                }
            )

        # Compute exact driver attributions for the latest as_of_date from logistic_l2
        attribution_model = latest_logistic_model or live_logistic
        if attribution_model is not None:
            contribution_records.extend(
                _compute_logistic_contributions(
                    fitted_pipeline=attribution_model,
                    x_row=X_all[-1],
                    as_of_date=latest_as_of_date,
                    horizon_months=h,
                    region=region,
                    updated_at=updated_at,
                )
            )

    predictions_df = pd.DataFrame(prediction_records)
    predictions_df["actual_recession"] = predictions_df["actual_recession"].astype("Int64")
    predictions_df["horizon_months"] = predictions_df["horizon_months"].astype("int64")
    predictions_df["recession_prob"] = predictions_df["recession_prob"].astype("float64")
    predictions_df["is_oos"] = predictions_df["is_oos"].astype("bool")

    contributions_df = pd.DataFrame(contribution_records)
    if not contributions_df.empty:
        contributions_df["horizon_months"] = contributions_df["horizon_months"].astype("int64")

    metrics_df = compute_oos_metrics(predictions_df, updated_at=updated_at)
    return predictions_df, metrics_df, contributions_df


def compute_oos_metrics(
    predictions_df: pd.DataFrame,
    updated_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute OOS evaluation metrics strictly on `is_oos == True` and non-null actuals."""
    ts = updated_at or pd.Timestamp.now(tz="UTC")
    scored = predictions_df[
        predictions_df["is_oos"] & predictions_df["actual_recession"].notna()
    ].copy()

    metric_rows: list[dict] = []
    for (region, horizon, model_name), grp in scored.groupby(
        ["region", "horizon_months", "model_name"], sort=True
    ):
        y_true = grp["actual_recession"].astype(int).to_numpy()
        y_prob = grp["recession_prob"].astype(float).to_numpy()

        has_both_classes = len(np.unique(y_true)) == 2
        roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")
        pr_auc = (
            float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan")
        )
        brier = float(brier_score_loss(y_true, y_prob))

        metric_rows.append(
            {
                "region": str(region),
                "horizon_months": int(horizon),
                "model_name": str(model_name),
                "eval_start_date": pd.Timestamp(grp["as_of_date"].min()),
                "eval_end_date": pd.Timestamp(grp["as_of_date"].max()),
                "n_obs": int(len(y_true)),
                "n_recessions": int(np.sum(y_true)),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "base_rate": float(np.mean(y_true)),
                "updated_at": ts,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            ["region", "horizon_months", "model_name"]
        ).reset_index(drop=True)
    return metrics_df
