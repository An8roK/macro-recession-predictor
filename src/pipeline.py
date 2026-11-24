"""CLI entrypoint for the end-to-end macro recession ETL & modeling pipeline."""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from src.config import (
    BQ_DATASET,
    BQ_LOCATION,
    DEFAULT_REGION,
    GCP_PROJECT_ID,
    MODEL_LOGISTIC,
)
from src.features import build_monthly_features
from src.ingest import fetch_all_fred_series
from src.modeling import run_walk_forward_models
from src.warehouse import BigQueryWarehouse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    region: str = DEFAULT_REGION,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = BQ_DATASET,
    location: str = BQ_LOCATION,
) -> dict[str, object]:
    """Execute ingest -> features -> walk-forward modeling -> BigQuery batch load."""
    load_dotenv()

    logger.info("Step 1/4: Ingesting FRED macro series for region=%s...", region)
    raw_df = fetch_all_fred_series(region=region)
    logger.info(
        "Fetched %d raw observations across %d series.",
        len(raw_df),
        raw_df["series_id"].nunique(),
    )

    logger.info("Step 2/4: Engineering point-in-time monthly features...")
    features_df = build_monthly_features(raw_df, region=region)
    logger.info(
        "Built %d monthly feature rows (%s to %s).",
        len(features_df),
        features_df["as_of_date"].min().strftime("%Y-%m-%d"),
        features_df["as_of_date"].max().strftime("%Y-%m-%d"),
    )

    logger.info("Step 3/4: Running expanding-window walk-forward ML models...")
    predictions_df, metrics_df, contributions_df = run_walk_forward_models(
        features_df, region=region
    )
    logger.info(
        "Generated %d prediction rows, %d metric rows, %d contribution rows.",
        len(predictions_df),
        len(metrics_df),
        len(contributions_df),
    )

    logger.info(
        "Step 4/4: Batch-loading all 5 tables to BigQuery (%s.%s in %s)...",
        project_id,
        dataset_id,
        location,
    )
    warehouse = BigQueryWarehouse(
        project_id=project_id,
        dataset_id=dataset_id,
        location=location,
    )
    row_counts = warehouse.write_all_tables(
        raw_df=raw_df,
        features_df=features_df,
        predictions_df=predictions_df,
        metrics_df=metrics_df,
        contributions_df=contributions_df,
    )

    latest_date = predictions_df["as_of_date"].max()
    latest_curve = predictions_df[
        (predictions_df["as_of_date"] == latest_date)
        & (predictions_df["model_name"] == MODEL_LOGISTIC)
    ].sort_values("horizon_months")

    print("\n=== BigQuery Table Row Counts ===")
    for table_name, count in row_counts.items():
        print(f"  {project_id}.{dataset_id}.{table_name}: {count:,} rows")

    print(f"\n=== Latest Risk Curve ({MODEL_LOGISTIC}, as_of_date={latest_date.strftime('%Y-%m-%d')}) ===")
    for _, row in latest_curve.iterrows():
        print(f"  {int(row['horizon_months']):>2d}M: {row['recession_prob'] * 100:5.1f}%")

    print("\n=== Out-of-Sample Model Metrics ===")
    display_cols = [
        "horizon_months",
        "model_name",
        "n_obs",
        "n_recessions",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "base_rate",
    ]
    print(metrics_df[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return {
        "row_counts": row_counts,
        "latest_as_of_date": latest_date,
        "metrics_df": metrics_df,
    }


def main() -> int:
    run_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
