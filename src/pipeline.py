"""CLI entrypoint for the Phase 2 global multi-region ETL & modeling pipeline."""

from __future__ import annotations

import logging
import sys
import time
from typing import Iterable

from dotenv import load_dotenv

from src.config import (
    ALL_REGIONS,
    BQ_DATASET,
    BQ_LOCATION,
    GCP_PROJECT_ID,
    HORIZONS,
    MODEL_LOGISTIC,
    MODEL_REGIONS,
)
from src.features import build_all_regions_features
from src.ingest import fetch_all_regions_raw
from src.modeling import run_all_regions_modeling
from src.warehouse import BigQueryWarehouse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    regions: Iterable[str] = MODEL_REGIONS,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = BQ_DATASET,
    location: str = BQ_LOCATION,
) -> dict[str, object]:
    """Execute multi-region ingest -> features -> walk-forward ML + GLOBAL -> BigQuery."""
    t0 = time.perf_counter()
    load_dotenv()
    region_list = tuple(regions)

    logger.info("Step 1/4: Ingesting FRED, BIS SDMX, and OECD SDMX series for %s...", region_list)
    raw_df = fetch_all_regions_raw(regions=region_list)
    logger.info(
        "Fetched %d raw observations across %d regions and %d distinct series.",
        len(raw_df),
        raw_df["region"].nunique(),
        raw_df[["region", "series_id"]].drop_duplicates().shape[0],
    )

    logger.info("Step 2/4: Engineering point-in-time monthly features & Option A targets...")
    features_df = build_all_regions_features(raw_df, regions=region_list)
    for reg in region_list:
        sub = features_df[features_df["region"] == reg]
        logger.info(
            "  [%s] %d monthly rows (%s to %s; labeled=%d)",
            reg,
            len(sub),
            sub["as_of_date"].min().strftime("%Y-%m-%d"),
            sub["as_of_date"].max().strftime("%Y-%m-%d"),
            int(sub["target_recession"].notna().sum()),
        )

    logger.info("Step 3/4: Running multi-region walk-forward ML & GLOBAL GDP-weighted aggregate...")
    predictions_df, metrics_df, contributions_df = run_all_regions_modeling(
        features_df, regions=region_list
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

    elapsed = time.perf_counter() - t0

    print(f"\n=== Pipeline Runtime: {elapsed:.1f}s ===")
    print("\n=== BigQuery Table Row Counts ===")
    for table_name, count in row_counts.items():
        print(f"  {project_id}.{dataset_id}.{table_name}: {count:,} rows")

    print(f"\n=== Latest Risk Curves ({MODEL_LOGISTIC}) Across All 6 Regions ===")
    header = f"{'Region':<8} {'Latest Date':<12} " + " ".join(f"{h:>5d}M" for h in HORIZONS)
    print(header)
    print("-" * len(header))
    for reg in ALL_REGIONS:
        sub = predictions_df[
            (predictions_df["region"] == reg) & (predictions_df["model_name"] == MODEL_LOGISTIC)
        ]
        if sub.empty:
            continue
        latest_dt = sub["as_of_date"].max()
        latest_row = sub[sub["as_of_date"] == latest_dt].set_index("horizon_months")["recession_prob"]
        vals = " ".join(
            f"{latest_row[h] * 100:5.1f}%" if h in latest_row.index else "    - "
            for h in HORIZONS
        )
        print(f"{reg:<8} {latest_dt.strftime('%Y-%m-%d'):<12} {vals}")

    print("\n=== Cross-Region Out-of-Sample Metrics (ROC-AUC / PR-AUC / Brier Score) ===")
    display_cols = [
        "region",
        "horizon_months",
        "model_name",
        "n_obs",
        "n_recessions",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "base_rate",
    ]
    ordered_metrics = metrics_df.copy()
    ordered_metrics["_reg_order"] = ordered_metrics["region"].map(
        {r: i for i, r in enumerate(MODEL_REGIONS)}
    )
    ordered_metrics = ordered_metrics.sort_values(
        ["_reg_order", "horizon_months", "model_name"]
    ).drop(columns=["_reg_order"])
    print(ordered_metrics[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return {
        "elapsed_seconds": elapsed,
        "row_counts": row_counts,
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
    }


def main() -> int:
    run_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
