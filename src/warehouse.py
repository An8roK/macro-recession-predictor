"""BigQuery warehouse management: free batch loads (WRITE_TRUNCATE) and multi-region queries."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
from google.cloud import bigquery

from src.config import (
    ALL_FEATURE_COLUMNS,
    ALL_REGIONS,
    ALL_TABLES,
    BQ_DATASET,
    BQ_LOCATION,
    DEFAULT_REGION,
    GCP_PROJECT_ID,
    TABLE_CONTRIBUTIONS,
    TABLE_METRICS,
    TABLE_MONTHLY_FEATURES,
    TABLE_PREDICTIONS,
    TABLE_RAW_SERIES,
)

logger = logging.getLogger(__name__)

# Explicit BigQuery schemas matching docs/ARCHITECTURE.md Section 6.
# All 21 feature columns in monthly_features are NULLABLE so US and international rows coexist cleanly.
TABLE_SCHEMAS: dict[str, list[bigquery.SchemaField]] = {
    TABLE_RAW_SERIES: [
        bigquery.SchemaField("region", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("series_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("observation_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("value", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED"),
    ],
    TABLE_MONTHLY_FEATURES: [
        bigquery.SchemaField("region", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("as_of_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("target_recession", "INT64", mode="NULLABLE"),
        *[bigquery.SchemaField(col, "FLOAT64", mode="NULLABLE") for col in ALL_FEATURE_COLUMNS],
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
    ],
    TABLE_PREDICTIONS: [
        bigquery.SchemaField("region", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("as_of_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("horizon_months", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("recession_prob", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("is_oos", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("actual_recession", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
    ],
    TABLE_METRICS: [
        bigquery.SchemaField("region", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("horizon_months", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("eval_start_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("eval_end_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("n_obs", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("n_recessions", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("roc_auc", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("pr_auc", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("brier_score", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("base_rate", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
    ],
    TABLE_CONTRIBUTIONS: [
        bigquery.SchemaField("region", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("as_of_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("horizon_months", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("feature_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("raw_value", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("standardized_value", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("coef", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("log_odds_contrib", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
    ],
}


def _prepare_dataframe_for_bq(
    df: pd.DataFrame,
    schema: list[bigquery.SchemaField],
) -> pd.DataFrame:
    """Coerce DATE, TIMESTAMP, INT64, and FLOAT64 columns cleanly for pyarrow batch loads."""
    out = df[[field.name for field in schema]].copy()
    for field in schema:
        col = field.name
        if field.field_type == "DATE":
            out[col] = pd.to_datetime(out[col]).dt.date
        elif field.field_type == "TIMESTAMP":
            out[col] = pd.to_datetime(out[col], utc=True)
        elif field.field_type == "INT64":
            out[col] = out[col].astype("Int64")
        elif field.field_type == "FLOAT64":
            out[col] = out[col].astype("float64")
        elif field.field_type == "BOOL":
            out[col] = out[col].astype("bool")
        elif field.field_type == "STRING":
            out[col] = out[col].astype(str)
    return out


class BigQueryWarehouse:
    """Thin BigQuery wrapper enforcing Always-Free batch loads (never streaming inserts)."""

    def __init__(
        self,
        project_id: str = GCP_PROJECT_ID,
        dataset_id: str = BQ_DATASET,
        location: str = BQ_LOCATION,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.location = location
        self.client = bigquery.Client(project=project_id, location=location)

    @property
    def full_dataset_id(self) -> str:
        return f"{self.project_id}.{self.dataset_id}"

    def table_ref(self, table_name: str) -> str:
        return f"{self.project_id}.{self.dataset_id}.{table_name}"

    def ensure_dataset(self) -> bigquery.Dataset:
        """Create the BigQuery dataset in us-west1 if it does not already exist."""
        dataset = bigquery.Dataset(self.full_dataset_id)
        dataset.location = self.location
        created = self.client.create_dataset(dataset, exists_ok=True)
        logger.info("Ensured BigQuery dataset %s (%s).", self.full_dataset_id, self.location)
        return created

    def load_table(self, table_name: str, df: pd.DataFrame) -> int:
        """Overwrite a BigQuery table via a free batch load job (WRITE_TRUNCATE)."""
        if table_name not in TABLE_SCHEMAS:
            raise ValueError(f"Unknown BigQuery table name: {table_name}")

        schema = TABLE_SCHEMAS[table_name]
        prepared_df = _prepare_dataframe_for_bq(df, schema)
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        destination = self.table_ref(table_name)
        load_job = self.client.load_table_from_dataframe(
            prepared_df,
            destination,
            job_config=job_config,
            location=self.location,
        )
        load_job.result()
        table = self.client.get_table(destination)
        logger.info("Loaded %d rows into %s (WRITE_TRUNCATE).", table.num_rows, destination)
        return int(table.num_rows)

    def write_all_tables(
        self,
        raw_df: pd.DataFrame,
        features_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        contributions_df: pd.DataFrame,
    ) -> dict[str, int]:
        """Ensure dataset exists and batch-load all 5 tables with WRITE_TRUNCATE."""
        self.ensure_dataset()
        payloads = {
            TABLE_RAW_SERIES: raw_df,
            TABLE_MONTHLY_FEATURES: features_df,
            TABLE_PREDICTIONS: predictions_df,
            TABLE_METRICS: metrics_df,
            TABLE_CONTRIBUTIONS: contributions_df,
        }
        return {name: self.load_table(name, frame) for name, frame in payloads.items()}

    def get_table_row_counts(self, tables: Iterable[str] = ALL_TABLES) -> dict[str, int]:
        """Fetch table row counts from BigQuery table metadata (zero query bytes billed)."""
        counts: dict[str, int] = {}
        for name in tables:
            table = self.client.get_table(self.table_ref(name))
            counts[name] = int(table.num_rows)
        return counts

    def _query_by_region(self, sql: str, region: str) -> pd.DataFrame:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("region", "STRING", region)]
        )
        return self.client.query(sql, job_config=job_config, location=self.location).to_dataframe()

    def read_available_regions(self) -> list[str]:
        """Return distinct regions populated in recession_predictions in canonical UI order."""
        sql = f"""
            SELECT DISTINCT region
            FROM `{self.table_ref(TABLE_PREDICTIONS)}`
        """
        df = self.client.query(sql, location=self.location).to_dataframe()
        if df.empty:
            return [DEFAULT_REGION]
        found = set(df["region"].tolist())
        ordered = [r for r in ALL_REGIONS if r in found]
        extras = sorted(found - set(ALL_REGIONS))
        return ordered + extras

    def read_predictions(self, region: str = DEFAULT_REGION) -> pd.DataFrame:
        """Read precomputed recession predictions for a single region."""
        sql = f"""
            SELECT region, as_of_date, horizon_months, model_name,
                   recession_prob, is_oos, actual_recession, updated_at
            FROM `{self.table_ref(TABLE_PREDICTIONS)}`
            WHERE region = @region
            ORDER BY as_of_date, horizon_months, model_name
        """
        df = self._query_by_region(sql, region)
        if not df.empty:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        return df

    def read_all_predictions(self) -> pd.DataFrame:
        """Read precomputed recession predictions across all regions (for GLOBAL views)."""
        sql = f"""
            SELECT region, as_of_date, horizon_months, model_name,
                   recession_prob, is_oos, actual_recession, updated_at
            FROM `{self.table_ref(TABLE_PREDICTIONS)}`
            ORDER BY region, as_of_date, horizon_months, model_name
        """
        df = self.client.query(sql, location=self.location).to_dataframe()
        if not df.empty:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        return df

    def read_metrics(self, region: str = DEFAULT_REGION) -> pd.DataFrame:
        """Read OOS validation metrics for a single region."""
        sql = f"""
            SELECT region, horizon_months, model_name, eval_start_date, eval_end_date,
                   n_obs, n_recessions, roc_auc, pr_auc, brier_score, base_rate, updated_at
            FROM `{self.table_ref(TABLE_METRICS)}`
            WHERE region = @region
            ORDER BY horizon_months, model_name
        """
        df = self._query_by_region(sql, region)
        if not df.empty:
            df["eval_start_date"] = pd.to_datetime(df["eval_start_date"])
            df["eval_end_date"] = pd.to_datetime(df["eval_end_date"])
        return df

    def read_all_metrics(self) -> pd.DataFrame:
        """Read OOS validation metrics across all modeled economies."""
        sql = f"""
            SELECT region, horizon_months, model_name, eval_start_date, eval_end_date,
                   n_obs, n_recessions, roc_auc, pr_auc, brier_score, base_rate, updated_at
            FROM `{self.table_ref(TABLE_METRICS)}`
            ORDER BY region, horizon_months, model_name
        """
        df = self.client.query(sql, location=self.location).to_dataframe()
        if not df.empty:
            df["eval_start_date"] = pd.to_datetime(df["eval_start_date"])
            df["eval_end_date"] = pd.to_datetime(df["eval_end_date"])
        return df

    def read_contributions(self, region: str = DEFAULT_REGION) -> pd.DataFrame:
        """Read latest driver attributions (or GLOBAL regional breakdown) for a region."""
        sql = f"""
            SELECT region, as_of_date, horizon_months, feature_name,
                   raw_value, standardized_value, coef, log_odds_contrib, updated_at
            FROM `{self.table_ref(TABLE_CONTRIBUTIONS)}`
            WHERE region = @region
            ORDER BY horizon_months, ABS(log_odds_contrib) DESC
        """
        df = self._query_by_region(sql, region)
        if not df.empty:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        return df

    def read_monthly_features(self, region: str = DEFAULT_REGION) -> pd.DataFrame:
        """Read monthly point-in-time features and target indicator for a region."""
        feature_cols_sql = ", ".join(ALL_FEATURE_COLUMNS)
        sql = f"""
            SELECT region, as_of_date, target_recession, {feature_cols_sql}, updated_at
            FROM `{self.table_ref(TABLE_MONTHLY_FEATURES)}`
            WHERE region = @region
            ORDER BY as_of_date
        """
        df = self._query_by_region(sql, region)
        if not df.empty:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        return df
