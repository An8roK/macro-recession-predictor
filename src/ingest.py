"""Direct FRED REST API ingestion and month-end alignment (pandas 3.x safe)."""

from __future__ import annotations

import logging
import os
from typing import Iterable

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    DEFAULT_REGION,
    FRED_API_BASE_URL,
    FRED_SERIES_IDS,
    HISTORY_START_DATE,
)

logger = logging.getLogger(__name__)
# Suppress urllib3 retry URL logging so FRED api_key query parameters never appear in logs.
logging.getLogger("urllib3").setLevel(logging.ERROR)


def _build_retry_session(
    total_retries: int = 4,
    backoff_factor: float = 0.8,
) -> requests.Session:
    """Create a requests Session with exponential backoff for transient FRED errors."""
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_fred_api_key() -> str:
    """Load FRED_API_KEY from environment / .env without ever hardcoding credentials."""
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "FRED_API_KEY is missing. Set it in .env or the environment before running ingestion."
        )
    return api_key


def fetch_fred_series(
    series_id: str,
    api_key: str,
    start_date: str = HISTORY_START_DATE,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch a single FRED series via REST API and return [observation_date, value]."""
    own_session = session is None
    http = session or _build_retry_session()
    try:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        response = http.get(FRED_API_BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    finally:
        if own_session:
            http.close()

    observations = payload.get("observations", [])
    records: list[tuple[str, float]] = []
    for obs in observations:
        raw_val = obs.get("value")
        if raw_val is None or raw_val == ".":
            continue
        try:
            numeric_val = float(raw_val)
        except ValueError:
            continue
        records.append((obs["date"], numeric_val))

    if not records:
        raise RuntimeError(f"FRED returned no valid observations for series '{series_id}'.")

    df = pd.DataFrame(records, columns=["observation_date", "value"])
    df["observation_date"] = pd.to_datetime(df["observation_date"], format="%Y-%m-%d")
    df["value"] = df["value"].astype("float64")
    return df.sort_values("observation_date").reset_index(drop=True)


def fetch_all_fred_series(
    series_ids: Iterable[str] = FRED_SERIES_IDS,
    region: str = DEFAULT_REGION,
    start_date: str = HISTORY_START_DATE,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch all configured FRED series into the normalized raw_macro_series schema."""
    key = api_key or get_fred_api_key()
    ingested_at = pd.Timestamp.now(tz="UTC")
    frames: list[pd.DataFrame] = []

    with _build_retry_session() as session:
        for series_id in series_ids:
            logger.info("Fetching FRED series %s from %s...", series_id, start_date)
            series_df = fetch_fred_series(
                series_id=series_id,
                api_key=key,
                start_date=start_date,
                session=session,
            )
            series_df["region"] = region
            series_df["series_id"] = series_id
            series_df["ingested_at"] = ingested_at
            frames.append(series_df)

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = raw_df[["region", "series_id", "observation_date", "value", "ingested_at"]]
    raw_df = raw_df.sort_values(["series_id", "observation_date"]).reset_index(drop=True)
    return raw_df


def pivot_to_month_end(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long raw_macro_series to a regular month-end ('ME') DataFrame.

    Uses pandas 3.x-safe `resample("ME").last()` so daily/monthly FRED series
    align cleanly to calendar month-end timestamps.
    """
    if raw_df.empty:
        raise ValueError("Cannot pivot an empty raw_macro_series DataFrame.")

    work = raw_df.copy()
    work["observation_date"] = pd.to_datetime(work["observation_date"])
    wide = work.pivot_table(
        index="observation_date",
        columns="series_id",
        values="value",
        aggfunc="last",
    )
    wide = wide.sort_index()
    monthly = wide.resample("ME").last()
    monthly.index.name = "as_of_date"
    return monthly
