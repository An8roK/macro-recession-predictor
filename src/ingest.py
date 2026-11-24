"""Multi-source REST/CSV ingestion for FRED, BIS SDMX, and OECD SDMX (pandas 3.x safe)."""

from __future__ import annotations

import io
import logging
import os
from typing import Iterable

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    BIS_SDMX_BASE_URL,
    DEFAULT_REGION,
    FRED_API_BASE_URL,
    FRED_SERIES_IDS,
    HISTORY_START_DATE,
    INTL_HISTORY_START_DATE,
    MODEL_REGIONS,
    OECD_SDMX_BASE_URL,
)

logger = logging.getLogger(__name__)
# Suppress urllib3 retry URL logging so FRED api_key query parameters never appear in logs.
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Per-region international FRED series (EU, JP, CN, IN)
INTL_FRED_SERIES: dict[str, tuple[str, ...]] = {
    "EU": ("CLV10MEURB1GQSCAEA20Q", "RBXMBIS"),
    "JP": ("JPNRGDPEXP", "IRLTLT01JPM156N", "SPASTT01JPM661N", "RBJPBIS"),
    "CN": ("IR3TIB01CNM156N", "SPASTT01CNM661N", "RBCNBIS"),
    "IN": ("SPASTT01INM661N", "RBINBIS"),
}


def _build_retry_session(
    total_retries: int = 4,
    backoff_factor: float = 0.8,
) -> requests.Session:
    """Create a requests Session with exponential backoff for transient API errors."""
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
    session.headers.update({"User-Agent": "macro-recession-predictor/2.0"})
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


def _parse_sdmx_period(period: str) -> pd.Timestamp | None:
    """Parse SDMX period strings (YYYY-MM, YYYY-MM-DD, YYYY-Q#) to month-end Timestamps."""
    p = str(period).strip()
    if not p:
        return None
    try:
        if "Q" in p and len(p) in (6, 7):
            return pd.Period(p.replace("-", ""), freq="Q").to_timestamp(how="end").normalize()
        if len(p) == 7 and p[4] == "-":
            return pd.Period(p, freq="M").to_timestamp(how="end").normalize()
        if len(p) == 10:
            return pd.Timestamp(p)
        return pd.to_datetime(p)
    except Exception:
        return None


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
    session: requests.Session | None = None,
    ingested_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch a set of FRED series for one region into the normalized raw_macro_series schema."""
    key = api_key or get_fred_api_key()
    ts = ingested_at or pd.Timestamp.now(tz="UTC")
    own_session = session is None
    http = session or _build_retry_session()
    frames: list[pd.DataFrame] = []

    try:
        for series_id in series_ids:
            logger.info("Fetching FRED [%s] %s from %s...", region, series_id, start_date)
            series_df = fetch_fred_series(
                series_id=series_id,
                api_key=key,
                start_date=start_date,
                session=http,
            )
            series_df["region"] = region
            series_df["series_id"] = series_id
            series_df["ingested_at"] = ts
            frames.append(series_df)
    finally:
        if own_session:
            http.close()

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = raw_df[["region", "series_id", "observation_date", "value", "ingested_at"]]
    return raw_df.sort_values(["region", "series_id", "observation_date"]).reset_index(drop=True)


def fetch_bis_sdmx_series(
    dataflow: str,
    key: str,
    area_to_region: dict[str, str],
    series_id_template: str,
    start_date: str = INTL_HISTORY_START_DATE,
    session: requests.Session | None = None,
    ingested_at: pd.Timestamp | None = None,
    timeout: int = 35,
) -> pd.DataFrame:
    """Fetch a multi-country BIS SDMX v1 CSV dataflow (e.g., WS_CBPOL, WS_LONG_CPI)."""
    url = f"{BIS_SDMX_BASE_URL}/{dataflow}/{key}?format=csv"
    ts = ingested_at or pd.Timestamp.now(tz="UTC")
    min_ts = pd.Timestamp(start_date)
    own_session = session is None
    http = session or _build_retry_session()

    logger.info("Fetching BIS SDMX %s/%s...", dataflow, key)
    try:
        response = http.get(url, timeout=timeout)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
    finally:
        if own_session:
            http.close()

    area_col = next((c for c in ("REF_AREA", "Reference area") if c in df.columns), None)
    time_col = next((c for c in ("TIME_PERIOD", "Time Period", "PERIOD") if c in df.columns), None)
    val_col = next((c for c in ("OBS_VALUE", "Observation Value", "VALUE") if c in df.columns), None)
    if not (area_col and time_col and val_col):
        raise RuntimeError(f"Unexpected BIS SDMX CSV columns for {dataflow}: {list(df.columns)[:8]}")

    frames: list[pd.DataFrame] = []
    for area_code, region in area_to_region.items():
        # BIS SDMX CSV sometimes formats area values as "XM: Euro area" or bare "XM".
        sub = df[df[area_col].astype(str).str.split(":").str[0].str.strip() == area_code]
        records: list[tuple[pd.Timestamp, float]] = []
        for p, v in zip(sub[time_col], sub[val_col]):
            if pd.isna(v):
                continue
            dt = _parse_sdmx_period(str(p))
            if dt is not None and dt >= min_ts:
                try:
                    records.append((dt, float(v)))
                except (ValueError, TypeError):
                    continue
        if not records:
            raise RuntimeError(f"BIS SDMX {dataflow} returned no observations for area '{area_code}'.")
        rdf = pd.DataFrame(records, columns=["observation_date", "value"])
        rdf = rdf.drop_duplicates(subset=["observation_date"], keep="last").sort_values("observation_date")
        rdf["region"] = region
        rdf["series_id"] = series_id_template.format(area=area_code)
        rdf["value"] = rdf["value"].astype("float64")
        rdf["ingested_at"] = ts
        frames.append(rdf[["region", "series_id", "observation_date", "value", "ingested_at"]])

    return pd.concat(frames, ignore_index=True)


def fetch_oecd_sdmx_series(
    agency_flow: str,
    key: str,
    area_to_region: dict[str, str],
    series_id_template: str,
    start_period: str = "1980-01",
    session: requests.Session | None = None,
    ingested_at: pd.Timestamp | None = None,
    timeout: int = 40,
) -> pd.DataFrame:
    """Fetch an OECD SDMX REST CSV dataset (e.g., DSD_STES@DF_CLI, DSD_STES@DF_FINMARK)."""
    url = (
        f"{OECD_SDMX_BASE_URL}/{agency_flow}/{key}"
        f"?startPeriod={start_period}&format=csvfilewithlabels"
    )
    ts = ingested_at or pd.Timestamp.now(tz="UTC")
    own_session = session is None
    http = session or _build_retry_session()

    logger.info("Fetching OECD SDMX %s/%s...", agency_flow, key)
    try:
        response = http.get(url, timeout=timeout)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
    finally:
        if own_session:
            http.close()

    area_col = "REF_AREA" if "REF_AREA" in df.columns else df.columns[0]
    time_col = "TIME_PERIOD" if "TIME_PERIOD" in df.columns else "Time"
    val_col = "OBS_VALUE" if "OBS_VALUE" in df.columns else "Value"

    frames: list[pd.DataFrame] = []
    for area_code, region in area_to_region.items():
        sub = df[df[area_col] == area_code]
        records: list[tuple[pd.Timestamp, float]] = []
        for p, v in zip(sub[time_col], sub[val_col]):
            if pd.isna(v):
                continue
            dt = _parse_sdmx_period(str(p))
            if dt is not None:
                try:
                    records.append((dt, float(v)))
                except (ValueError, TypeError):
                    continue
        if not records:
            raise RuntimeError(f"OECD SDMX {agency_flow} returned no observations for area '{area_code}'.")
        rdf = pd.DataFrame(records, columns=["observation_date", "value"])
        rdf = rdf.drop_duplicates(subset=["observation_date"], keep="last").sort_values("observation_date")
        rdf["region"] = region
        rdf["series_id"] = series_id_template.format(area=area_code)
        rdf["value"] = rdf["value"].astype("float64")
        rdf["ingested_at"] = ts
        frames.append(rdf[["region", "series_id", "observation_date", "value", "ingested_at"]])

    return pd.concat(frames, ignore_index=True)


def fetch_all_regions_raw(
    regions: Iterable[str] = MODEL_REGIONS,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch all verified Phase 1 (US) and Phase 2 (EU, JP, CN, IN) series across FRED, BIS, and OECD."""
    region_set = set(regions)
    key = api_key or get_fred_api_key()
    ingested_at = pd.Timestamp.now(tz="UTC")
    frames: list[pd.DataFrame] = []

    with _build_retry_session() as session:
        # 1. US (Phase 1 — completely intact from 1954-01-01)
        if "US" in region_set:
            frames.append(
                fetch_all_fred_series(
                    series_ids=FRED_SERIES_IDS,
                    region="US",
                    start_date=HISTORY_START_DATE,
                    api_key=key,
                    session=session,
                    ingested_at=ingested_at,
                )
            )

        # 2. International FRED series (GDP targets, yields, equities, REER)
        for reg in ("EU", "JP", "CN", "IN"):
            if reg in region_set:
                frames.append(
                    fetch_all_fred_series(
                        series_ids=INTL_FRED_SERIES[reg],
                        region=reg,
                        start_date=INTL_HISTORY_START_DATE,
                        api_key=key,
                        session=session,
                        ingested_at=ingested_at,
                    )
                )

        # 3. BIS SDMX batched fetches (Policy Rate & Headline CPI) for EU/JP/CN/IN
        bis_area_map = {
            area: reg
            for area, reg in (("XM", "EU"), ("JP", "JP"), ("CN", "CN"), ("IN", "IN"))
            if reg in region_set
        }
        if bis_area_map:
            area_key = "+".join(bis_area_map.keys())
            frames.append(
                fetch_bis_sdmx_series(
                    dataflow="WS_CBPOL",
                    key=f"M.{area_key}",
                    area_to_region=bis_area_map,
                    series_id_template="BIS:WS_CBPOL/M.{area}",
                    start_date=INTL_HISTORY_START_DATE,
                    session=session,
                    ingested_at=ingested_at,
                )
            )
            frames.append(
                fetch_bis_sdmx_series(
                    dataflow="WS_LONG_CPI",
                    key=f"M.{area_key}.628",
                    area_to_region=bis_area_map,
                    series_id_template="BIS:WS_LONG_CPI/M.{area}",
                    start_date=INTL_HISTORY_START_DATE,
                    session=session,
                    ingested_at=ingested_at,
                )
            )

        # 4. OECD SDMX CLI (G4E for EU, JPN, CHN, IND)
        cli_area_map = {
            area: reg
            for area, reg in (("G4E", "EU"), ("JPN", "JP"), ("CHN", "CN"), ("IND", "IN"))
            if reg in region_set
        }
        if cli_area_map:
            cli_key = "+".join(cli_area_map.keys())
            frames.append(
                fetch_oecd_sdmx_series(
                    agency_flow="OECD.SDD.STES,DSD_STES@DF_CLI",
                    key=f"{cli_key}.M.LI...AA...H",
                    area_to_region=cli_area_map,
                    series_id_template="OECD:DF_CLI:{area}",
                    start_period="1990-01",
                    session=session,
                    ingested_at=ingested_at,
                )
            )

        # 5. OECD SDMX Financial Markets for EU (EA20 10Y, 3M, Broad Equities)
        if "EU" in region_set:
            for measure, unit_code, series_tpl in (
                ("IRLT", "PA", "OECD:DF_FINMARK:IRLT:{area}"),
                ("IR3TIB", "PA", "OECD:DF_FINMARK:IR3TIB:{area}"),
                ("SHARE", "IX", "OECD:DF_FINMARK:SHARE:{area}"),
            ):
                frames.append(
                    fetch_oecd_sdmx_series(
                        agency_flow="OECD.SDD.STES,DSD_STES@DF_FINMARK",
                        key=f"EA20.M.{measure}.{unit_code}.....",
                        area_to_region={"EA20": "EU"},
                        series_id_template=series_tpl,
                        start_period="1980-01",
                        session=session,
                        ingested_at=ingested_at,
                    )
                )

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = raw_df[["region", "series_id", "observation_date", "value", "ingested_at"]]
    return raw_df.sort_values(["region", "series_id", "observation_date"]).reset_index(drop=True)


def pivot_to_month_end(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long raw_macro_series (single region) to a regular month-end ('ME') DataFrame.

    Uses pandas 3.x-safe `resample("ME").last()` so daily/monthly series
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
