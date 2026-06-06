from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from scipy.ndimage import gaussian_filter
import streamlit as st
import streamlit.components.v1 as components


APP_TITLE = "ARRL Digital Contest Band Advisor"
WSPRNET_ENDPOINT = "https://www.wsprnet.org/drupal/wsprnet/spots/json"
WSPR_LIVE_ENDPOINT = "https://db1.wspr.live/"
PSK_REPORTER_ENDPOINT = (
    "https://retrieve.pskreporter.info/query"
    "?senderCallsign=&mode=FT8&flowStartSeconds=-900&statsSummary"
    "&lastSequenceNumber=0&fCallSigns&noactive=1"
)

WSPR_BANDS = [160, 80, 40, 30, 20, 17, 15, 12, 10]
DEFAULT_BANDS = {40, 20, 15, 10}
BAND_LABELS = {band: f"{band}m" for band in WSPR_BANDS}
WSPR_BAND_IDS = {
    160: 1,
    80: 3,
    40: 7,
    30: 10,
    20: 14,
    17: 18,
    15: 21,
    12: 24,
    10: 28,
}
BAND_RANGES_MHZ = {
    160: (1.8, 2.0),
    80: (3.5, 4.0),
    40: (7.0, 7.3),
    30: (10.1, 10.15),
    20: (14.0, 14.35),
    17: (18.068, 18.168),
    15: (21.0, 21.45),
    12: (24.89, 24.99),
    10: (28.0, 29.7),
}

MODEL_LOOKBACK_MINUTES_DEFAULT = 30
WSPR_ROWS_PER_BAND = 50_000
PSK_LOOKBACK_SECONDS = 900
DISTANCE_BIN_KM = 500
MAX_MODEL_DISTANCE_KM = 20_000
DEFAULT_MAX_CONTACT_DISTANCE_KM = 20_000
AZIMUTH_BIN_DEG = 30
TIME_BUCKET_MINUTES = 15
BETA_ALPHA = 0.5
BETA_BETA = 9.5
GLOBAL_MAP_LAT_STEP_DEG = 2.5
GLOBAL_MAP_LON_STEP_DEG = 5.0
GRID4_RE = re.compile(r"^[A-R]{2}[0-9]{2}$")
GRID_RE = re.compile(r"^[A-R]{2}[0-9]{2}([A-X]{2})?$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_label(value: str | datetime | pd.Timestamp | None) -> str:
    if value is None:
        return "unknown"
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return "unknown"
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")


def as_records(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("spots", "data", "rows", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        values = list(payload.values())
        if values and all(isinstance(item, dict) for item in values):
            return values
    return []


def pick_series(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    default: Any = pd.NA,
) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index)


def string_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    return pick_series(frame, candidates).astype("string").str.strip()


def numeric_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    return pd.to_numeric(pick_series(frame, candidates), errors="coerce")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_dates = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    if numeric.notna().any():
        median_value = float(numeric.dropna().median())
        unit = "ms" if median_value > 1_000_000_000_000 else "s"
        numeric_dates = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

    text_dates = pd.to_datetime(series, utc=True, errors="coerce")
    return numeric_dates.combine_first(text_dates)


def maidenhead_to_latlon(grid: str | None) -> tuple[float, float] | None:
    if not grid:
        return None
    clean = str(grid).strip().upper()
    if not GRID_RE.fullmatch(clean):
        return None

    lon = -180 + (ord(clean[0]) - ord("A")) * 20
    lat = -90 + (ord(clean[1]) - ord("A")) * 10
    lon += int(clean[2]) * 2
    lat += int(clean[3])

    if len(clean) >= 6:
        lon += (ord(clean[4]) - ord("A")) * (5 / 60)
        lat += (ord(clean[5]) - ord("A")) * (2.5 / 60)
        lon += 2.5 / 60
        lat += 1.25 / 60
    else:
        lon += 1
        lat += 0.5
    return lat, lon


def nearby_maidenhead4(center_grid: str, radius: int = 2) -> list[str]:
    clean = center_grid.strip().upper()
    if not GRID4_RE.fullmatch(clean):
        return []

    lon_index = (ord(clean[0]) - ord("A")) * 10 + int(clean[2])
    lat_index = (ord(clean[1]) - ord("A")) * 10 + int(clean[3])
    grids: list[str] = []
    for lat_offset in range(-radius, radius + 1):
        for lon_offset in range(-radius, radius + 1):
            candidate_lon = lon_index + lon_offset
            candidate_lat = lat_index + lat_offset
            if not (0 <= candidate_lon < 180 and 0 <= candidate_lat < 180):
                continue
            grids.append(
                f"{chr(65 + candidate_lon // 10)}"
                f"{chr(65 + candidate_lat // 10)}"
                f"{candidate_lon % 10}{candidate_lat % 10}"
            )
    return sorted(grids)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_km_array(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    radius_km = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(d_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - (
        math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def bearing_degrees_array(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_lambda = np.radians(lon2 - lon1)
    y = np.sin(d_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - (
        np.sin(phi1) * np.cos(phi2) * np.cos(d_lambda)
    )
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frequency_to_band_m(frequency: str | float | int | None) -> int | None:
    value = parse_float(str(frequency)) if frequency is not None else None
    if value is None or value <= 0:
        return None
    if value > 1_000_000:
        mhz = value / 1_000_000
    elif value > 1_000:
        mhz = value / 1_000
    else:
        mhz = value
    for band, (low, high) in BAND_RANGES_MHZ.items():
        if low <= mhz <= high:
            return band
    return None


def distance_bin_start(values: pd.Series | np.ndarray | float) -> pd.Series | int:
    numeric = pd.to_numeric(values, errors="coerce")
    binned = (np.floor(numeric / DISTANCE_BIN_KM) * DISTANCE_BIN_KM).clip(
        0,
        MAX_MODEL_DISTANCE_KM - DISTANCE_BIN_KM,
    )
    if np.isscalar(values):
        return int(binned) if not pd.isna(binned) else 0
    return pd.Series(binned, index=getattr(values, "index", None), dtype="float")


def azimuth_bin_start(values: pd.Series | np.ndarray | float) -> pd.Series | int:
    numeric = pd.to_numeric(values, errors="coerce") % 360
    binned = np.floor(numeric / AZIMUTH_BIN_DEG) * AZIMUTH_BIN_DEG
    if np.isscalar(values):
        return int(binned) if not pd.isna(binned) else 0
    return pd.Series(binned, index=getattr(values, "index", None), dtype="float")


def azimuth_label(start: int | float) -> str:
    begin = int(start) % 360
    end = (begin + AZIMUTH_BIN_DEG) % 360
    return f"{begin:03d}-{end:03d}"


def normalise_wspr_spots(records: list[dict[str, Any]], requested_band: int) -> pd.DataFrame:
    columns = [
        "band_m",
        "callsign",
        "reporter",
        "snr_db",
        "distance_km",
        "timestamp",
        "sender_grid",
        "reporter_grid",
        "source",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    raw = pd.DataFrame(records)
    raw.columns = [str(column).strip().lower() for column in raw.columns]

    normalised = pd.DataFrame(index=raw.index)
    normalised["band_m"] = requested_band
    normalised["callsign"] = (
        string_series(raw, ["callsign", "txcall", "tx_call", "tx_sign", "sender"])
        .str.upper()
        .replace({"": pd.NA})
    )
    normalised["reporter"] = (
        string_series(raw, ["reporter", "rxcall", "rx_call", "rx_sign", "receiver"])
        .str.upper()
        .replace({"": pd.NA})
    )
    normalised["snr_db"] = numeric_series(raw, ["rxsnr", "snr", "signal", "s_n_r", "db"])
    normalised["distance_km"] = numeric_series(
        raw,
        ["dist", "distance", "distancekm", "distance_km"],
    )
    normalised["timestamp"] = parse_timestamp_series(
        pick_series(raw, ["timestamp", "time", "date", "datetime", "spotdate"]),
    )
    normalised["sender_grid"] = (
        string_series(
            raw,
            [
                "txgrid",
                "tx_grid",
                "txloc",
                "tx_loc",
                "sendergrid",
                "sender_locator",
                "senderlocator",
                "grid",
            ],
        )
        .str.upper()
        .replace({"": pd.NA})
    )
    normalised["reporter_grid"] = (
        string_series(
            raw,
            [
                "rxgrid",
                "rx_grid",
                "rxloc",
                "rx_loc",
                "reportergrid",
                "reporter_grid",
                "receivergrid",
                "receiver_locator",
                "receiverlocator",
            ],
        )
        .str.upper()
        .replace({"": pd.NA})
    )
    normalised["source"] = "unknown"
    return normalised[columns]


def coords_from_grid_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    coords = series.apply(maidenhead_to_latlon)
    lat = coords.apply(lambda value: value[0] if value else np.nan)
    lon = coords.apply(lambda value: value[1] if value else np.nan)
    return lat.astype(float), lon.astype(float)


def fetch_wsprnet_band_frame(
    band: int,
    count: int,
    lookback_minutes: int,
) -> pd.DataFrame:
    params = {
        "band": str(WSPR_BAND_IDS[band]),
        "minutes": str(max(2, lookback_minutes)),
        "exclude_special": "1",
    }
    response = requests.get(WSPRNET_ENDPOINT, params=params, timeout=10)
    response.raise_for_status()
    records = as_records(response.json())
    if not records:
        return pd.DataFrame()
    frame = normalise_wspr_spots(records, band)
    frame["source"] = "WSPRnet"
    return frame.sort_values("timestamp", ascending=False, na_position="last").head(count)


def fetch_wsprlive_band_frame(
    band: int,
    count: int,
    lookback_minutes: int,
) -> pd.DataFrame:
    band_id = WSPR_BAND_IDS[band]
    clean_count = max(1, min(int(count), 50_000))
    clean_minutes = max(2, min(int(lookback_minutes), 24 * 60))
    query = f"""
    SELECT
        bucket AS time,
        band,
        tx_sign,
        tx_grid AS tx_loc,
        rx_sign,
        rx_grid AS rx_loc,
        avg_snr AS snr,
        path_distance_km AS distance
    FROM
    (
        SELECT
            toStartOfInterval(time, INTERVAL {TIME_BUCKET_MINUTES} MINUTE) AS bucket,
            band,
            tx_sign,
            any(tx_loc) AS tx_grid,
            rx_sign,
            any(rx_loc) AS rx_grid,
            avg(snr) AS avg_snr,
            max(distance) AS path_distance_km
        FROM wspr.rx
        WHERE time >= now() - INTERVAL {clean_minutes} MINUTE
          AND time < now()
          AND band = {band_id}
          AND distance BETWEEN 0 AND 20000
          AND code = 1
          AND snr BETWEEN -40 AND 40
          AND tx_loc != ''
          AND rx_loc != ''
        GROUP BY bucket, band, tx_sign, rx_sign
    )
    ORDER BY time DESC
    LIMIT {clean_count}
    FORMAT JSON
    """
    response = requests.get(WSPR_LIVE_ENDPOINT, params={"query": query}, timeout=10)
    response.raise_for_status()
    records = as_records(response.json())
    if not records:
        return pd.DataFrame()
    frame = normalise_wspr_spots(records, band)
    frame["source"] = "WSPR.live"
    return frame


@st.cache_data(ttl=300, show_spinner=False)
def fetch_wspr_band(
    band: int,
    count: int,
    lookback_minutes: int,
    cache_bust: str,
) -> tuple[pd.DataFrame, str | None, str]:
    del cache_bust
    fetched_at = utc_now().isoformat()
    wsprnet_error: str | None = None
    try:
        frame = fetch_wsprnet_band_frame(band, count, lookback_minutes)
        if not frame.empty:
            return frame, None, fetched_at
        wsprnet_error = "WSPRnet returned no spots"
    except requests.RequestException as exc:
        wsprnet_error = str(exc)
    except ValueError as exc:
        wsprnet_error = f"invalid JSON: {exc}"

    try:
        frame = fetch_wsprlive_band_frame(band, count, lookback_minutes)
    except requests.RequestException as exc:
        return (
            pd.DataFrame(),
            f"WSPRnet unavailable for {BAND_LABELS[band]} ({wsprnet_error}); WSPR.live fallback also failed: {exc}",
            fetched_at,
        )
    except ValueError as exc:
        return (
            pd.DataFrame(),
            f"WSPRnet unavailable for {BAND_LABELS[band]} ({wsprnet_error}); WSPR.live returned invalid JSON: {exc}",
            fetched_at,
        )

    if frame.empty:
        return (
            pd.DataFrame(),
            f"No WSPR spots found for {BAND_LABELS[band]} from WSPRnet or WSPR.live.",
            fetched_at,
        )
    return frame, None, fetched_at


def fetch_wspr_dataset(
    bands: tuple[int, ...],
    *,
    count: int,
    lookback_minutes: int,
    cache_bust: str,
) -> tuple[pd.DataFrame, list[str], str | None, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    fetched_at_values: list[str] = []
    source_counts: dict[str, int] = {}
    for band in bands:
        frame, warning, fetched_at = fetch_wspr_band(
            band,
            count,
            lookback_minutes,
            cache_bust,
        )
        fetched_at_values.append(fetched_at)
        if warning:
            warnings.append(warning)
        if not frame.empty:
            source = str(frame["source"].iloc[0]) if "source" in frame else "unknown"
            source_counts[source] = source_counts.get(source, 0) + int(len(frame))
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, warnings, max(fetched_at_values) if fetched_at_values else None, source_counts


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def xml_value(element: ET.Element, candidates: list[str]) -> str | None:
    candidate_set = {candidate.lower() for candidate in candidates}
    attrs = {key.lower(): value for key, value in element.attrib.items()}
    for candidate in candidate_set:
        value = attrs.get(candidate)
        if value not in (None, ""):
            return value
    for child in list(element):
        if local_name(child.tag) in candidate_set and child.text:
            return child.text.strip()
    return None


def parse_psk_report(element: ET.Element) -> dict[str, Any] | None:
    frequency = xml_value(element, ["frequency", "freq"])
    band = frequency_to_band_m(frequency)
    if band is None:
        return None

    receiver_call = xml_value(
        element,
        ["receiverCallsign", "reporterCallsign", "receiver", "reporter"],
    )
    sender_call = xml_value(element, ["senderCallsign", "callsign", "sender"])
    receiver_grid = xml_value(
        element,
        ["receiverLocator", "reporterLocator", "receiverGrid", "reporter_grid"],
    )
    sender_grid = xml_value(element, ["senderLocator", "senderGrid", "sender_grid"])
    sender_coords = maidenhead_to_latlon(sender_grid)
    receiver_coords = maidenhead_to_latlon(receiver_grid)
    path_distance = None
    if sender_coords and receiver_coords:
        path_distance = haversine_km(
            sender_coords[0],
            sender_coords[1],
            receiver_coords[0],
            receiver_coords[1],
        )

    return {
        "band_m": band,
        "frequency_hz": parse_float(frequency),
        "sender_callsign": sender_call.strip().upper() if sender_call else pd.NA,
        "sender_grid": sender_grid.strip().upper() if sender_grid else pd.NA,
        "receiver_callsign": receiver_call.strip().upper() if receiver_call else pd.NA,
        "receiver_grid": receiver_grid.strip().upper() if receiver_grid else pd.NA,
        "path_distance_km": path_distance,
        "snr_db": parse_float(xml_value(element, ["sNR", "snr", "signal"])),
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_psk_reporter(cache_bust: str) -> tuple[pd.DataFrame, str | None, str]:
    del cache_bust
    fetched_at = utc_now().isoformat()
    try:
        response = requests.get(PSK_REPORTER_ENDPOINT, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return pd.DataFrame(), f"PSKReporter unavailable: {exc}", fetched_at

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        return (
            pd.DataFrame(),
            f"PSKReporter XML parse failed; scoring will use no PSK density. {exc}",
            fetched_at,
        )

    reports = []
    for element in root.iter():
        if local_name(element.tag) != "receptionreport":
            continue
        parsed = parse_psk_report(element)
        if parsed:
            reports.append(parsed)
    if not reports:
        return (
            pd.DataFrame(),
            "PSKReporter returned no usable FT8 reception reports.",
            fetched_at,
        )
    return pd.DataFrame(reports), None, fetched_at


def prepare_wspr_spots(
    wspr: pd.DataFrame,
    selected_bands: tuple[int, ...],
    lookback_minutes: int,
    max_distance_km: int,
) -> pd.DataFrame:
    columns = [
        "band_m",
        "callsign",
        "reporter",
        "snr_db",
        "distance_km",
        "timestamp",
        "sender_grid",
        "reporter_grid",
        "source",
    ]
    if wspr.empty:
        return pd.DataFrame(columns=columns)

    now = pd.Timestamp.now(tz="UTC")
    working = wspr.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working["distance_km"] = pd.to_numeric(working["distance_km"], errors="coerce")
    working["callsign"] = working["callsign"].astype("string").str.upper().str.strip()
    working["reporter"] = working["reporter"].astype("string").str.upper().str.strip()
    working["sender_grid"] = working["sender_grid"].astype("string").str.upper().str.strip()
    working["reporter_grid"] = working["reporter_grid"].astype("string").str.upper().str.strip()
    mask = (
        working["band_m"].isin(selected_bands)
        & working["timestamp"].notna()
        & (
            working["timestamp"]
            >= now - pd.Timedelta(minutes=lookback_minutes + TIME_BUCKET_MINUTES)
        )
        & (working["timestamp"] <= now + pd.Timedelta(minutes=5))
        & working["distance_km"].between(0, 20000)
        & working["sender_grid"].map(lambda value: bool(GRID_RE.fullmatch(str(value))))
        & working["reporter_grid"].map(lambda value: bool(GRID_RE.fullmatch(str(value))))
        & working["callsign"].notna()
        & working["reporter"].notna()
    )
    working = working.loc[mask, columns].copy()
    if working.empty:
        return working

    sender_lat, sender_lon = coords_from_grid_series(working["sender_grid"])
    reporter_lat, reporter_lon = coords_from_grid_series(working["reporter_grid"])
    working["sender_lat"] = sender_lat
    working["sender_lon"] = sender_lon
    working["reporter_lat"] = reporter_lat
    working["reporter_lon"] = reporter_lon
    working = working.dropna(
        subset=["sender_lat", "sender_lon", "reporter_lat", "reporter_lon"],
    )
    working["time_bucket"] = working["timestamp"].dt.floor(f"{TIME_BUCKET_MINUTES}min")
    return working


def station_distance_from_home(
    frame: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    home_coords: tuple[float, float],
) -> tuple[pd.Series, pd.Series]:
    distances = []
    bearings = []
    for row in frame[[lat_col, lon_col]].itertuples(index=False):
        lat = float(row[0])
        lon = float(row[1])
        distances.append(haversine_km(home_coords[0], home_coords[1], lat, lon))
        bearings.append(bearing_degrees(home_coords[0], home_coords[1], lat, lon))
    return pd.Series(distances, index=frame.index), pd.Series(bearings, index=frame.index)


def build_station_activity(
    spots: pd.DataFrame,
    home_coords: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spots.empty:
        empty_cols = [
            "band_m",
            "time_bucket",
            "callsign",
            "grid",
            "grid4",
            "lat",
            "lon",
            "distance_home_km",
            "azimuth_home_deg",
        ]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    tx = (
        spots[
            [
                "band_m",
                "time_bucket",
                "callsign",
                "sender_grid",
                "sender_lat",
                "sender_lon",
            ]
        ]
        .rename(
            columns={
                "callsign": "callsign",
                "sender_grid": "grid",
                "sender_lat": "lat",
                "sender_lon": "lon",
            },
        )
        .drop_duplicates()
    )
    rx = (
        spots[
            [
                "band_m",
                "time_bucket",
                "reporter",
                "reporter_grid",
                "reporter_lat",
                "reporter_lon",
            ]
        ]
        .rename(
            columns={
                "reporter": "callsign",
                "reporter_grid": "grid",
                "reporter_lat": "lat",
                "reporter_lon": "lon",
            },
        )
        .drop_duplicates()
    )
    for frame in (tx, rx):
        frame["grid4"] = frame["grid"].astype("string").str[:4].str.upper()
        distance, bearing = station_distance_from_home(frame, "lat", "lon", home_coords)
        frame["distance_home_km"] = distance
        frame["azimuth_home_deg"] = bearing
    return tx, rx


def build_wspr_opportunities(
    spots: pd.DataFrame,
    home_coords: tuple[float, float],
    local_grids: set[str],
    max_distance_km: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx_activity, rx_activity = build_station_activity(spots, home_coords)
    if spots.empty or tx_activity.empty or rx_activity.empty:
        return pd.DataFrame(), tx_activity, rx_activity

    success_pairs = set(
        spots[["band_m", "time_bucket", "callsign", "reporter"]]
        .drop_duplicates()
        .itertuples(index=False, name=None),
    )
    opportunities: list[dict[str, Any]] = []
    group_keys = sorted(
        set(tx_activity[["band_m", "time_bucket"]].itertuples(index=False, name=None))
        | set(rx_activity[["band_m", "time_bucket"]].itertuples(index=False, name=None)),
    )

    for band, bucket in group_keys:
        tx_group = tx_activity[
            (tx_activity["band_m"] == band) & (tx_activity["time_bucket"] == bucket)
        ]
        rx_group = rx_activity[
            (rx_activity["band_m"] == band) & (rx_activity["time_bucket"] == bucket)
        ]
        if tx_group.empty or rx_group.empty:
            continue

        local_rx = rx_group[rx_group["grid4"].isin(local_grids)]
        local_tx = tx_group[tx_group["grid4"].isin(local_grids)]

        for local in local_rx.itertuples(index=False):
            for remote in tx_group.itertuples(index=False):
                if str(local.callsign) == str(remote.callsign):
                    continue
                distance = float(remote.distance_home_km)
                if remote.grid4 in local_grids or distance > max_distance_km:
                    continue
                azimuth = float(remote.azimuth_home_deg)
                success = (band, bucket, remote.callsign, local.callsign) in success_pairs
                opportunities.append(
                    {
                        "band_m": band,
                        "time_bucket": bucket,
                        "direction": "inbound to local RX",
                        "local_call": local.callsign,
                        "remote_call": remote.callsign,
                        "remote_grid": remote.grid,
                        "remote_lat": remote.lat,
                        "remote_lon": remote.lon,
                        "distance_home_km": distance,
                        "azimuth_home_deg": azimuth,
                        "success": int(success),
                    },
                )

        for local in local_tx.itertuples(index=False):
            for remote in rx_group.itertuples(index=False):
                if str(local.callsign) == str(remote.callsign):
                    continue
                distance = float(remote.distance_home_km)
                if remote.grid4 in local_grids or distance > max_distance_km:
                    continue
                azimuth = float(remote.azimuth_home_deg)
                success = (band, bucket, local.callsign, remote.callsign) in success_pairs
                opportunities.append(
                    {
                        "band_m": band,
                        "time_bucket": bucket,
                        "direction": "outbound from local TX",
                        "local_call": local.callsign,
                        "remote_call": remote.callsign,
                        "remote_grid": remote.grid,
                        "remote_lat": remote.lat,
                        "remote_lon": remote.lon,
                        "distance_home_km": distance,
                        "azimuth_home_deg": azimuth,
                        "success": int(success),
                    },
                )

    opportunities_frame = pd.DataFrame(opportunities)
    if opportunities_frame.empty:
        return opportunities_frame, tx_activity, rx_activity
    opportunities_frame["distance_bin_km"] = distance_bin_start(
        opportunities_frame["distance_home_km"],
    ).astype(int)
    opportunities_frame["azimuth_bin_deg"] = azimuth_bin_start(
        opportunities_frame["azimuth_home_deg"],
    ).astype(int)
    opportunities_frame["Band"] = opportunities_frame["band_m"].map(BAND_LABELS)
    opportunities_frame["Azimuth"] = opportunities_frame["azimuth_bin_deg"].map(azimuth_label)
    return opportunities_frame, tx_activity, rx_activity


def smoothed_probability(successes: float, exposures: float) -> float:
    return float((successes + BETA_ALPHA) / (exposures + BETA_ALPHA + BETA_BETA))


def build_propagation_surface(opportunities: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "band_m",
        "Band",
        "distance_bin_km",
        "azimuth_bin_deg",
        "Azimuth",
        "exposures",
        "successes",
        "raw_success_rate",
        "p_contact",
    ]
    if opportunities.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        opportunities.groupby(["band_m", "distance_bin_km", "azimuth_bin_deg"], dropna=False)
        .agg(exposures=("success", "size"), successes=("success", "sum"))
        .reset_index()
    )
    grouped["Band"] = grouped["band_m"].map(BAND_LABELS)
    grouped["Azimuth"] = grouped["azimuth_bin_deg"].map(azimuth_label)
    grouped["raw_success_rate"] = grouped["successes"] / grouped["exposures"]
    grouped["p_contact"] = [
        smoothed_probability(success, exposure)
        for success, exposure in zip(grouped["successes"], grouped["exposures"], strict=True)
    ]
    return grouped[columns]


def aggregate_probability(
    opportunities: pd.DataFrame,
    group_cols: list[str],
    label_cols: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if opportunities.empty:
        return pd.DataFrame()
    grouped = (
        opportunities.groupby(group_cols, dropna=False)
        .agg(exposures=("success", "size"), successes=("success", "sum"))
        .reset_index()
    )
    grouped["p_contact"] = [
        smoothed_probability(success, exposure)
        for success, exposure in zip(grouped["successes"], grouped["exposures"], strict=True)
    ]
    if label_cols:
        for column, value in label_cols.items():
            grouped[column] = value
    return grouped


def build_distance_profile(opportunities: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "band_m",
        "Band",
        "distance_bin_km",
        "distance_midpoint_km",
        "exposures",
        "successes",
        "p_contact",
        "expected_points",
    ]
    if opportunities.empty:
        return pd.DataFrame(columns=columns)

    profile = aggregate_probability(opportunities, ["band_m", "distance_bin_km"])
    if profile.empty:
        return pd.DataFrame(columns=columns)
    profile["Band"] = profile["band_m"].map(BAND_LABELS)
    profile["distance_midpoint_km"] = (
        profile["distance_bin_km"].astype(float) + DISTANCE_BIN_KM / 2
    )
    profile["expected_points"] = profile["distance_midpoint_km"] * profile["p_contact"]
    return profile[columns].sort_values(["band_m", "distance_bin_km"])


def build_probability_lookup(opportunities: pd.DataFrame) -> dict[str, Any]:
    exact = build_propagation_surface(opportunities)
    by_distance = aggregate_probability(opportunities, ["band_m", "distance_bin_km"])
    by_band = aggregate_probability(opportunities, ["band_m"])
    global_prior = smoothed_probability(
        float(opportunities["success"].sum()) if not opportunities.empty else 0,
        float(len(opportunities)) if not opportunities.empty else 0,
    )

    return {
        "exact": {
            (int(row.band_m), int(row.distance_bin_km), int(row.azimuth_bin_deg)): (
                float(row.p_contact),
                int(row.exposures),
                "azimuth+distance",
            )
            for row in exact.itertuples(index=False)
        },
        "by_distance": {
            (int(row.band_m), int(row.distance_bin_km)): (
                float(row.p_contact),
                int(row.exposures),
                "distance-only",
            )
            for row in by_distance.itertuples(index=False)
        },
        "by_band": {
            int(row.band_m): (float(row.p_contact), int(row.exposures), "band-only")
            for row in by_band.itertuples(index=False)
        },
        "global": (global_prior, int(len(opportunities)), "global prior"),
    }


def lookup_probability(
    lookup: dict[str, Any],
    band: int,
    distance_bin: int,
    azimuth_bin: int,
) -> tuple[float, int, str]:
    exact = lookup["exact"].get((band, distance_bin, azimuth_bin))
    if exact:
        return exact
    by_distance = lookup["by_distance"].get((band, distance_bin))
    if by_distance:
        return by_distance
    by_band = lookup["by_band"].get(band)
    if by_band:
        return by_band
    return lookup["global"]


def build_psk_active_stations(
    psk_reports: pd.DataFrame,
    selected_bands: tuple[int, ...],
    home_coords: tuple[float, float],
    max_distance_km: int,
) -> pd.DataFrame:
    columns = ["band_m", "callsign", "grid", "role"]
    if psk_reports.empty:
        return pd.DataFrame(columns=columns)

    sender = (
        psk_reports[["band_m", "sender_callsign", "sender_grid"]]
        .rename(columns={"sender_callsign": "callsign", "sender_grid": "grid"})
        .assign(role="sender")
    )
    receiver = (
        psk_reports[["band_m", "receiver_callsign", "receiver_grid"]]
        .rename(columns={"receiver_callsign": "callsign", "receiver_grid": "grid"})
        .assign(role="receiver")
    )
    stations = pd.concat([sender, receiver], ignore_index=True)
    stations["callsign"] = stations["callsign"].astype("string").str.upper().str.strip()
    stations["grid"] = stations["grid"].astype("string").str.upper().str.strip()
    stations = stations[
        stations["band_m"].isin(selected_bands)
        & stations["callsign"].notna()
        & stations["grid"].map(lambda value: bool(GRID_RE.fullmatch(str(value))))
    ].drop_duplicates(["band_m", "callsign", "grid"])
    if stations.empty:
        return stations

    lat, lon = coords_from_grid_series(stations["grid"])
    stations["lat"] = lat
    stations["lon"] = lon
    stations = stations.dropna(subset=["lat", "lon"]).copy()
    distance, bearing = station_distance_from_home(stations, "lat", "lon", home_coords)
    stations["distance_home_km"] = distance
    stations["azimuth_home_deg"] = bearing
    stations = stations[
        stations["distance_home_km"].between(0, max_distance_km)
    ].copy()
    if stations.empty:
        return stations

    stations["distance_bin_km"] = distance_bin_start(stations["distance_home_km"]).astype(int)
    stations["azimuth_bin_deg"] = azimuth_bin_start(stations["azimuth_home_deg"]).astype(int)
    stations["Band"] = stations["band_m"].map(BAND_LABELS)
    stations["Azimuth"] = stations["azimuth_bin_deg"].map(azimuth_label)
    return stations


def score_psk_stations(
    psk_stations: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if psk_stations.empty:
        return pd.DataFrame(), pd.DataFrame(), psk_stations.copy()

    lookup = build_probability_lookup(opportunities)
    scored = psk_stations.copy()
    probabilities = []
    evidence_counts = []
    evidence_sources = []
    for row in scored.itertuples(index=False):
        probability, evidence, source = lookup_probability(
            lookup,
            int(row.band_m),
            int(row.distance_bin_km),
            int(row.azimuth_bin_deg),
        )
        probabilities.append(probability)
        evidence_counts.append(evidence)
        evidence_sources.append(source)
    scored["p_contact"] = probabilities
    scored["evidence_exposures"] = evidence_counts
    scored["probability_source"] = evidence_sources
    scored["expected_points"] = scored["distance_home_km"] * scored["p_contact"]

    band_summary = (
        scored.groupby(["band_m", "Band"], dropna=False)
        .agg(
            active_psk_stations=("callsign", "nunique"),
            expected_points=("expected_points", "sum"),
            mean_p_contact=("p_contact", "mean"),
            median_distance_km=("distance_home_km", "median"),
        )
        .reset_index()
    )
    evidence = (
        opportunities.groupby("band_m", dropna=False)
        .agg(
            wspr_opportunities=("success", "size"),
            wspr_successes=("success", "sum"),
            local_wspr_calls=("local_call", "nunique"),
        )
        .reset_index()
        if not opportunities.empty
        else pd.DataFrame(columns=["band_m", "wspr_opportunities", "wspr_successes", "local_wspr_calls"])
    )
    band_summary = band_summary.merge(evidence, on="band_m", how="left").fillna(
        {"wspr_opportunities": 0, "wspr_successes": 0, "local_wspr_calls": 0},
    )
    direction_summary = (
        scored.groupby(["band_m", "Band", "azimuth_bin_deg", "Azimuth"], dropna=False)
        .agg(
            active_psk_stations=("callsign", "nunique"),
            expected_points=("expected_points", "sum"),
            mean_p_contact=("p_contact", "mean"),
        )
        .reset_index()
        .sort_values("expected_points", ascending=False)
    )
    best_direction = direction_summary.drop_duplicates("band_m")[
        ["band_m", "Azimuth", "expected_points"]
    ].rename(
        columns={
            "Azimuth": "best_direction",
            "expected_points": "best_direction_points",
        },
    )
    band_summary = band_summary.merge(best_direction, on="band_m", how="left")
    band_summary = band_summary.sort_values("expected_points", ascending=False)
    band_summary["rank"] = np.arange(1, len(band_summary) + 1)
    band_summary["Recommendation"] = band_summary["rank"].map(
        {1: "best band now", 2: "second choice"},
    ).fillna("usable")
    return band_summary, direction_summary, scored


def style_band_summary(band_summary: pd.DataFrame) -> Any:
    if band_summary.empty:
        return band_summary
    display = band_summary[
        [
            "Band",
            "local_wspr_calls",
            "wspr_opportunities",
            "wspr_successes",
            "mean_p_contact",
            "active_psk_stations",
            "expected_points",
            "best_direction",
            "Recommendation",
        ]
    ].rename(
        columns={
            "local_wspr_calls": "Local WSPR Calls",
            "wspr_opportunities": "WSPR Opportunities",
            "wspr_successes": "WSPR Successes",
            "mean_p_contact": "Mean P(contact)",
            "active_psk_stations": "PSK Stations",
            "expected_points": "E[Points]",
            "best_direction": "Best Direction",
        },
    )

    def row_style(row: pd.Series) -> list[str]:
        recommendation = row["Recommendation"]
        if recommendation == "best band now":
            color = "#d9ead3"
        elif recommendation == "second choice":
            color = "#fff2cc"
        else:
            color = "#eeeeee"
        return [f"background-color: {color}; color: #111827" for _ in row]

    return (
        display.style.apply(row_style, axis=1)
        .format(
            {
                "Local WSPR Calls": "{:,.0f}",
                "WSPR Opportunities": "{:,.0f}",
                "WSPR Successes": "{:,.0f}",
                "Mean P(contact)": "{:.3f}",
                "PSK Stations": "{:,.0f}",
                "E[Points]": "{:,.0f}",
            },
            na_rep="-",
        )
    )


def expected_points_chart(band_summary: pd.DataFrame) -> go.Figure:
    if band_summary.empty:
        fig = go.Figure()
        fig.update_layout(title="No PSKReporter stations available for scoring", height=360)
        return fig
    color_map = {
        "best band now": "#2e7d32",
        "second choice": "#f9ab00",
        "usable": "#9aa0a6",
    }
    fig = px.bar(
        band_summary,
        x="Band",
        y="expected_points",
        color="Recommendation",
        color_discrete_map=color_map,
        hover_data={
            "active_psk_stations": True,
            "mean_p_contact": ":.3f",
            "wspr_opportunities": ":,",
            "best_direction": True,
        },
    )
    fig.update_layout(
        height=420,
        xaxis_title="Band",
        yaxis_title="Expected points from current PSKReporter activity",
        legend_title_text="Recommendation",
    )
    return fig


def direction_chart(direction_summary: pd.DataFrame) -> go.Figure:
    if direction_summary.empty:
        fig = go.Figure()
        fig.update_layout(title="No direction score available", height=420)
        return fig
    ordered = sorted(direction_summary["azimuth_bin_deg"].unique())
    labels = [azimuth_label(value) for value in ordered]
    fig = px.bar(
        direction_summary,
        x="Azimuth",
        y="expected_points",
        color="Band",
        category_orders={"Azimuth": labels},
        hover_data={
            "active_psk_stations": True,
            "mean_p_contact": ":.3f",
        },
    )
    fig.update_layout(
        height=460,
        xaxis_title="Azimuth from home",
        yaxis_title="Expected points",
        barmode="group",
    )
    return fig


def band_direction_chart(direction_summary: pd.DataFrame, band: int) -> go.Figure:
    if direction_summary.empty or "band_m" not in direction_summary:
        fig = go.Figure()
        fig.update_layout(
            title=f"No PSKReporter direction score for {BAND_LABELS[band]}",
            height=420,
        )
        return fig
    band_direction = direction_summary[direction_summary["band_m"] == band].copy()
    if band_direction.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No PSKReporter direction score for {BAND_LABELS[band]}",
            height=420,
        )
        return fig
    ordered_labels = [azimuth_label(value) for value in range(0, 360, AZIMUTH_BIN_DEG)]
    fig = px.bar(
        band_direction,
        x="Azimuth",
        y="expected_points",
        category_orders={"Azimuth": ordered_labels},
        hover_data={
            "active_psk_stations": True,
            "mean_p_contact": ":.3f",
        },
    )
    fig.update_traces(marker_color="#4c78a8")
    fig.update_layout(
        height=430,
        xaxis_title="Azimuth from home",
        yaxis_title="Expected points from current PSKReporter stations",
        title=f"{BAND_LABELS[band]} expected points by direction",
        showlegend=False,
    )
    return fig


def distance_profile_chart(profile: pd.DataFrame, y_axis: str) -> go.Figure:
    if profile.empty:
        fig = go.Figure()
        fig.update_layout(title="No distance profile available", height=460)
        return fig

    if y_axis == "E[points]":
        y_column = "expected_points"
        y_title = "E[points] for one station at distance"
        hover_format = ":,.0f"
    else:
        y_column = "p_contact"
        y_title = "P(contact)"
        hover_format = ":.3f"

    fig = px.line(
        profile,
        x="distance_midpoint_km",
        y=y_column,
        color="Band",
        markers=True,
        hover_data={
            "distance_bin_km": ":,",
            "exposures": ":,",
            "successes": ":,",
            y_column: hover_format,
        },
    )
    fig.update_layout(
        height=500,
        xaxis_title="Distance from home grid (km)",
        yaxis_title=y_title,
        legend_title_text="Band",
    )
    fig.update_xaxes(range=[0, MAX_MODEL_DISTANCE_KM])
    if y_axis == "P(contact)":
        fig.update_yaxes(range=[0, max(float(profile["p_contact"].max()) * 1.08, 0.05)])
    return fig


def propagation_heatmap(surface: pd.DataFrame, band: int) -> go.Figure:
    band_surface = surface[surface["band_m"] == band].copy()
    if band_surface.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No WSPR opportunity evidence for {BAND_LABELS[band]}", height=480)
        return fig
    band_surface["distance_label"] = band_surface["distance_bin_km"].map(
        lambda value: f"{int(value):,}-{int(value + DISTANCE_BIN_KM):,}",
    )
    pivot = band_surface.pivot_table(
        index="distance_label",
        columns="Azimuth",
        values="p_contact",
        aggfunc="mean",
    )
    distance_order = [
        f"{start:,}-{start + DISTANCE_BIN_KM:,}"
        for start in range(0, MAX_MODEL_DISTANCE_KM, DISTANCE_BIN_KM)
        if f"{start:,}-{start + DISTANCE_BIN_KM:,}" in pivot.index
    ]
    azimuth_order = [
        azimuth_label(start)
        for start in range(0, 360, AZIMUTH_BIN_DEG)
        if azimuth_label(start) in pivot.columns
    ]
    pivot = pivot.reindex(index=distance_order, columns=azimuth_order)
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"x": "Azimuth from home", "y": "Distance bin km", "color": "P(contact)"},
    )
    fig.update_layout(height=560, title=f"{BAND_LABELS[band]} WSPR-derived propagation probability")
    return fig


def smoothed_band_probability_matrix(
    surface: pd.DataFrame,
    band: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    distance_bins = np.arange(0, MAX_MODEL_DISTANCE_KM, DISTANCE_BIN_KM)
    azimuth_bins = np.arange(0, 360, AZIMUTH_BIN_DEG)
    band_surface = surface[surface["band_m"] == band].copy()
    if band_surface.empty:
        return None, distance_bins, azimuth_bins

    band_successes = float(band_surface["successes"].sum())
    band_exposures = float(band_surface["exposures"].sum())
    band_prior = smoothed_probability(band_successes, band_exposures)

    distance_probability = {
        int(row.distance_bin_km): smoothed_probability(
            float(row.successes),
            float(row.exposures),
        )
        for row in (
            band_surface.groupby("distance_bin_km", dropna=False)
            .agg(successes=("successes", "sum"), exposures=("exposures", "sum"))
            .reset_index()
        ).itertuples(index=False)
    }

    matrix = np.full((len(distance_bins), len(azimuth_bins)), band_prior, dtype=float)
    for row_idx, distance_bin in enumerate(distance_bins):
        matrix[row_idx, :] = distance_probability.get(int(distance_bin), band_prior)

    distance_index = {int(value): idx for idx, value in enumerate(distance_bins)}
    azimuth_index = {int(value): idx for idx, value in enumerate(azimuth_bins)}
    for row in band_surface.itertuples(index=False):
        row_idx = distance_index.get(int(row.distance_bin_km))
        column_idx = azimuth_index.get(int(row.azimuth_bin_deg))
        if row_idx is None or column_idx is None:
            continue
        matrix[row_idx, column_idx] = float(row.p_contact)

    smoothed = gaussian_filter(matrix, sigma=(1.0, 1.0), mode=("nearest", "wrap"))
    return np.clip(smoothed, 0, 1), distance_bins, azimuth_bins


def build_global_probability_grid(
    surface: pd.DataFrame,
    band: int,
    home_coords: tuple[float, float],
    max_distance_km: int,
) -> pd.DataFrame:
    matrix, distance_bins, azimuth_bins = smoothed_band_probability_matrix(surface, band)
    if matrix is None:
        return pd.DataFrame()

    lats = np.arange(
        -90 + GLOBAL_MAP_LAT_STEP_DEG / 2,
        90,
        GLOBAL_MAP_LAT_STEP_DEG,
    )
    lons = np.arange(
        -180 + GLOBAL_MAP_LON_STEP_DEG / 2,
        180,
        GLOBAL_MAP_LON_STEP_DEG,
    )
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    distances = haversine_km_array(home_coords[0], home_coords[1], lat_grid, lon_grid)
    bearings = bearing_degrees_array(home_coords[0], home_coords[1], lat_grid, lon_grid)

    distance_idx = np.floor(distances / DISTANCE_BIN_KM).astype(int)
    azimuth_idx = np.floor((bearings % 360) / AZIMUTH_BIN_DEG).astype(int)
    distance_idx = np.clip(distance_idx, 0, len(distance_bins) - 1)
    azimuth_idx = np.clip(azimuth_idx, 0, len(azimuth_bins) - 1)

    probabilities = matrix[distance_idx, azimuth_idx]
    probabilities = np.where(distances <= max_distance_km, probabilities, np.nan)
    return pd.DataFrame(
        {
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            "distance_home_km": distances.ravel(),
            "azimuth_home_deg": bearings.ravel(),
            "p_contact": probabilities.ravel(),
        },
    ).dropna(subset=["p_contact"])


def build_station_density_grid(
    psk_stations: pd.DataFrame,
) -> pd.DataFrame:
    if psk_stations.empty:
        return pd.DataFrame()

    lats = np.arange(
        -90 + GLOBAL_MAP_LAT_STEP_DEG / 2,
        90,
        GLOBAL_MAP_LAT_STEP_DEG,
    )
    lons = np.arange(
        -180 + GLOBAL_MAP_LON_STEP_DEG / 2,
        180,
        GLOBAL_MAP_LON_STEP_DEG,
    )
    hist, _, _ = np.histogram2d(
        psk_stations["lat"].astype(float),
        psk_stations["lon"].astype(float),
        bins=[np.linspace(-90, 90, len(lats) + 1), np.linspace(-180, 180, len(lons) + 1)],
    )
    density = gaussian_filter(hist, sigma=(2.0, 2.0), mode="wrap")
    if not np.isfinite(density).any() or float(np.nanmax(density)) <= 0:
        return pd.DataFrame()

    density = np.log1p(density)
    density = density / float(np.nanmax(density))
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return pd.DataFrame(
        {
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            "station_density": density.ravel(),
        },
    )


def global_propagation_map(
    surface: pd.DataFrame,
    psk_stations: pd.DataFrame,
    band: int,
    home_coords: tuple[float, float],
    home_grid: str,
    max_distance_km: int,
    projection_view: str,
) -> go.Figure:
    grid = build_global_probability_grid(surface, band, home_coords, max_distance_km)
    if grid.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No global propagation estimate for {BAND_LABELS[band]}",
            height=620,
        )
        return fig

    density = build_station_density_grid(psk_stations)
    if not density.empty:
        grid = grid.merge(density, on=["lat", "lon"], how="left")
        grid["station_density"] = grid["station_density"].fillna(0.0)
    else:
        grid["station_density"] = 1.0
    grid["expected_contact"] = grid["p_contact"] * grid["station_density"]

    color_ceiling = float(np.nanpercentile(grid["expected_contact"], 95))
    color_ceiling = max(color_ceiling, 0.02)
    projection_type = "orthographic" if projection_view == "Globe" else "natural earth"
    fig = go.Figure(
        go.Scattergeo(
            lat=grid["lat"],
            lon=grid["lon"],
            mode="markers",
            marker={
                "size": 7 if projection_type == "natural earth" else 6,
                "color": grid["expected_contact"],
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": color_ceiling,
                "opacity": 0.72,
                "colorbar": {"title": "Expected contact"},
            },
            customdata=np.stack(
                [
                    grid["distance_home_km"],
                    grid["azimuth_home_deg"],
                    grid["p_contact"],
                    grid["station_density"],
                    grid["expected_contact"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Lat %{lat:.1f}<br>"
                "Lon %{lon:.1f}<br>"
                "Distance %{customdata[0]:,.0f} km<br>"
                "Azimuth %{customdata[1]:.0f} deg<br>"
                "P(contact) %{customdata[2]:.3f}"
                "<br>Station density %{customdata[3]:.3f}"
                "<br>Expected contact %{customdata[4]:.3f}"
                "<extra></extra>"
            ),
            name="Projected propagation",
        ),
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[home_coords[0]],
            lon=[home_coords[1]],
            mode="markers+text",
            marker={"size": 12, "color": "black", "symbol": "star"},
            text=[home_grid],
            textposition="top center",
            name="Home",
        ),
    )
    fig.update_geos(
        projection_type=projection_type,
        projection_rotation={"lon": home_coords[1], "lat": home_coords[0]},
        showland=True,
        landcolor="#f4f4f0",
        showocean=True,
        oceancolor="#eef6fb",
        showcountries=True,
        countrycolor="#9aa0a6",
        coastlinecolor="#6b7280",
    )
    fig.update_layout(
        height=650,
        title=f"{BAND_LABELS[band]} projected propagation from {home_grid}",
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        showlegend=False,
    )
    return fig


def evidence_chart(surface: pd.DataFrame, band: int) -> go.Figure:
    band_surface = surface[surface["band_m"] == band].copy()
    if band_surface.empty:
        return go.Figure()
    band_surface["distance_label"] = band_surface["distance_bin_km"].map(
        lambda value: f"{int(value):,}-{int(value + DISTANCE_BIN_KM):,}",
    )
    fig = px.scatter(
        band_surface,
        x="distance_bin_km",
        y="p_contact",
        size="exposures",
        color="Azimuth",
        hover_data={
            "successes": True,
            "exposures": True,
            "raw_success_rate": ":.3f",
        },
    )
    fig.update_layout(
        height=420,
        xaxis_title="Distance bin start km",
        yaxis_title="Smoothed P(contact)",
    )
    return fig


def psk_station_map(
    scored_stations: pd.DataFrame,
    home_coords: tuple[float, float] | None,
    home_grid: str | None,
) -> go.Figure:
    if scored_stations.empty:
        return go.Figure()
    fig = px.scatter_geo(
        scored_stations,
        lat="lat",
        lon="lon",
        color="Band",
        size="expected_points",
        hover_name="callsign",
        hover_data={
            "grid": True,
            "distance_home_km": ":,.0f",
            "azimuth_home_deg": ":.0f",
            "p_contact": ":.3f",
            "expected_points": ":,.0f",
            "probability_source": True,
            "lat": False,
            "lon": False,
        },
        projection="natural earth",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    if home_coords and home_grid:
        fig.add_trace(
            go.Scattergeo(
                lat=[home_coords[0]],
                lon=[home_coords[1]],
                mode="markers+text",
                marker={"size": 12, "color": "black", "symbol": "star"},
                text=[home_grid],
                textposition="top center",
                name="Home",
            ),
        )
    fig.update_layout(height=620, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    return fig


def sidebar_controls() -> tuple[
    str,
    str | None,
    tuple[float, float] | None,
    int,
    int,
    int,
    tuple[int, ...],
    str,
]:
    st.sidebar.header("Operator")
    operator_call = st.sidebar.text_input("Operator callsign", value="").strip().upper()
    home_grid_raw = st.sidebar.text_input("Home grid square", value="EM49").strip().upper()
    home_grid = home_grid_raw if GRID4_RE.fullmatch(home_grid_raw) else None
    home_coords = maidenhead_to_latlon(home_grid) if home_grid else None
    if home_grid_raw and home_grid is None:
        st.sidebar.error("Enter a valid 4-character Maidenhead grid, e.g. EM49.")

    st.sidebar.header("Refresh")
    refresh_interval_minutes = st.sidebar.slider(
        "Refresh interval",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        format="%d min",
    )
    if "manual_refresh_token" not in st.session_state:
        st.session_state["manual_refresh_token"] = "initial"
    if st.sidebar.button("Manual refresh", use_container_width=True):
        st.session_state["manual_refresh_token"] = str(time.time())
    auto_bucket = int(time.time() // (refresh_interval_minutes * 60))
    cache_bust = f"{st.session_state['manual_refresh_token']}-{auto_bucket}"
    components.html(
        f"""
        <script>
        window.setTimeout(function() {{
            window.parent.location.reload();
        }}, {refresh_interval_minutes * 60 * 1000});
        </script>
        """,
        height=0,
    )

    st.sidebar.header("Model")
    model_lookback_minutes = st.sidebar.slider(
        "WSPR model lookback",
        min_value=30,
        max_value=180,
        value=MODEL_LOOKBACK_MINUTES_DEFAULT,
        step=15,
        format="%d min",
    )
    max_distance_km = st.sidebar.slider(
        "Max contact distance",
        min_value=500,
        max_value=MAX_MODEL_DISTANCE_KM,
        value=DEFAULT_MAX_CONTACT_DISTANCE_KM,
        step=500,
        format="%d km",
    )

    st.sidebar.header("Bands")
    selected_bands = []
    for band in WSPR_BANDS:
        if st.sidebar.checkbox(
            BAND_LABELS[band],
            value=band in DEFAULT_BANDS,
            key=f"band_{band}",
        ):
            selected_bands.append(band)

    return (
        operator_call,
        home_grid,
        home_coords,
        refresh_interval_minutes,
        model_lookback_minutes,
        max_distance_km,
        tuple(selected_bands),
        cache_bust,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    (
        operator_call,
        home_grid,
        home_coords,
        refresh_interval_minutes,
        model_lookback_minutes,
        max_distance_km,
        selected_bands,
        cache_bust,
    ) = sidebar_controls()

    st.title(APP_TITLE)
    context_parts = []
    if operator_call:
        context_parts.append(f"operator {operator_call}")
    if home_grid:
        context_parts.append(f"home grid {home_grid}")
    context_parts.append(f"WSPR lookback {model_lookback_minutes} min")
    context_parts.append(f"browser refresh every {refresh_interval_minutes} min")
    st.caption(" | ".join(context_parts))

    if not selected_bands:
        st.warning("Select at least one band in the sidebar.")
        st.stop()
    if home_coords is None:
        st.warning("Enter a valid 4-character home grid to build the local opportunity model.")
        st.stop()
    local_grids = set(nearby_maidenhead4(home_grid, radius=2))

    with st.spinner("Fetching WSPR and PSKReporter data..."):
        wspr_loaded, wspr_warnings, wspr_fetched_at, wspr_sources = fetch_wspr_dataset(
            selected_bands,
            count=WSPR_ROWS_PER_BAND,
            lookback_minutes=model_lookback_minutes,
            cache_bust=cache_bust,
        )
        psk_reports, psk_warning, psk_fetched_at = fetch_psk_reporter(cache_bust)

    wspr_spots = prepare_wspr_spots(
        wspr_loaded,
        selected_bands,
        model_lookback_minutes,
        max_distance_km,
    )
    opportunities, tx_activity, rx_activity = build_wspr_opportunities(
        wspr_spots,
        home_coords,
        local_grids,
        max_distance_km,
    )
    propagation_surface = build_propagation_surface(opportunities)
    distance_profile = build_distance_profile(opportunities)
    psk_stations = build_psk_active_stations(
        psk_reports,
        selected_bands,
        home_coords,
        max_distance_km,
    )
    band_summary, direction_summary, scored_stations = score_psk_stations(
        psk_stations,
        opportunities,
    )

    top_cols = st.columns(4)
    top_cols[0].metric("Last WSPR fetch", utc_label(wspr_fetched_at))
    top_cols[1].metric("WSPR spots loaded", f"{len(wspr_loaded):,}")
    top_cols[2].metric("WSPR opportunities", f"{len(opportunities):,}")
    top_cols[3].metric("PSK active stations", f"{len(psk_stations):,}")

    if wspr_sources:
        source_text = ", ".join(
            f"{source}: {count:,}" for source, count in sorted(wspr_sources.items())
        )
        st.caption(f"WSPR spot source: {source_text}.")
    st.caption(
        "WSPR is used as an active-station opportunity model: local TX/RX stations "
        f"in the 5x5 Maidenhead grid block centered on {home_grid} "
        f"({', '.join(sorted(local_grids))}) are crossed with active remote "
        "WSPR stations on the same band/time bucket; observed spots are successes. "
        "PSKReporter supplies the current FT8 station field to score expected points.",
    )
    st.caption(
        "Important limitation: public spot data can infer only stations observed in "
        "at least one WSPR decode during the lookback. Completely silent or never-decoded "
        "stations are not observable in this denominator.",
    )
    for warning in wspr_warnings:
        st.warning(warning)
    if psk_warning:
        st.warning(psk_warning)
    if not psk_reports.empty:
        st.caption(f"PSKReporter fetch: {utc_label(psk_fetched_at)}.")

    if opportunities.empty:
        st.warning(
            "No local WSPR opportunities were built. Extend the WSPR lookback "
            "or select bands with active WSPR stations near home.",
        )

    tab_band, tab_direction, tab_evidence = st.tabs(
        [
            "Band Comparison",
            "Band Direction",
            "Evidence & Maps",
        ],
    )

    with tab_band:
        if band_summary.empty:
            st.warning("No PSKReporter active stations are available for scoring.")
        else:
            st.dataframe(
                style_band_summary(band_summary),
                hide_index=True,
                use_container_width=True,
            )
            st.plotly_chart(expected_points_chart(band_summary), use_container_width=True)
        if distance_profile.empty:
            st.warning("No WSPR distance profile is available.")
        else:
            y_axis = st.radio(
                "Distance profile y-axis",
                ["P(contact)", "E[points]"],
                horizontal=True,
                key="band_distance_profile_y_axis",
            )
            st.plotly_chart(
                distance_profile_chart(distance_profile, y_axis),
                use_container_width=True,
            )
            st.caption(
                "Distance profile is aggregated over all azimuth sectors. "
                "`E[points]` is distance-bin midpoint times `P(contact)`, i.e. expected points for one available station at that distance.",
            )

    with tab_direction:
        has_direction_rows = not direction_summary.empty and "band_m" in direction_summary
        available_bands = [
            band
            for band in selected_bands
            if (
                not propagation_surface[propagation_surface["band_m"] == band].empty
                or (
                    has_direction_rows
                    and not direction_summary[direction_summary["band_m"] == band].empty
                )
            )
        ]
        if not available_bands:
            st.warning("No band-specific direction evidence is available.")
        else:
            default_band = available_bands[0]
            if not band_summary.empty:
                ranked_bands = band_summary.sort_values("rank")["band_m"].tolist()
                default_band = next(
                    (band for band in ranked_bands if band in available_bands),
                    available_bands[0],
                )
            selected_direction_band = st.selectbox(
                "Band",
                available_bands,
                index=available_bands.index(default_band),
                format_func=lambda value: BAND_LABELS[value],
                key="direction_band",
            )
            projection_view = st.radio(
                "Propagation map projection",
                ["Globe", "World map"],
                horizontal=True,
                key="global_propagation_projection",
            )
            st.plotly_chart(
                global_propagation_map(
                    propagation_surface,
                    psk_stations,
                    selected_direction_band,
                    home_coords,
                    home_grid,
                    max_distance_km,
                    projection_view,
                ),
                use_container_width=True,
            )
            st.caption(
                "The global map projects the selected band's smoothed WSPR "
                "distance/azimuth probability surface onto latitude and longitude, "
                "then weights it by a PSKReporter station-density layer. "
                "It estimates expected contact strength where stations actually exist.",
            )
            st.plotly_chart(
                propagation_heatmap(propagation_surface, selected_direction_band),
                use_container_width=True,
            )
            st.plotly_chart(
                band_direction_chart(direction_summary, selected_direction_band),
                use_container_width=True,
            )
            st.plotly_chart(
                evidence_chart(propagation_surface, selected_direction_band),
                use_container_width=True,
            )
            band_direction_rows = (
                direction_summary[direction_summary["band_m"] == selected_direction_band].copy()
                if has_direction_rows
                else pd.DataFrame()
            )
            if band_direction_rows.empty:
                st.warning(f"No PSKReporter azimuth score for {BAND_LABELS[selected_direction_band]}.")
            else:
                st.subheader(f"{BAND_LABELS[selected_direction_band]} Direction Scores")
                st.dataframe(
                    band_direction_rows[
                        [
                            "Azimuth",
                            "active_psk_stations",
                            "mean_p_contact",
                            "expected_points",
                        ]
                    ].rename(
                        columns={
                            "active_psk_stations": "PSK Stations",
                            "mean_p_contact": "Mean P(contact)",
                            "expected_points": "E[Points]",
                        },
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

    with tab_evidence:
        left, right = st.columns(2)
        local_tx = tx_activity[tx_activity["grid4"].isin(local_grids)]
        local_rx = rx_activity[rx_activity["grid4"].isin(local_grids)]
        left.metric("Local active WSPR TX calls", f"{local_tx['callsign'].nunique() if not local_tx.empty else 0:,}")
        right.metric("Local active WSPR RX calls", f"{local_rx['callsign'].nunique() if not local_rx.empty else 0:,}")
        st.caption("Local propagation grids: " + ", ".join(sorted(local_grids)))

        if not propagation_surface.empty:
            st.subheader("WSPR Opportunity Bins")
            st.dataframe(
                propagation_surface.sort_values(
                    ["band_m", "distance_bin_km", "azimuth_bin_deg"],
                ),
                hide_index=True,
                use_container_width=True,
            )

        if not scored_stations.empty:
            st.subheader("PSKReporter Station Map")
            st.plotly_chart(
                psk_station_map(scored_stations, home_coords, home_grid),
                use_container_width=True,
            )
            with st.expander("Scored PSKReporter stations"):
                st.dataframe(
                    scored_stations[
                        [
                            "Band",
                            "callsign",
                            "grid",
                            "distance_home_km",
                            "Azimuth",
                            "p_contact",
                            "probability_source",
                            "expected_points",
                        ]
                    ].sort_values("expected_points", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
