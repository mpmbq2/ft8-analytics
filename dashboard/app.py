from __future__ import annotations

import json
import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


WSPR_LIVE_ENDPOINT = "https://db1.wspr.live/"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_VERSION = 1


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    grid6: str


QUINCY_IL = Location(
    name="Quincy, IL",
    lat=39.9356,
    lon=-91.4099,
    grid6="EM49hw",
)


BANDS = pd.DataFrame(
    [
        {"band": -1, "display": "LF", "dial_hz": 136_000},
        {"band": 0, "display": "MF", "dial_hz": 474_200},
        {"band": 1, "display": "160m", "dial_hz": 1_836_600},
        {"band": 3, "display": "80m", "dial_hz": 3_568_600},
        {"band": 5, "display": "60m", "dial_hz": 5_287_200},
        {"band": 7, "display": "40m", "dial_hz": 7_038_600},
        {"band": 10, "display": "30m", "dial_hz": 10_138_700},
        {"band": 14, "display": "20m", "dial_hz": 14_095_600},
        {"band": 18, "display": "17m", "dial_hz": 18_104_600},
        {"band": 21, "display": "15m", "dial_hz": 21_094_600},
        {"band": 24, "display": "12m", "dial_hz": 24_924_600},
        {"band": 28, "display": "10m", "dial_hz": 28_124_600},
        {"band": 50, "display": "6m", "dial_hz": 50_293_000},
        {"band": 70, "display": "4m", "dial_hz": 70_091_000},
        {"band": 144, "display": "2m", "dial_hz": 144_489_000},
        {"band": 432, "display": "70cm", "dial_hz": 432_300_000},
        {"band": 1296, "display": "23cm", "dial_hz": 1_296_500_000},
    ]
)

REACH_RANKS = {
    "No reach observed": 0,
    "Local/sparse": 1,
    "Regional": 2,
    "Open": 3,
    "DX observed": 4,
}

CONFIDENCE_RANKS = {
    "no evidence": 0,
    "low confidence": 1,
    "medium confidence": 2,
    "high confidence": 3,
}


def maidenhead4(lat: float, lon: float) -> str:
    lon_adjusted = lon + 180
    lat_adjusted = lat + 90
    lon_field = int(lon_adjusted // 20)
    lat_field = int(lat_adjusted // 10)
    lon_square = int((lon_adjusted % 20) // 2)
    lat_square = int(lat_adjusted % 10)
    return (
        f"{chr(65 + lon_field)}{chr(65 + lat_field)}"
        f"{lon_square}{lat_square}"
    )


def nearby_maidenhead4(center_grid: str, radius: int = 1) -> list[str]:
    lon_index = (ord(center_grid[0]) - 65) * 10 + int(center_grid[2])
    lat_index = (ord(center_grid[1]) - 65) * 10 + int(center_grid[3])
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


def sql_string_list(values: list[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def cache_key_for_grids(grids: list[str]) -> str:
    digest = hashlib.sha1(",".join(sorted(grids)).encode("utf-8")).hexdigest()[:12]
    return f"grids-{digest}"


def baseline_cache_paths(grids: list[str]) -> tuple[Path, Path]:
    key = cache_key_for_grids(grids)
    return CACHE_DIR / f"wspr_month_baseline_{key}.csv", CACHE_DIR / (
        f"wspr_month_baseline_{key}.json"
    )


def read_baseline_metadata(grids: list[str]) -> dict[str, Any] | None:
    _, metadata_path = baseline_cache_paths(grids)
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_baseline_cache(grids: list[str]) -> pd.DataFrame | None:
    data_path, _ = baseline_cache_paths(grids)
    if not data_path.exists():
        return None
    return pd.read_csv(data_path)


def write_baseline_cache(
    baseline: pd.DataFrame,
    grids: list[str],
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data_path, metadata_path = baseline_cache_paths(grids)
    baseline.to_csv(data_path, index=False)
    metadata = {
        "cache_version": CACHE_VERSION,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "window_start_utc": start_utc.isoformat(),
        "window_end_utc": end_utc.isoformat(),
        "grids": grids,
        "rows": int(len(baseline)),
        "data_path": str(data_path),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def throttle_wsprlive(cooldown_seconds: float = 5.2) -> None:
    last_request = st.session_state.get("_last_wsprlive_request", 0.0)
    elapsed = time.monotonic() - last_request
    if elapsed < cooldown_seconds:
        time.sleep(cooldown_seconds - elapsed)


def wsprlive_df(query: str, *, timeout: int = 90) -> pd.DataFrame:
    throttle_wsprlive()
    clean_query = query.strip().rstrip(";")
    if " format " not in clean_query.lower():
        clean_query = f"{clean_query} FORMAT JSON"
    url = WSPR_LIVE_ENDPOINT + "?" + urllib.parse.urlencode({"query": clean_query})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"WSPR.live HTTP {exc.code}: {body[:1000]}") from exc
    finally:
        st.session_state["_last_wsprlive_request"] = time.monotonic()
    return pd.DataFrame(payload["data"])


def reach_label(row: pd.Series) -> str:
    if row["spots"] == 0:
        return "No reach observed"
    if row["paths_over_3000km"] >= 1 or row["over_3000km"] >= 1:
        return "DX observed"
    if row["paths_over_1000km"] >= 3 or row["over_1000km"] >= 5:
        return "Open"
    if row["paths_over_500km"] >= 3 or row["over_500km"] >= 3:
        return "Regional"
    return "Local/sparse"


def confidence_label(row: pd.Series) -> str:
    if row["spots"] == 0:
        return "no evidence"
    if row["paths"] >= 20 and row["active_local_stations"] >= 3:
        return "high confidence"
    if row["paths"] >= 5 and row["active_local_stations"] >= 2:
        return "medium confidence"
    return "low confidence"


def normalized_openness_score(row: pd.Series) -> float:
    active_local_stations = max(float(row["active_local_stations"]), 1.0)
    weighted_paths = (
        3.0 * float(row["paths_over_3000km"])
        + 1.0 * float(row["paths_over_1000km"])
        + 0.25 * float(row["paths_over_500km"])
    )
    return weighted_paths / active_local_stations


def build_queries(lookback_minutes: int, grids: list[str]) -> tuple[str, str]:
    grid_sql = sql_string_list(grids)
    band_sql = ", ".join(str(int(band)) for band in BANDS["band"].tolist())

    summary_query = f"""
    SELECT
        band,
        count() AS spots,
        countIf(left(rx_loc, 4) IN ({grid_sql})) AS inbound_spots,
        countIf(left(tx_loc, 4) IN ({grid_sql})) AS outbound_spots,
        countIf(left(rx_loc, 4) IN ({grid_sql}) AND left(tx_loc, 4) IN ({grid_sql})) AS local_spots,
        uniqExact(tx_sign) AS transmitters,
        uniqExact(rx_sign) AS receivers,
        uniqExactIf(tx_sign, left(tx_loc, 4) IN ({grid_sql})) AS local_transmitters,
        uniqExactIf(rx_sign, left(rx_loc, 4) IN ({grid_sql})) AS local_receivers,
        uniqExact(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc)) AS paths,
        max(distance) AS max_distance_km,
        quantile(0.5)(distance) AS median_distance_km,
        quantile(0.9)(distance) AS p90_distance_km,
        avg(snr) AS mean_snr_db,
        quantile(0.5)(snr) AS median_snr_db,
        countIf(distance >= 500) AS over_500km,
        countIf(distance >= 1000) AS over_1000km,
        countIf(distance >= 3000) AS over_3000km,
        uniqExactIf(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc), distance >= 500) AS paths_over_500km,
        uniqExactIf(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc), distance >= 1000) AS paths_over_1000km,
        uniqExactIf(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc), distance >= 3000) AS paths_over_3000km,
        min(time) AS first_spot_utc,
        max(time) AS last_spot_utc
    FROM wspr.rx
    WHERE time >= now() - INTERVAL {lookback_minutes} MINUTE
      AND time < now()
      AND band IN ({band_sql})
      AND (left(rx_loc, 4) IN ({grid_sql}) OR left(tx_loc, 4) IN ({grid_sql}))
      AND distance BETWEEN 10 AND 20000
      AND code = 1
      AND snr BETWEEN -40 AND 40
      AND tx_loc != ''
      AND rx_loc != ''
    GROUP BY band
    ORDER BY band
    """

    paths_query = f"""
    SELECT *
    FROM
    (
        SELECT
            time,
            band,
            tx_sign,
            tx_loc,
            rx_sign,
            rx_loc,
            toUInt32(distance) AS distance_km,
            snr,
            power,
            multiIf(
                left(tx_loc, 4) IN ({grid_sql}) AND left(rx_loc, 4) IN ({grid_sql}), 'local',
                left(tx_loc, 4) IN ({grid_sql}), 'outbound',
                left(rx_loc, 4) IN ({grid_sql}), 'inbound',
                'other'
            ) AS direction
        FROM wspr.rx
        WHERE time >= now() - INTERVAL {lookback_minutes} MINUTE
          AND time < now()
          AND band IN ({band_sql})
          AND (left(rx_loc, 4) IN ({grid_sql}) OR left(tx_loc, 4) IN ({grid_sql}))
          AND distance BETWEEN 10 AND 20000
          AND code = 1
          AND snr BETWEEN -40 AND 40
          AND tx_loc != ''
          AND rx_loc != ''
    )
    ORDER BY distance_km DESC, time DESC
    LIMIT 60
    """
    return summary_query, paths_query


def build_baseline_query(
    grids: list[str],
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> str:
    grid_sql = sql_string_list(grids)
    band_sql = ", ".join(str(int(band)) for band in BANDS["band"].tolist())
    start_text = start_utc.strftime("%Y-%m-%d %H:%M:%S")
    end_text = end_utc.strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    SELECT
        band,
        toDate(time) AS spot_date,
        toHour(time) AS utc_hour,
        count() AS spots,
        countIf(left(rx_loc, 4) IN ({grid_sql})) AS inbound_spots,
        countIf(left(tx_loc, 4) IN ({grid_sql})) AS outbound_spots,
        uniqExactIf(tx_sign, left(tx_loc, 4) IN ({grid_sql})) AS local_transmitters,
        uniqExactIf(rx_sign, left(rx_loc, 4) IN ({grid_sql})) AS local_receivers,
        uniqExact(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc)) AS paths,
        max(distance) AS max_distance_km,
        quantile(0.5)(distance) AS median_distance_km,
        quantile(0.9)(distance) AS p90_distance_km,
        avg(snr) AS mean_snr_db,
        quantile(0.5)(snr) AS median_snr_db,
        countIf(distance >= 500) AS over_500km,
        countIf(distance >= 1000) AS over_1000km,
        countIf(distance >= 3000) AS over_3000km,
        uniqExactIf(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc), distance >= 500) AS paths_over_500km,
        uniqExactIf(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc), distance >= 1000) AS paths_over_1000km,
        uniqExactIf(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc), distance >= 3000) AS paths_over_3000km
    FROM wspr.rx
    WHERE time >= toDateTime('{start_text}')
      AND time < toDateTime('{end_text}')
      AND band IN ({band_sql})
      AND (left(rx_loc, 4) IN ({grid_sql}) OR left(tx_loc, 4) IN ({grid_sql}))
      AND distance BETWEEN 10 AND 20000
      AND code = 1
      AND snr BETWEEN -40 AND 40
      AND tx_loc != ''
      AND rx_loc != ''
    GROUP BY band, spot_date, utc_hour
    ORDER BY band, spot_date, utc_hour
    """


def update_baseline_cache(grids: list[str]) -> dict[str, Any]:
    end_utc = pd.Timestamp.now(tz="UTC").floor("h")
    start_utc = end_utc - pd.Timedelta(days=30)
    baseline = wsprlive_df(
        build_baseline_query(grids, start_utc=start_utc, end_utc=end_utc),
        timeout=240,
    )
    return write_baseline_cache(
        baseline,
        grids,
        start_utc=start_utc,
        end_utc=end_utc,
    )


def prepare_baseline_scores(baseline: pd.DataFrame) -> pd.DataFrame:
    scored = baseline.copy()
    numeric_cols = [
        "spots",
        "local_transmitters",
        "local_receivers",
        "paths",
        "max_distance_km",
        "paths_over_500km",
        "paths_over_1000km",
        "paths_over_3000km",
    ]
    for column in numeric_cols:
        if column not in scored:
            scored[column] = 0
    scored[numeric_cols] = scored[numeric_cols].fillna(0)
    scored["active_local_stations"] = (
        scored["local_transmitters"] + scored["local_receivers"]
    )
    scored["normalized_openness_score"] = scored.apply(
        normalized_openness_score,
        axis=1,
    )
    return scored


def add_historical_percentiles(
    band_conditions: pd.DataFrame,
    baseline: pd.DataFrame | None,
    *,
    snapshot_utc: pd.Timestamp,
    lookback_minutes: int,
) -> pd.DataFrame:
    enriched = band_conditions.copy()
    enriched["historical_score_percentile"] = pd.NA
    enriched["historical_reach_percentile"] = pd.NA
    enriched["historical_samples"] = 0
    if baseline is None or baseline.empty:
        return enriched

    scored_baseline = prepare_baseline_scores(baseline)
    same_hour = scored_baseline[scored_baseline["utc_hour"] == snapshot_utc.hour]
    scale_to_hour = 60 / lookback_minutes
    for band in enriched["band"].tolist():
        band_rows = same_hour[same_hour["band"] == band]
        if band_rows.empty:
            continue
        current = enriched.loc[enriched["band"] == band].iloc[0]
        current_hourly_score = current["normalized_openness_score"] * scale_to_hour
        score_percentile = (
            band_rows["normalized_openness_score"] <= current_hourly_score
        ).mean() * 100
        reach_percentile = (
            band_rows["max_distance_km"] <= current["max_distance_km"]
        ).mean() * 100
        mask = enriched["band"] == band
        enriched.loc[mask, "historical_score_percentile"] = score_percentile
        enriched.loc[mask, "historical_reach_percentile"] = reach_percentile
        enriched.loc[mask, "historical_samples"] = len(band_rows)
    return enriched


def build_condition_frame(
    summary: pd.DataFrame,
    *,
    lookback_minutes: int,
    baseline: pd.DataFrame | None,
    snapshot_utc: pd.Timestamp,
) -> pd.DataFrame:
    band_conditions = BANDS[["band", "display"]].merge(
        summary,
        on="band",
        how="left",
    )
    numeric_cols = [
        "spots",
        "inbound_spots",
        "outbound_spots",
        "local_spots",
        "transmitters",
        "receivers",
        "local_transmitters",
        "local_receivers",
        "paths",
        "max_distance_km",
        "median_distance_km",
        "p90_distance_km",
        "mean_snr_db",
        "median_snr_db",
        "over_500km",
        "over_1000km",
        "over_3000km",
        "paths_over_500km",
        "paths_over_1000km",
        "paths_over_3000km",
    ]
    for column in numeric_cols:
        if column not in band_conditions:
            band_conditions[column] = 0
    band_conditions[numeric_cols] = band_conditions[numeric_cols].fillna(0)
    band_conditions["active_local_stations"] = (
        band_conditions["local_transmitters"] + band_conditions["local_receivers"]
    )
    band_conditions["reach"] = band_conditions.apply(reach_label, axis=1)
    band_conditions["reach_rank"] = band_conditions["reach"].map(REACH_RANKS)
    band_conditions["confidence"] = band_conditions.apply(confidence_label, axis=1)
    band_conditions["confidence_rank"] = band_conditions["confidence"].map(
        CONFIDENCE_RANKS
    )
    band_conditions["reach_confidence"] = (
        band_conditions["reach"] + " / " + band_conditions["confidence"]
    )
    band_conditions["spots_per_path"] = band_conditions.apply(
        lambda row: row["spots"] / row["paths"] if row["paths"] else 0,
        axis=1,
    )
    band_conditions["dx_fraction"] = band_conditions.apply(
        lambda row: row["over_3000km"] / row["spots"] if row["spots"] else 0,
        axis=1,
    )
    band_conditions["normalized_openness_score"] = band_conditions.apply(
        normalized_openness_score,
        axis=1,
    )
    band_conditions["normalized_openness_hourly_equiv"] = (
        band_conditions["normalized_openness_score"] * 60 / lookback_minutes
    )
    band_conditions = add_historical_percentiles(
        band_conditions,
        baseline,
        snapshot_utc=snapshot_utc,
        lookback_minutes=lookback_minutes,
    )
    return band_conditions


def band_table(band_conditions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "display",
        "reach",
        "confidence",
        "normalized_openness_score",
        "historical_score_percentile",
        "spots",
        "inbound_spots",
        "outbound_spots",
        "paths",
        "paths_over_3000km",
        "median_distance_km",
        "p90_distance_km",
        "max_distance_km",
        "median_snr_db",
        "over_1000km",
        "over_3000km",
    ]
    return (
        band_conditions[columns]
        .rename(
            columns={
                "display": "band",
                "inbound_spots": "heard near Quincy",
                "outbound_spots": "sent from near Quincy",
                "paths_over_3000km": "DX paths",
                "normalized_openness_score": "norm score",
                "historical_score_percentile": "hist pct",
                "median_distance_km": "median km",
                "p90_distance_km": "p90 km",
                "max_distance_km": "max km",
                "median_snr_db": "median SNR",
                "over_1000km": ">=1000 km",
                "over_3000km": ">=3000 km",
            }
        )
        .sort_values(["spots", "max km"], ascending=[False, False])
        .reset_index(drop=True)
    )


def headline(band_conditions: pd.DataFrame) -> str:
    nonzero = band_conditions[band_conditions["spots"] > 0]
    if nonzero.empty:
        return "No regional WSPR spots in the selected window."
    sortable = nonzero.assign(
        historical_score_sort=nonzero["historical_score_percentile"].fillna(-1)
    )
    best = sortable.sort_values(
        [
            "reach_rank",
            "confidence_rank",
            "historical_score_sort",
            "normalized_openness_score",
            "max_distance_km",
            "spots",
        ],
        ascending=False,
    ).iloc[0]
    percentile = best["historical_score_percentile"]
    percentile_text = (
        ""
        if pd.isna(percentile)
        else f", {float(percentile):.0f}th percentile for this UTC hour"
    )
    return (
        f"{best['display']} looks best now: {best['reach']} "
        f"({best['confidence']}) with "
        f"{int(best['spots'])} spots, {int(best['paths'])} paths, "
        f"max {int(best['max_distance_km'])} km{percentile_text}."
    )


def direction_counts(band_conditions: pd.DataFrame) -> pd.DataFrame:
    counts = band_conditions.melt(
        id_vars=["band", "display"],
        value_vars=["inbound_spots", "outbound_spots", "local_spots"],
        var_name="direction",
        value_name="direction_spots",
    )
    counts["direction"] = counts["direction"].replace(
        {
            "inbound_spots": "heard near Quincy",
            "outbound_spots": "sent from near Quincy",
            "local_spots": "local",
        }
    )
    return counts[counts["direction_spots"] > 0]


def format_integer_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        if column in formatted:
            formatted[column] = formatted[column].round().astype("Int64")
    return formatted


def run_refresh(
    lookback_minutes: int,
    grids: list[str],
    *,
    baseline: pd.DataFrame | None,
) -> dict[str, Any]:
    summary_query, paths_query = build_queries(lookback_minutes, grids)
    summary = wsprlive_df(summary_query, timeout=90)
    top_paths = wsprlive_df(paths_query, timeout=90)
    snapshot_utc = pd.Timestamp.now(tz="UTC").floor("s")
    band_conditions = build_condition_frame(
        summary,
        lookback_minutes=lookback_minutes,
        baseline=baseline,
        snapshot_utc=snapshot_utc,
    )
    return {
        "band_conditions": band_conditions,
        "conditions_table": band_table(band_conditions),
        "top_paths": top_paths,
        "headline": headline(band_conditions),
        "lookback_minutes": lookback_minutes,
        "grids": grids,
        "snapshot_utc": snapshot_utc,
        "window_start_utc": snapshot_utc - pd.Timedelta(minutes=lookback_minutes),
    }


def render_dashboard(result: dict[str, Any]) -> None:
    band_conditions = result["band_conditions"]
    conditions_table = result["conditions_table"]
    top_paths = result["top_paths"]

    st.subheader(result["headline"])
    st.caption(
        "Window: "
        f"{result['window_start_utc']:%Y-%m-%d %H:%M:%S} to "
        f"{result['snapshot_utc']:%Y-%m-%d %H:%M:%S} UTC. "
        f"Region: {', '.join(result['grids'])}."
    )
    with st.expander("How openness is measured"):
        st.markdown(
            """
            - **Reach + confidence** separates what was observed from how much
              independent evidence supports it.
            - **Activity-normalized openness** weights unique paths over 500,
              1000, and 3000 km, then divides by active local WSPR stations.
            - **Historical percentile** compares the current normalized score
              with cached hourly observations for this same UTC hour over the
              last month.
            """
        )

    nonzero = band_conditions[band_conditions["spots"] > 0]
    total_spots = int(nonzero["spots"].sum()) if not nonzero.empty else 0
    total_paths = int(nonzero["paths"].sum()) if not nonzero.empty else 0
    dx_spots = int(nonzero["over_3000km"].sum()) if not nonzero.empty else 0
    dx_paths = int(nonzero["paths_over_3000km"].sum()) if not nonzero.empty else 0
    active_bands = int((band_conditions["spots"] > 0).sum())

    metric_cols = st.columns(4)
    metric_cols[0].metric("Active bands", active_bands)
    metric_cols[1].metric("Regional spots", total_spots)
    metric_cols[2].metric("Unique paths", total_paths)
    metric_cols[3].metric("DX paths", dx_paths, help=f"{dx_spots} spots >=3000 km")

    counts = direction_counts(band_conditions)
    if not counts.empty:
        spots_chart = (
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("display:N", title="band", sort=list(BANDS["display"])),
                y=alt.Y("direction_spots:Q", title="spots"),
                color=alt.Color("direction:N", title="direction"),
                tooltip=["display", "direction", "direction_spots"],
            )
            .properties(height=280)
        )
        distance_chart = (
            alt.Chart(band_conditions[band_conditions["spots"] > 0])
            .mark_bar()
            .encode(
                x=alt.X("display:N", title="band", sort=list(BANDS["display"])),
                y=alt.Y("max_distance_km:Q", title="max observed distance (km)"),
                color=alt.Color("reach:N", title="reach"),
                tooltip=[
                    "display",
                    "reach",
                    "confidence",
                    "spots",
                    "paths",
                    "max_distance_km",
                    "paths_over_3000km",
                ],
            )
            .properties(height=280)
        )
        left, right = st.columns(2)
        left.altair_chart(spots_chart, use_container_width=True)
        right.altair_chart(distance_chart, use_container_width=True)

        score_chart = (
            alt.Chart(band_conditions[band_conditions["spots"] > 0])
            .mark_bar()
            .encode(
                x=alt.X("display:N", title="band", sort=list(BANDS["display"])),
                y=alt.Y(
                    "normalized_openness_score:Q",
                    title="activity-normalized openness",
                ),
                color=alt.Color("confidence:N", title="confidence"),
                tooltip=[
                    "display",
                    "reach",
                    "confidence",
                    "normalized_openness_score",
                    "active_local_stations",
                    "paths_over_1000km",
                    "paths_over_3000km",
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(score_chart, use_container_width=True)

        percentile_rows = band_conditions[
            band_conditions["historical_score_percentile"].notna()
            & (band_conditions["spots"] > 0)
        ]
        if not percentile_rows.empty:
            percentile_chart = (
                alt.Chart(percentile_rows)
                .mark_bar()
                .encode(
                    x=alt.X("display:N", title="band", sort=list(BANDS["display"])),
                    y=alt.Y(
                        "historical_score_percentile:Q",
                        title="historical percentile",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    color=alt.Color("reach:N", title="reach"),
                    tooltip=[
                        "display",
                        "historical_score_percentile",
                        "historical_reach_percentile",
                        "historical_samples",
                        "normalized_openness_hourly_equiv",
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(percentile_chart, use_container_width=True)
        else:
            st.info(
                "Build the month baseline cache from the sidebar to enable "
                "historical percentile scoring."
            )
    else:
        st.info("No regional WSPR spots were found for this window.")

    st.subheader("Band Summary")
    table = format_integer_columns(
        conditions_table,
        [
            "spots",
            "heard near Quincy",
            "sent from near Quincy",
            "paths",
            "DX paths",
            "median km",
            "p90 km",
            "max km",
            ">=1000 km",
            ">=3000 km",
        ],
    )
    for column in ["norm score", "hist pct"]:
        if column in table:
            table[column] = table[column].astype("Float64").round(1)
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Longest Recent Paths Touching The Quincy Region")
    if top_paths.empty:
        st.info("No path details were returned for this window.")
    else:
        path_table = top_paths.merge(
            BANDS[["band", "display"]],
            on="band",
            how="left",
        )
        path_table = path_table[
            [
                "time",
                "display",
                "direction",
                "tx_sign",
                "tx_loc",
                "rx_sign",
                "rx_loc",
                "distance_km",
                "snr",
                "power",
            ]
        ].rename(columns={"display": "band"})
        st.dataframe(path_table, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Quincy WSPR Band Conditions",
        layout="wide",
    )
    st.title("Quincy WSPR Band Conditions")
    st.caption(
        "Button-triggered WSPR.live snapshot for current band conditions near "
        f"{QUINCY_IL.name}."
    )

    quincy_grid4 = maidenhead4(QUINCY_IL.lat, QUINCY_IL.lon)
    grid_sets = {
        f"{quincy_grid4} only": [quincy_grid4],
        "Quincy + adjacent 4-char grids": nearby_maidenhead4(quincy_grid4, radius=1),
        "Wider 5x5 grid region": nearby_maidenhead4(quincy_grid4, radius=2),
    }

    with st.sidebar:
        st.header("Snapshot")
        lookback_minutes = st.slider(
            "Lookback minutes",
            min_value=5,
            max_value=120,
            value=20,
            step=5,
        )
        region_label = st.selectbox(
            "Quincy grid region",
            options=list(grid_sets.keys()),
            index=1,
        )
        selected_grids = grid_sets[region_label]
        st.caption(f"{QUINCY_IL.name} is approximately {QUINCY_IL.grid6}.")
        st.caption("Selected grids: " + ", ".join(selected_grids))
        metadata = read_baseline_metadata(selected_grids)
        if metadata is None:
            st.caption("Historical baseline cache: not built")
        else:
            generated = pd.Timestamp(metadata["generated_at_utc"]).strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            st.caption(
                "Historical baseline cache: "
                f"{metadata['rows']} hourly observations, built {generated}"
            )
        update_cache = st.button("Update month baseline cache")
        refresh = st.button("Refresh current WSPR conditions", type="primary")

    if update_cache:
        with st.spinner("Building local month baseline cache from WSPR.live..."):
            try:
                metadata = update_baseline_cache(selected_grids)
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        st.success(
            "Baseline cache updated: "
            f"{metadata['rows']} hourly observations from "
            f"{metadata['window_start_utc']} to {metadata['window_end_utc']}."
        )

    if refresh:
        with st.spinner("Querying WSPR.live..."):
            try:
                baseline = read_baseline_cache(selected_grids)
                st.session_state["latest_result"] = run_refresh(
                    lookback_minutes,
                    selected_grids,
                    baseline=baseline,
                )
            except Exception as exc:
                st.error(str(exc))
                st.stop()

    result = st.session_state.get("latest_result")
    if result is None:
        st.info("Click refresh to query WSPR.live for the current window.")
        st.stop()

    render_dashboard(result)


if __name__ == "__main__":
    main()
