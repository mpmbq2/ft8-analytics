import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import urllib.parse
    import urllib.request
    import json
    import time
    from datetime import UTC, datetime, timedelta

    return alt, json, mo, pd, time, urllib


@app.cell
def _(mo):
    mo.md("""
    # WSPR data access

    This notebook starts with data access, then builds toward average propagation behavior by band and time of day.

    Primary data source: [WSPR.live](https://wspr.live/), which exposes a public read-only ClickHouse HTTP interface over the `wspr.rx` spot table. The service asks users to keep queries fair; queries should filter by `time` and `band`, and aggregate in ClickHouse when possible instead of downloading raw history.

    Backup/source-of-record path: monthly WSPRnet archive files are available at `https://www.wsprnet.org/archive/`, but for this analysis the ClickHouse endpoint is better because it can aggregate billions of spots server-side.
    """)
    return


@app.cell
def _(json, pd, time, urllib):
    WSPR_LIVE_ENDPOINT = "https://db1.wspr.live/"
    WSPRNET_ARCHIVE = "https://www.wsprnet.org/archive/"
    _wsprlive_request_clock = {"last": 0.0}


    def wsprlive_get(query: str, *, timeout: int = 60, cooldown: float = 5.2) -> dict:
        """Run a read-only ClickHouse query against WSPR.live and return JSON."""
        elapsed = time.monotonic() - _wsprlive_request_clock["last"]
        if elapsed < cooldown:
            time.sleep(cooldown - elapsed)

        clean_query = query.strip().rstrip(";")
        if " format " not in clean_query.lower():
            clean_query = f"{clean_query} FORMAT JSON"
        url = WSPR_LIVE_ENDPOINT + "?" + urllib.parse.urlencode({"query": clean_query})
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        _wsprlive_request_clock["last"] = time.monotonic()
        return payload


    def wsprlive_df(
        query: str, *, timeout: int = 60, cooldown: float = 5.2
    ) -> pd.DataFrame:
        """Run a query and return the `data` payload as a pandas DataFrame."""
        return pd.DataFrame(wsprlive_get(query, timeout=timeout, cooldown=cooldown)["data"])

    return (wsprlive_df,)


@app.cell
def _(pd):
    bands = pd.DataFrame(
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

    hf_bands = [3, 7, 10, 14, 18, 21, 24, 28]
    bands
    return bands, hf_bands


@app.cell
def _(wsprlive_df):
    sample_spots = wsprlive_df(
        """
        SELECT
            id,
            time,
            band,
            rx_sign,
            rx_loc,
            tx_sign,
            tx_loc,
            distance,
            frequency,
            power,
            snr,
            drift,
            code
        FROM wspr.rx
        WHERE time >= toDateTime('2026-06-01 00:00:00')
          AND time < toDateTime('2026-06-01 00:10:00')
          AND band = 14
        ORDER BY time, id
        LIMIT 20
        """
    )

    sample_spots
    return


@app.cell
def _(bands, hf_bands, wsprlive_df):
    analysis_start = "2026-06-01 00:00:00"
    analysis_end = "2026-06-05 00:00:00"

    hourly_band_query = f"""
    SELECT
        band,
        toHour(time) AS utc_hour,
        count() AS spots,
        countDistinct(tx_sign) AS transmitters,
        countDistinct(rx_sign) AS receivers,
        avg(distance) AS mean_distance_km,
        quantile(0.5)(distance) AS median_distance_km,
        quantile(0.9)(distance) AS p90_distance_km,
        avg(snr) AS mean_snr_db
    FROM wspr.rx
    WHERE time >= toDateTime('{analysis_start}')
      AND time < toDateTime('{analysis_end}')
      AND band IN ({", ".join(str(band) for band in hf_bands)})
      AND distance > 0
      AND code = 1
    GROUP BY band, utc_hour
    ORDER BY band, utc_hour
    """

    hourly_band_summary = (
        wsprlive_df(hourly_band_query, timeout=120)
        .merge(bands[["band", "display"]], on="band", how="left")
        .sort_values(["band", "utc_hour"])
    )

    hourly_band_summary
    return analysis_end, analysis_start, hourly_band_summary


@app.cell
def _(alt, analysis_end, analysis_start, hourly_band_summary):
    spot_count_chart = (
        alt.Chart(hourly_band_summary)
        .mark_line(point=True)
        .encode(
            x=alt.X("utc_hour:O", title="UTC hour"),
            y=alt.Y("spots:Q", title="spots"),
            color=alt.Color("display:N", title="band"),
            tooltip=["display", "utc_hour", "spots", "transmitters", "receivers"],
        )
        .properties(
            title=f"WSPR spots by band and UTC hour, {analysis_start[:10]} to {analysis_end[:10]}"
        )
    )

    spot_count_chart
    return


@app.cell
def _(alt, hourly_band_summary):
    distance_chart = (
        alt.Chart(hourly_band_summary)
        .mark_line(point=True)
        .encode(
            x=alt.X("utc_hour:O", title="UTC hour"),
            y=alt.Y("median_distance_km:Q", title="median distance (km)"),
            color=alt.Color("display:N", title="band"),
            tooltip=[
                "display",
                "utc_hour",
                "median_distance_km",
                "p90_distance_km",
                "mean_snr_db",
            ],
        )
        .properties(title="Median WSPR path distance by band and UTC hour")
    )

    distance_chart
    return


@app.cell
def _(mo):
    mo.md(f"""
    ## First access result

    The notebook can query WSPR data. The `sample_spots` table pulls raw rows from `wspr.rx`, and `hourly_band_summary` performs the first server-side aggregation for HF bands.

    Caveat for the propagation question: raw spot counts mix propagation with human/station activity. For a better average propagation signal, we should compare normalized metrics such as distance quantiles, SNR distributions, active receiver/transmitter counts, or path availability for stable station pairs, and we should aggregate over many days/months with solar/seasonal controls.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Toward average propagation by time of day

    The next cells move beyond raw spot counts. They aggregate completed UTC days, normalize by active stations and paths, add receiver-local-hour summaries, and then estimate hourly availability for stable TX/RX paths.

    Filters used here are intentionally conservative: WSPR-2 only (`code = 1`), non-zero plausible paths (`10 <= distance <= 20000` km), populated locators, and plausible SNR values.
    """)
    return


@app.cell
def _(hf_bands, pd):
    avg_start = "2026-05-06 00:00:00"
    avg_end = "2026-06-05 00:00:00"
    hf_band_sql = ", ".join(str(band) for band in hf_bands)
    clean_spot_filter = """
      AND distance BETWEEN 10 AND 20000
      AND code = 1
      AND snr BETWEEN -40 AND 40
      AND tx_loc != ''
      AND rx_loc != ''
    """

    avg_window_days = (pd.Timestamp(avg_end) - pd.Timestamp(avg_start)).days
    avg_window_days
    return avg_end, avg_start, avg_window_days, clean_spot_filter, hf_band_sql


@app.cell
def _(avg_end, avg_start, bands, clean_spot_filter, hf_band_sql, wsprlive_df):
    daily_hourly_query = f"""
    SELECT
        band,
        toDate(time) AS spot_date,
        toHour(time) AS utc_hour,
        count() AS spots,
        countDistinct(tx_sign) AS transmitters,
        countDistinct(rx_sign) AS receivers,
        countDistinct(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc)) AS paths,
        avg(distance) AS mean_distance_km,
        quantile(0.5)(distance) AS median_distance_km,
        quantile(0.9)(distance) AS p90_distance_km,
        avg(snr) AS mean_snr_db,
        quantile(0.5)(snr) AS median_snr_db,
        countIf(distance >= 1000) / count() AS frac_over_1000km,
        countIf(distance >= 3000) / count() AS frac_over_3000km
    FROM wspr.rx
    WHERE time >= toDateTime('{avg_start}')
      AND time < toDateTime('{avg_end}')
      AND band IN ({hf_band_sql})
    {clean_spot_filter}
    GROUP BY band, spot_date, utc_hour
    ORDER BY band, spot_date, utc_hour
    """

    daily_hourly_band = (
        wsprlive_df(daily_hourly_query, timeout=180)
        .merge(bands[["band", "display"]], on="band", how="left")
        .sort_values(["band", "spot_date", "utc_hour"])
    )

    daily_hourly_band = daily_hourly_band.assign(
        spots_per_receiver=lambda df: df["spots"] / df["receivers"],
        spots_per_transmitter=lambda df: df["spots"] / df["transmitters"],
        spots_per_path=lambda df: df["spots"] / df["paths"],
    )

    daily_hourly_band
    return (daily_hourly_band,)


@app.cell
def _(daily_hourly_band):
    hourly_avg = (
        daily_hourly_band.groupby(["band", "display", "utc_hour"], as_index=False)
        .agg(
            n_days=("spot_date", "nunique"),
            mean_daily_spots=("spots", "mean"),
            sd_daily_spots=("spots", "std"),
            mean_receivers=("receivers", "mean"),
            mean_transmitters=("transmitters", "mean"),
            mean_paths=("paths", "mean"),
            avg_spots_per_receiver=("spots_per_receiver", "mean"),
            avg_spots_per_transmitter=("spots_per_transmitter", "mean"),
            avg_spots_per_path=("spots_per_path", "mean"),
            avg_mean_distance_km=("mean_distance_km", "mean"),
            avg_median_distance_km=("median_distance_km", "mean"),
            avg_p90_distance_km=("p90_distance_km", "mean"),
            avg_mean_snr_db=("mean_snr_db", "mean"),
            avg_median_snr_db=("median_snr_db", "mean"),
            avg_frac_over_1000km=("frac_over_1000km", "mean"),
            avg_frac_over_3000km=("frac_over_3000km", "mean"),
        )
        .sort_values(["band", "utc_hour"])
    )

    hourly_avg["sem_daily_spots"] = hourly_avg["sd_daily_spots"] / (
        hourly_avg["n_days"] ** 0.5
    )
    hourly_avg
    return (hourly_avg,)


@app.cell
def _(alt, avg_end, avg_start, hourly_avg):
    hourly_distance_avg_chart = (
        alt.Chart(hourly_avg)
        .mark_line(point=True)
        .encode(
            x=alt.X("utc_hour:O", title="UTC hour"),
            y=alt.Y("avg_median_distance_km:Q", title="avg daily median distance (km)"),
            color=alt.Color("display:N", title="band"),
            tooltip=[
                "display",
                "utc_hour",
                "n_days",
                "avg_median_distance_km",
                "avg_p90_distance_km",
            ],
        )
        .properties(
            title=f"Average daily median path distance, {avg_start[:10]} to {avg_end[:10]}"
        )
    )

    long_path_fraction_chart = (
        alt.Chart(hourly_avg)
        .mark_line(point=True)
        .encode(
            x=alt.X("utc_hour:O", title="UTC hour"),
            y=alt.Y(
                "avg_frac_over_3000km:Q",
                title="fraction of spots over 3000 km",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("display:N", title="band"),
            tooltip=["display", "utc_hour", "avg_frac_over_1000km", "avg_frac_over_3000km"],
        )
        .properties(title="Long-path fraction by UTC hour")
    )

    hourly_distance_avg_chart & long_path_fraction_chart
    return


@app.cell
def _(alt, hourly_avg):
    activity_normalized_chart = (
        alt.Chart(hourly_avg)
        .mark_line(point=True)
        .encode(
            x=alt.X("utc_hour:O", title="UTC hour"),
            y=alt.Y("avg_spots_per_path:Q", title="avg spots per active path"),
            color=alt.Color("display:N", title="band"),
            tooltip=[
                "display",
                "utc_hour",
                "avg_spots_per_path",
                "mean_paths",
                "mean_receivers",
                "mean_transmitters",
            ],
        )
        .properties(title="Activity-normalized spot density by UTC hour")
    )

    activity_normalized_chart
    return


@app.cell
def _(avg_end, avg_start, bands, clean_spot_filter, hf_band_sql, wsprlive_df):
    receiver_local_hour_query = f"""
    SELECT
        band,
        positiveModulo(toHour(time) + toInt32(round(rx_lon / 15)), 24) AS rx_local_hour,
        count() AS spots,
        countDistinct(rx_sign) AS receivers,
        countDistinct(tx_sign) AS transmitters,
        countDistinct(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc)) AS paths,
        quantile(0.5)(distance) AS median_distance_km,
        quantile(0.9)(distance) AS p90_distance_km,
        avg(snr) AS mean_snr_db,
        countIf(distance >= 1000) / count() AS frac_over_1000km,
        countIf(distance >= 3000) / count() AS frac_over_3000km
    FROM wspr.rx
    WHERE time >= toDateTime('{avg_start}')
      AND time < toDateTime('{avg_end}')
      AND band IN ({hf_band_sql})
    {clean_spot_filter}
    GROUP BY band, rx_local_hour
    ORDER BY band, rx_local_hour
    """

    receiver_local_hour = (
        wsprlive_df(receiver_local_hour_query, timeout=180)
        .merge(bands[["band", "display"]], on="band", how="left")
        .sort_values(["band", "rx_local_hour"])
    )
    receiver_local_hour["spots_per_path"] = (
        receiver_local_hour["spots"] / receiver_local_hour["paths"]
    )
    receiver_local_hour
    return (receiver_local_hour,)


@app.cell
def _(alt, receiver_local_hour):
    local_hour_distance_chart = (
        alt.Chart(receiver_local_hour)
        .mark_line(point=True)
        .encode(
            x=alt.X("rx_local_hour:O", title="receiver local hour, approximate solar time"),
            y=alt.Y("median_distance_km:Q", title="median distance (km)"),
            color=alt.Color("display:N", title="band"),
            tooltip=[
                "display",
                "rx_local_hour",
                "median_distance_km",
                "p90_distance_km",
                "frac_over_3000km",
            ],
        )
        .properties(title="Path distance by receiver local hour")
    )

    local_hour_long_path_chart = (
        alt.Chart(receiver_local_hour)
        .mark_line(point=True)
        .encode(
            x=alt.X("rx_local_hour:O", title="receiver local hour, approximate solar time"),
            y=alt.Y(
                "frac_over_3000km:Q",
                title="fraction over 3000 km",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("display:N", title="band"),
            tooltip=["display", "rx_local_hour", "spots", "paths", "frac_over_3000km"],
        )
        .properties(title="Long-path fraction by receiver local hour")
    )

    local_hour_distance_chart & local_hour_long_path_chart
    return


@app.cell
def _(bands, clean_spot_filter, hf_bands, pd, wsprlive_df):
    stable_start = "2026-05-22 00:00:00"
    stable_end = "2026-06-05 00:00:00"
    stable_window_days = (pd.Timestamp(stable_end) - pd.Timestamp(stable_start)).days
    stable_min_days = 5
    stable_min_hours = 8
    stable_min_spots = 30


    def stable_catalog_query_for_band(selected_band: int) -> str:
        return f"""
    SELECT
        band,
        count() AS stable_paths,
        avg(path_distance_km) AS mean_path_distance_km,
        quantile(0.5)(path_distance_km) AS median_path_distance_km,
        quantile(0.9)(path_distance_km) AS p90_path_distance_km,
        avg(active_days) AS mean_active_days,
        avg(active_hours) AS mean_active_hours,
        avg(spots) AS mean_spots_per_path
    FROM
    (
        SELECT
            band,
            concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc) AS path,
            avg(distance) AS path_distance_km,
            uniqExact(toDate(time)) AS active_days,
            uniqExact(toHour(time)) AS active_hours,
            count() AS spots
        FROM wspr.rx
        WHERE time >= toDateTime('{stable_start}')
          AND time < toDateTime('{stable_end}')
          AND band = {selected_band}
    {clean_spot_filter}
        GROUP BY band, path
        HAVING active_days >= {stable_min_days}
           AND active_hours >= {stable_min_hours}
           AND spots >= {stable_min_spots}
    )
    GROUP BY band
    ORDER BY band
    """


    stable_path_catalog_frames = []
    stable_path_catalog_errors = {}
    for _band in hf_bands:
        try:
            stable_path_catalog_frames.append(
                wsprlive_df(stable_catalog_query_for_band(_band), timeout=180)
            )
        except Exception as exc:
            stable_path_catalog_errors[_band] = repr(exc)

    stable_path_catalog = pd.concat(stable_path_catalog_frames, ignore_index=True)
    stable_path_catalog = stable_path_catalog.merge(
        bands[["band", "display"]],
        on="band",
        how="left",
    )
    stable_path_catalog
    return (
        stable_end,
        stable_min_days,
        stable_min_hours,
        stable_min_spots,
        stable_path_catalog,
        stable_start,
        stable_window_days,
    )


@app.cell
def _(
    clean_spot_filter,
    hf_bands,
    pd,
    stable_end,
    stable_min_days,
    stable_min_hours,
    stable_min_spots,
    stable_path_catalog,
    stable_start,
    stable_window_days,
    wsprlive_df,
):
    def stable_hourly_query_for_band(selected_band: int) -> str:
        return f"""
    SELECT
        p.band AS band,
        p.utc_hour AS utc_hour,
        count() AS observed_path_days,
        uniqExact(p.path) AS paths_observed,
        avg(p.distance_km) AS mean_distance_km,
        quantile(0.5)(p.distance_km) AS median_distance_km,
        quantile(0.9)(p.distance_km) AS p90_distance_km,
        avg(p.mean_snr_db) AS mean_snr_db,
        avg(p.spots) AS mean_spots_when_observed
    FROM
    (
        SELECT
            band,
            toDate(time) AS spot_date,
            toHour(time) AS utc_hour,
            concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc) AS path,
            avg(distance) AS distance_km,
            avg(snr) AS mean_snr_db,
            count() AS spots
        FROM wspr.rx
        WHERE time >= toDateTime('{stable_start}')
          AND time < toDateTime('{stable_end}')
          AND band = {selected_band}
    {clean_spot_filter}
        GROUP BY band, spot_date, utc_hour, path
    ) AS p
    ANY INNER JOIN
    (
        SELECT
            band,
            concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc) AS path
        FROM wspr.rx
        WHERE time >= toDateTime('{stable_start}')
          AND time < toDateTime('{stable_end}')
          AND band = {selected_band}
    {clean_spot_filter}
        GROUP BY band, path
        HAVING uniqExact(toDate(time)) >= {stable_min_days}
           AND uniqExact(toHour(time)) >= {stable_min_hours}
           AND count() >= {stable_min_spots}
    ) AS s
    USING (band, path)
    GROUP BY p.band, p.utc_hour
    ORDER BY p.band, p.utc_hour
    """


    stable_path_hourly_frames = []
    stable_path_query_errors = {}
    for _band in hf_bands:
        try:
            stable_path_hourly_frames.append(
                wsprlive_df(stable_hourly_query_for_band(_band), timeout=240)
            )
        except Exception as exc:
            stable_path_query_errors[_band] = repr(exc)

    stable_path_hourly = pd.concat(stable_path_hourly_frames, ignore_index=True)
    stable_path_hourly = stable_path_hourly.merge(
        stable_path_catalog[["band", "display", "stable_paths"]],
        on="band",
        how="left",
    )
    stable_path_hourly["observed_fraction"] = stable_path_hourly["observed_path_days"] / (
        stable_path_hourly["stable_paths"] * stable_window_days
    )

    stable_path_hourly
    return (stable_path_hourly,)


@app.cell
def _(alt, stable_path_hourly):
    stable_availability_chart = (
        alt.Chart(stable_path_hourly)
        .mark_line(point=True)
        .encode(
            x=alt.X("utc_hour:O", title="UTC hour"),
            y=alt.Y(
                "observed_fraction:Q",
                title="observed stable path-days / possible",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("display:N", title="band"),
            tooltip=[
                "display",
                "utc_hour",
                "observed_fraction",
                "paths_observed",
                "stable_paths",
            ],
        )
        .properties(title="Stable path availability by UTC hour")
    )

    stable_path_distance_chart = (
        alt.Chart(stable_path_hourly)
        .mark_line(point=True)
        .encode(
            x=alt.X("utc_hour:O", title="UTC hour"),
            y=alt.Y("median_distance_km:Q", title="median stable-path distance (km)"),
            color=alt.Color("display:N", title="band"),
            tooltip=[
                "display",
                "utc_hour",
                "median_distance_km",
                "p90_distance_km",
                "mean_snr_db",
            ],
        )
        .properties(title="Stable path distance by UTC hour")
    )

    stable_availability_chart & stable_path_distance_chart
    return


@app.cell
def _(
    avg_window_days,
    mo,
    stable_min_days,
    stable_min_hours,
    stable_min_spots,
    stable_window_days,
):
    mo.md(f"""
    ## What these views mean

    `hourly_avg` is the main average-by-hour table for the full `{avg_window_days}` completed-day window. It averages each band/hour across days and includes activity-normalized columns such as spots per receiver and spots per active path.

    `receiver_local_hour` shifts each spot into the receiver's approximate local solar hour. That is usually more meaningful than UTC hour for day/night propagation questions.

    `stable_path_hourly` restricts to paths that appeared on at least `{stable_min_days}` of the `{stable_window_days}` completed days, across at least `{stable_min_hours}` UTC hours, with at least `{stable_min_spots}` total spots. Its availability fraction is still not a perfect propagation probability, because transmitters and receivers are not guaranteed to be active continuously, but it is less dominated by one-off activity spikes than raw counts.
    """)
    return


@app.cell
def _(bands):
    QUINCY_IL = {"name": "Quincy, IL", "lat": 39.9356, "lon": -91.4099}


    def maidenhead4(lat: float, lon: float) -> str:
        lon_adjusted = lon + 180
        lat_adjusted = lat + 90
        lon_field = int(lon_adjusted // 20)
        lat_field = int(lat_adjusted // 10)
        lon_square = int((lon_adjusted % 20) // 2)
        lat_square = int(lat_adjusted % 10)
        return f"{chr(65 + lon_field)}{chr(65 + lat_field)}{lon_square}{lat_square}"


    def nearby_maidenhead4(center_grid: str, radius: int = 1) -> list[str]:
        lon_index = (ord(center_grid[0]) - 65) * 10 + int(center_grid[2])
        lat_index = (ord(center_grid[1]) - 65) * 10 + int(center_grid[3])
        grids = []
        for lat_offset in range(-radius, radius + 1):
            for lon_offset in range(-radius, radius + 1):
                candidate_lon = lon_index + lon_offset
                candidate_lat = lat_index + lat_offset
                if not (0 <= candidate_lon < 180 and 0 <= candidate_lat < 180):
                    continue
                grids.append(
                    f"{chr(65 + candidate_lon // 10)}{chr(65 + candidate_lat // 10)}{candidate_lon % 10}{candidate_lat % 10}"
                )
        return sorted(grids)


    quincy_grid4 = maidenhead4(QUINCY_IL["lat"], QUINCY_IL["lon"])
    quincy_grid6_approx = "EM49hw"
    quincy_grid_sets = {
        "core": [quincy_grid4],
        "adjacent": nearby_maidenhead4(quincy_grid4, radius=1),
        "wide": nearby_maidenhead4(quincy_grid4, radius=2),
    }
    quincy_region_options = {
        f"{quincy_grid4} only": "core",
        "Quincy + adjacent 4-char grids": "adjacent",
        "Wider 5x5 grid region": "wide",
    }
    all_wspr_band_sql = ", ".join(str(int(band)) for band in bands["band"].tolist())


    def sql_string_list(values: list[str]) -> str:
        return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


    def condition_label(row) -> str:
        if row["spots"] == 0:
            return "No spots"
        if row["over_3000km"] >= 10 and row["paths"] >= 20:
            return "Strong DX"
        if row["over_3000km"] >= 1:
            return "DX open"
        if row["over_1000km"] >= 5:
            return "Open"
        if row["over_500km"] >= 3:
            return "Regional"
        return "Sparse"


    def condition_rank(label: str) -> int:
        return {
            "No spots": 0,
            "Sparse": 1,
            "Regional": 2,
            "Open": 3,
            "DX open": 4,
            "Strong DX": 5,
        }[label]

    return (
        all_wspr_band_sql,
        condition_label,
        condition_rank,
        quincy_grid6_approx,
        quincy_grid_sets,
        quincy_region_options,
        sql_string_list,
    )


@app.cell
def _(mo, quincy_grid6_approx, quincy_grid_sets, quincy_region_options):
    current_refresh = mo.ui.run_button(
        label="Refresh current WSPR conditions",
        kind="success",
    )
    current_lookback_minutes = mo.ui.slider(
        5,
        120,
        step=5,
        value=20,
        show_value=True,
        include_input=True,
        label="Lookback minutes",
    )
    quincy_grid_scope = mo.ui.dropdown(
        quincy_region_options,
        value="Quincy + adjacent 4-char grids",
        label="Quincy grid region",
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ## Live Quincy WSPR band conditions

                Quincy is approximately `{quincy_grid6_approx}`; the default region uses `{", ".join(quincy_grid_sets["adjacent"])}`.
                """
            ),
            mo.hstack([current_refresh, current_lookback_minutes, quincy_grid_scope]),
        ]
    )
    return current_lookback_minutes, current_refresh, quincy_grid_scope


@app.cell
def _(
    all_wspr_band_sql,
    alt,
    bands,
    condition_label,
    condition_rank,
    current_lookback_minutes,
    current_refresh,
    mo,
    pd,
    quincy_grid_scope,
    quincy_grid_sets,
    sql_string_list,
    wsprlive_df,
):
    mo.stop(
        not current_refresh.value,
        mo.callout(
            mo.md(
                "Click **Refresh current WSPR conditions** to query WSPR.live for the current window."
            ),
            kind="neutral",
        ),
    )

    current_lookback = int(current_lookback_minutes.value)
    current_grids = quincy_grid_sets[quincy_grid_scope.value]
    current_grid_sql = sql_string_list(current_grids)
    current_band_display = bands[["band", "display"]].copy()

    current_summary_query = f"""
    SELECT
        band,
        count() AS spots,
        countIf(left(rx_loc, 4) IN ({current_grid_sql})) AS inbound_spots,
        countIf(left(tx_loc, 4) IN ({current_grid_sql})) AS outbound_spots,
        countIf(left(rx_loc, 4) IN ({current_grid_sql}) AND left(tx_loc, 4) IN ({current_grid_sql})) AS local_spots,
        uniqExact(tx_sign) AS transmitters,
        uniqExact(rx_sign) AS receivers,
        uniqExactIf(tx_sign, left(tx_loc, 4) IN ({current_grid_sql})) AS local_transmitters,
        uniqExactIf(rx_sign, left(rx_loc, 4) IN ({current_grid_sql})) AS local_receivers,
        uniqExact(concat(tx_sign, '|', rx_sign, '|', tx_loc, '|', rx_loc)) AS paths,
        max(distance) AS max_distance_km,
        quantile(0.5)(distance) AS median_distance_km,
        quantile(0.9)(distance) AS p90_distance_km,
        avg(snr) AS mean_snr_db,
        quantile(0.5)(snr) AS median_snr_db,
        countIf(distance >= 500) AS over_500km,
        countIf(distance >= 1000) AS over_1000km,
        countIf(distance >= 3000) AS over_3000km,
        min(time) AS first_spot_utc,
        max(time) AS last_spot_utc
    FROM wspr.rx
    WHERE time >= now() - INTERVAL {current_lookback} MINUTE
      AND time < now()
      AND band IN ({all_wspr_band_sql})
      AND (left(rx_loc, 4) IN ({current_grid_sql}) OR left(tx_loc, 4) IN ({current_grid_sql}))
      AND distance BETWEEN 10 AND 20000
      AND code = 1
      AND snr BETWEEN -40 AND 40
      AND tx_loc != ''
      AND rx_loc != ''
    GROUP BY band
    ORDER BY band
    """

    current_paths_query = f"""
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
                left(tx_loc, 4) IN ({current_grid_sql}) AND left(rx_loc, 4) IN ({current_grid_sql}), 'local',
                left(tx_loc, 4) IN ({current_grid_sql}), 'outbound',
                left(rx_loc, 4) IN ({current_grid_sql}), 'inbound',
                'other'
            ) AS direction
        FROM wspr.rx
        WHERE time >= now() - INTERVAL {current_lookback} MINUTE
          AND time < now()
          AND band IN ({all_wspr_band_sql})
          AND (left(rx_loc, 4) IN ({current_grid_sql}) OR left(tx_loc, 4) IN ({current_grid_sql}))
          AND distance BETWEEN 10 AND 20000
          AND code = 1
          AND snr BETWEEN -40 AND 40
          AND tx_loc != ''
          AND rx_loc != ''
    )
    ORDER BY distance_km DESC, time DESC
    LIMIT 60
    """

    current_raw_summary = wsprlive_df(current_summary_query, timeout=90)
    current_top_paths = wsprlive_df(current_paths_query, timeout=90)
    current_snapshot_utc = pd.Timestamp.now(tz="UTC").floor("s")
    current_window_start_utc = current_snapshot_utc - pd.Timedelta(minutes=current_lookback)

    current_band_conditions = current_band_display.merge(
        current_raw_summary,
        on="band",
        how="left",
    )

    current_numeric_cols = [
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
    ]
    for _column in current_numeric_cols:
        if _column not in current_band_conditions:
            current_band_conditions[_column] = 0
    current_band_conditions[current_numeric_cols] = current_band_conditions[
        current_numeric_cols
    ].fillna(0)

    current_band_conditions["condition"] = current_band_conditions.apply(
        condition_label, axis=1
    )
    current_band_conditions["condition_rank"] = current_band_conditions["condition"].map(
        condition_rank
    )
    current_band_conditions["spots_per_path"] = current_band_conditions.apply(
        lambda row: row["spots"] / row["paths"] if row["paths"] else 0,
        axis=1,
    )
    current_band_conditions["dx_fraction"] = current_band_conditions.apply(
        lambda row: row["over_3000km"] / row["spots"] if row["spots"] else 0,
        axis=1,
    )

    current_table_cols = [
        "display",
        "condition",
        "spots",
        "inbound_spots",
        "outbound_spots",
        "paths",
        "median_distance_km",
        "p90_distance_km",
        "max_distance_km",
        "median_snr_db",
        "over_1000km",
        "over_3000km",
    ]
    current_conditions_table = (
        current_band_conditions[current_table_cols]
        .rename(
            columns={
                "display": "band",
                "inbound_spots": "heard near Quincy",
                "outbound_spots": "sent from near Quincy",
                "median_distance_km": "median km",
                "p90_distance_km": "p90 km",
                "max_distance_km": "max km",
                "median_snr_db": "median SNR",
                "over_1000km": ">=1000 km",
                "over_3000km": ">=3000 km",
            }
        )
        .sort_values(["spots", "max km"], ascending=[False, False])
    )

    nonzero_current = current_band_conditions[current_band_conditions["spots"] > 0]
    current_best_dx = nonzero_current.sort_values(
        ["condition_rank", "over_3000km", "max_distance_km", "spots"],
        ascending=False,
    ).head(1)

    if current_best_dx.empty:
        current_headline = "No regional WSPR spots in the selected window."
    else:
        _best = current_best_dx.iloc[0]
        current_headline = (
            f"{_best['display']} looks best now: {_best['condition']} with "
            f"{int(_best['spots'])} spots, {int(_best['paths'])} paths, "
            f"max {int(_best['max_distance_km'])} km."
        )

    current_direction_counts = current_band_conditions.melt(
        id_vars=["band", "display"],
        value_vars=["inbound_spots", "outbound_spots", "local_spots"],
        var_name="direction",
        value_name="direction_spots",
    )
    current_direction_counts["direction"] = current_direction_counts["direction"].replace(
        {
            "inbound_spots": "heard near Quincy",
            "outbound_spots": "sent from near Quincy",
            "local_spots": "local",
        }
    )

    current_spots_chart = (
        alt.Chart(current_direction_counts[current_direction_counts["direction_spots"] > 0])
        .mark_bar()
        .encode(
            x=alt.X("display:N", title="band", sort=list(bands["display"])),
            y=alt.Y("direction_spots:Q", title="spots"),
            color=alt.Color("direction:N", title="direction"),
            tooltip=["display", "direction", "direction_spots"],
        )
        .properties(title="Regional WSPR spots by band")
    )

    current_distance_chart = (
        alt.Chart(current_band_conditions[current_band_conditions["spots"] > 0])
        .mark_bar()
        .encode(
            x=alt.X("display:N", title="band", sort=list(bands["display"])),
            y=alt.Y("max_distance_km:Q", title="max observed distance (km)"),
            color=alt.Color("condition:N", title="condition"),
            tooltip=[
                "display",
                "condition",
                "spots",
                "paths",
                "max_distance_km",
                "over_3000km",
            ],
        )
        .properties(title="Best observed reach by band")
    )

    current_status = mo.md(
        f"""
        **{current_headline}**

        Window: `{current_window_start_utc:%Y-%m-%d %H:%M:%S}` to `{current_snapshot_utc:%Y-%m-%d %H:%M:%S}` UTC. Region: `{", ".join(current_grids)}`.
        """
    )

    mo.vstack(
        [
            current_status,
            mo.hstack([current_spots_chart, current_distance_chart]),
            mo.md("### Band summary"),
            mo.ui.table(current_conditions_table, pagination=False),
            mo.md("### Longest recent paths touching the Quincy region"),
            mo.ui.table(current_top_paths, pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
