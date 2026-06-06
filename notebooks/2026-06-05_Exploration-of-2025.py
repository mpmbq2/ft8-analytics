import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import gcsfs
    import google.auth
    import google.auth.transport.requests

    return gcsfs, google, mo, pl


@app.cell
def _(gcsfs, google, mo):
    def _get_fs():
        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        return gcsfs.GCSFileSystem(project="mccabemp", token=creds.token)


    fs = _get_fs()
    with fs.open("gs://project-forest-chicken/ft8-analytics/N9QY.txt", "r") as f:
        raw_lines = f.readlines()

    mo.md(f"Loaded **{len(raw_lines)}** lines")
    return (raw_lines,)


@app.cell
def _(mo, pl, raw_lines):
    qso_lines = [l.strip() for l in raw_lines if l.startswith("QSO:")]

    records = []
    for line in qso_lines:
        parts = line.split()
        # QSO: freq mode date time sent_call sent_grid rcvd_call rcvd_grid mult
        records.append(
            {
                "freq_khz": int(parts[1]),
                "mode": parts[2],
                "date": parts[3],
                "time": parts[4],
                "sent_call": parts[5],
                "sent_grid": parts[6],
                "rcvd_call": parts[7],
                "rcvd_grid": parts[8],
                "new_mult": int(parts[9]),
            }
        )

    df = pl.DataFrame(records).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("time").str.zfill(4).str.strptime(pl.Time, "%H%M"),
    )

    mo.md(f"**{len(df)}** QSOs parsed")
    return (df,)


@app.cell
def _(df, mo, pl):
    band_map = {
        1800: "160m",
        3500: "80m",
        7000: "40m",
        14000: "20m",
        21000: "15m",
        28000: "10m",
    }

    df_bands = df.with_columns(
        pl.col("freq_khz")
        .map_elements(
            lambda f: next(
                (v for k, v in sorted(band_map.items(), reverse=True) if f >= k), "other"
            ),
            return_dtype=pl.String,
        )
        .alias("band")
    )

    summary = (
        df_bands.group_by("band")
        .agg(
            pl.len().alias("qsos"),
            pl.col("new_mult").sum().alias("multipliers"),
        )
        .sort("qsos", descending=True)
    )

    mo.vstack(
        [
            mo.md("## QSOs by Band"),
            mo.ui.table(summary),
        ]
    )
    return (df_bands,)


@app.cell
def _(df_bands, mo, pl):
    import math
    import datetime as dt


    def grid_to_latlon(grid):
        g = grid.upper()
        lon = float((ord(g[0]) - ord("A")) * 20 - 180 + (ord(g[2]) - ord("0")) * 2 + 1)
        lat = float((ord(g[1]) - ord("A")) * 10 - 90 + (ord(g[3]) - ord("0")) * 1) + 0.5
        return lat, lon


    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        p = math.pi / 180
        a = (
            math.sin((lat2 - lat1) * p / 2) ** 2
            + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
        )
        return 2 * R * math.asin(math.sqrt(a))


    home_lat, home_lon = grid_to_latlon("EM49")

    df_full = df_bands.with_columns(
        (pl.col("date").cast(pl.Utf8) + "T" + pl.col("time").cast(pl.Utf8))
        .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S")
        .alias("datetime"),
    ).with_columns(
        pl.struct(["rcvd_grid"])
        .map_elements(
            lambda r: haversine_km(home_lat, home_lon, *grid_to_latlon(r["rcvd_grid"])),
            return_dtype=pl.Float64,
        )
        .alias("dist_km")
    )

    mo.md(
        f"Max distance: **{df_full['dist_km'].max():.0f} km** | Mean: **{df_full['dist_km'].mean():.0f} km**"
    )
    return df_full, dt, grid_to_latlon, home_lat, home_lon


@app.cell
def _(df_full, mo, pl):
    import altair as alt

    rate_df = (
        df_full.with_columns((pl.col("datetime").dt.truncate("30m")).alias("window"))
        .group_by(["window", "band"])
        .agg(pl.len().alias("qsos"))
        .sort("window")
        .with_columns(pl.col("window").cast(pl.Utf8))
    )

    band_order = ["10m", "15m", "20m", "40m", "80m"]

    chart_rate = (
        alt.Chart(rate_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "window:N",
                title="Time (UTC)",
                axis=alt.Axis(labelAngle=-45, labelOverlap=True),
            ),
            y=alt.Y("qsos:Q", title="QSOs"),
            color=alt.Color("band:N", sort=band_order, title="Band"),
            order=alt.Order("band:N", sort="ascending"),
            tooltip=["window", "band", "qsos"],
        )
        .properties(title="QSO Rate by Band (30-min windows)", width=700, height=300)
    )

    mo.vstack([mo.md("## Rate Analysis"), chart_rate])
    return alt, band_order


@app.cell
def _(alt, band_order, df_full, mo, pl):
    dist_pdf = df_full.select(["dist_km", "band"])

    chart_dist = (
        alt.Chart(dist_pdf)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("dist_km:Q", bin=alt.Bin(maxbins=40), title="Distance (km)"),
            y=alt.Y("count():Q", title="QSOs"),
            color=alt.Color("band:N", sort=band_order, title="Band"),
            tooltip=["band", "count()"],
        )
        .properties(title="Contact Distance Distribution", width=700, height=300)
    )

    stats = df_full.select(
        [
            pl.col("dist_km").mean().alias("mean_km"),
            pl.col("dist_km").median().alias("median_km"),
            pl.col("dist_km").max().alias("max_km"),
            pl.col("dist_km").min().alias("min_km"),
        ]
    ).row(0, named=True)

    mo.vstack(
        [
            mo.md("## Distance Analysis"),
            mo.md(
                f"Min: **{stats['min_km']:.0f} km** | Median: **{stats['median_km']:.0f} km** | Mean: **{stats['mean_km']:.0f} km** | Max: **{stats['max_km']:.0f} km**"
            ),
            chart_dist,
        ]
    )
    return


@app.cell
def _(band_order, df_full, grid_to_latlon, home_lat, home_lon, mo, pl):
    import plotly.graph_objects as go

    unique_grids = (
        df_full.group_by("rcvd_grid")
        .agg(pl.len().alias("qsos"), pl.col("band").first())
        .with_columns(
            pl.col("rcvd_grid")
            .map_elements(lambda g: grid_to_latlon(g)[0], return_dtype=pl.Float64)
            .alias("lat"),
            pl.col("rcvd_grid")
            .map_elements(lambda g: grid_to_latlon(g)[1], return_dtype=pl.Float64)
            .alias("lon"),
        )
    )

    fig = go.Figure()

    for band in band_order:
        sub = unique_grids.filter(pl.col("band") == band)
        if len(sub) == 0:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=sub["lon"].to_list(),
                lat=sub["lat"].to_list(),
                text=sub["rcvd_grid"].to_list(),
                mode="markers",
                marker=dict(size=6, opacity=0.8),
                name=band,
            )
        )

    # Home station
    fig.add_trace(
        go.Scattergeo(
            lon=[home_lon],
            lat=[home_lat],
            text=["N9QY (EM49)"],
            mode="markers",
            marker=dict(size=12, color="black", symbol="star"),
            name="N9QY",
        )
    )

    fig.update_layout(
        title=f"Unique Grids Worked ({unique_grids.height} grids)",
        geo=dict(
            scope="world",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="lightblue",
            center=dict(lon=-90, lat=35),
            projection_scale=2,
        ),
        height=500,
    )

    mo.vstack([mo.md("## Multipliers Map"), fig])
    return


@app.cell
def _(alt, band_order, df_full, mo, pl):
    timeline_df = (
        df_full.sort("datetime")
        .with_columns(pl.col("band").shift(1).alias("prev_band"))
        .filter(pl.col("band").ne(pl.col("prev_band")) | pl.col("prev_band").is_null())
        .select(["datetime", "band"])
        .with_columns(pl.col("datetime").cast(pl.Utf8))
    )

    rows = timeline_df.to_dicts()
    max_dt = str(df_full["datetime"].max())
    segments = []
    for i, row in enumerate(rows):
        end = rows[i + 1]["datetime"] if i + 1 < len(rows) else max_dt
        segments.append({"band": row["band"], "start": row["datetime"], "end": end})

    seg_df = pl.DataFrame(segments)

    chart_timeline = (
        alt.Chart(seg_df)
        .mark_bar(height=30)
        .encode(
            x=alt.X("start:T", title="Time (UTC)"),
            x2="end:T",
            y=alt.Y("band:N", sort=band_order, title="Band"),
            color=alt.Color("band:N", sort=band_order, legend=None),
            tooltip=["band", "start", "end"],
        )
        .properties(title="Band Activity Timeline", width=700, height=200)
    )

    mo.vstack([mo.md("## Band Transitions"), chart_timeline])
    return


@app.cell
def _(alt, band_order, df_full, mo, pl):
    cumulative_df = (
        df_full.sort("datetime")
        .with_columns(pl.col("dist_km").cum_sum().alias("cumulative_km"))
        .select(["datetime", "band", "rcvd_call", "rcvd_grid", "dist_km", "cumulative_km"])
        .with_columns(pl.col("datetime").cast(pl.Utf8))
    )

    chart_cumulative = (
        alt.Chart(cumulative_df)
        .mark_point()
        .encode(
            x=alt.X("datetime:T", title="Time (UTC)"),
            y=alt.Y("cumulative_km:Q", title="Cumulative Distance (km)"),
            color=alt.Color("band:N", sort=band_order, title="Band"),
            tooltip=[
                "datetime:T",
                "rcvd_call:N",
                "rcvd_grid:N",
                alt.Tooltip("dist_km:Q", format=".0f", title="QSO km"),
                alt.Tooltip("cumulative_km:Q", format=".0f", title="Total km"),
            ],
        )
        .properties(title="Cumulative Distance Worked Over Time", width=700, height=300)
    )

    windowed_df = (
        df_full.sort("datetime")
        .select(["datetime", "band", "dist_km"])
        .with_columns(pl.col("datetime").dt.truncate("30m"))
        .group_by(["datetime", "band"])
        .agg(
            pl.col("dist_km").sum().alias("km_added"),
            pl.len().alias("qsos"),
        )
        .with_columns((pl.col("km_added") / pl.col("qsos")).alias("km_per_qso"))
        .sort("datetime")
        .with_columns(pl.col("datetime").cast(pl.Utf8))
    )

    chart_dist_rate = (
        alt.Chart(windowed_df)
        .mark_bar()
        .encode(
            x=alt.X("datetime:T", title="Time (UTC)"),
            y=alt.Y("km_added:Q", title="km added"),
            color=alt.Color("band:N", sort=band_order, title="Band"),
            order=alt.Order("band:N", sort="ascending"),
            tooltip=[
                "datetime:T",
                "band:N",
                alt.Tooltip("km_added:Q", format=".0f", title="km"),
            ],
        )
        .properties(title="Distance Addition Rate (30-min windows)", width=700, height=200)
    )

    chart_km_per_qso = (
        alt.Chart(windowed_df)
        .mark_bar()
        .encode(
            x=alt.X("datetime:T", title="Time (UTC)"),
            y=alt.Y("km_per_qso:Q", title="km / QSO"),
            color=alt.Color("band:N", sort=band_order, title="Band"),
            order=alt.Order("band:N", sort="ascending"),
            tooltip=[
                "datetime:T",
                "band:N",
                alt.Tooltip("km_per_qso:Q", format=".0f", title="km/QSO"),
                alt.Tooltip("qsos:Q", title="QSOs"),
            ],
        )
        .properties(title="Distance per QSO (30-min windows)", width=700, height=200)
    )

    total_km = df_full["dist_km"].sum()
    mo.vstack(
        [
            mo.md(
                f"## Cumulative Distance\nTotal: **{total_km:,.0f} km** across {len(df_full)} QSOs"
            ),
            chart_cumulative,
            chart_dist_rate,
            chart_km_per_qso,
        ]
    )
    return


@app.cell
def _(alt, band_order, df_full, mo, pl):
    ranked_df = (
        df_full.sort("dist_km", descending=True)
        .with_row_index("rank")
        .select(["rank", "dist_km", "band", "rcvd_call", "rcvd_grid"])
        .with_columns((pl.col("rank") + 1).alias("rank"))
    )

    # Ranked dot plot
    chart_ranked = (
        alt.Chart(ranked_df)
        .mark_point()
        .encode(
            x=alt.X("rank:Q", title="QSO rank (1 = longest)"),
            y=alt.Y("dist_km:Q", title="Distance (km)"),
            color=alt.Color("band:N", sort=band_order, title="Band"),
            tooltip=[
                "rank:Q",
                "rcvd_call:N",
                "rcvd_grid:N",
                "band:N",
                alt.Tooltip("dist_km:Q", format=".0f", title="km"),
            ],
        )
        .properties(title="QSOs Ranked by Distance", width=700, height=300)
    )

    # Cumulative share of total distance
    total = ranked_df["dist_km"].sum()
    cumulative_share_df = ranked_df.sort("dist_km", descending=True).with_columns(
        (pl.col("dist_km").cum_sum() / total * 100).alias("pct_of_total")
    )

    chart_cumshare = (
        alt.Chart(cumulative_share_df)
        .mark_line()
        .encode(
            x=alt.X("rank:Q", title="Number of QSOs (longest first)"),
            y=alt.Y("pct_of_total:Q", title="% of total distance"),
            tooltip=[
                "rank:Q",
                alt.Tooltip("pct_of_total:Q", format=".1f", title="% total"),
                "rcvd_call:N",
                alt.Tooltip("dist_km:Q", format=".0f", title="km"),
            ],
        )
        .properties(title="Cumulative Share of Total Distance", width=700, height=250)
    )

    # Top 15 table
    top15 = ranked_df.head(15)

    # How many QSOs make up 50% of distance?
    n50 = cumulative_share_df.filter(pl.col("pct_of_total") <= 50).height
    n10pct = round(len(ranked_df) * 0.1)
    top10pct_share = cumulative_share_df.head(n10pct)["pct_of_total"].max()

    mo.vstack(
        [
            mo.md(
                f"## Distance Distribution\nTop **{n50} QSOs** account for 50% of total distance. "
                f"Top 10% ({n10pct} QSOs) account for **{top10pct_share:.1f}%** of distance."
            ),
            chart_ranked,
            chart_cumshare,
            mo.md("### Top 15 longest QSOs"),
            mo.ui.table(top15),
        ]
    )
    return


@app.cell
def _(alt, band_order, df_full, mo):
    kde_df = df_full.select(["dist_km", "band"])

    chart_kde = (
        alt.Chart(kde_df)
        .transform_density(
            "dist_km",
            as_=["dist_km", "density"],
            groupby=["band"],
            extent=[0, df_full["dist_km"].max() * 1.05],
        )
        .mark_line()
        .encode(
            x=alt.X("dist_km:Q", title="Distance (km)"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("band:N", sort=band_order, title="Band"),
            tooltip=["band:N", alt.Tooltip("dist_km:Q", format=".0f", title="km")],
        )
        .properties(title="KDE of QSO Distance by Band", width=700, height=350)
    )

    mo.vstack([mo.md("## Distance Distribution by Band (KDE)"), chart_kde])
    return


@app.cell
def _(band_order, df_full, mo):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io

    violin_df = df_full.select(["dist_km", "band"])

    violin_fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(
        data=violin_df,
        x="band",
        y="dist_km",
        order=band_order,
        hue="band",
        hue_order=band_order,
        palette="tab10",
        inner="quart",
        cut=0,
        ax=ax,
    )
    ax.set_xlabel("Band", fontsize=12)
    ax.set_ylabel("Distance (km)", fontsize=12)
    ax.set_title("QSO Distance by Band", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    violin_fig.tight_layout()

    buf = io.BytesIO()
    violin_fig.savefig(buf, format="png", dpi=150)
    plt.close(violin_fig)
    buf.seek(0)

    mo.vstack([mo.md("## Violin Plot: Distance by Band"), mo.image(buf.getvalue())])
    return io, plt, sns


@app.cell
def _(alt, df_full, grid_to_latlon, io, mo, pl, plt, sns):
    region_order = [
        "North America",
        "South America",
        "Europe",
        "Africa",
        "Asia / Australia",
    ]


    def classify_region(lat, lon):
        if lon < -30:
            return "North America" if lat > 10 else "South America"
        elif lon <= 55 and lat >= 35:
            return "Europe"
        elif lon <= 55 and lat < 35:
            return "Africa"
        else:
            return "Asia / Australia"


    df_region = df_full.with_columns(
        pl.struct(["rcvd_grid"])
        .map_elements(
            lambda r: classify_region(*grid_to_latlon(r["rcvd_grid"])),
            return_dtype=pl.String,
        )
        .alias("region")
    )

    # ── 1. Summary table + bar chart ─────────────────────────────────────────────
    region_summary = (
        df_region.group_by("region")
        .agg(
            pl.len().alias("qsos"),
            pl.col("dist_km").mean().alias("mean_dist_km"),
            pl.col("dist_km").median().alias("median_dist_km"),
            pl.col("dist_km").max().alias("max_dist_km"),
        )
        .sort("qsos", descending=True)
    )

    reg_bar = (
        alt.Chart(region_summary)
        .mark_bar()
        .encode(
            x=alt.X("region:N", sort="-y", title=None),
            y=alt.Y("qsos:Q", title="QSOs"),
            color=alt.Color("region:N", sort=region_order, legend=None),
            tooltip=[
                "region:N",
                "qsos:Q",
                alt.Tooltip("mean_dist_km:Q", format=".0f", title="Mean km"),
                alt.Tooltip("median_dist_km:Q", format=".0f", title="Median km"),
                alt.Tooltip("max_dist_km:Q", format=".0f", title="Max km"),
            ],
        )
        .properties(title="QSOs by Region", width=380, height=260)
    )

    # ── 2. Distance violin by region ──────────────────────────────────────────────
    reg_violin_df = df_region.select(["dist_km", "region"])

    reg_fig, reg_ax = plt.subplots(figsize=(10, 4))
    sns.violinplot(
        data=reg_violin_df,
        x="region",
        y="dist_km",
        order=region_order,
        hue="region",
        hue_order=region_order,
        palette="Set2",
        inner="quart",
        cut=0,
        ax=reg_ax,
    )
    reg_ax.set_xlabel(None)
    reg_ax.set_ylabel("Distance (km)")
    reg_ax.set_title("Contact Distance by Region")
    reg_ax.spines[["top", "right"]].set_visible(False)
    reg_fig.tight_layout()

    reg_buf = io.BytesIO()
    reg_fig.savefig(reg_buf, format="png", dpi=150)
    plt.close(reg_fig)
    reg_buf.seek(0)

    # ── 3. Contacts by region over time (hourly) ──────────────────────────────────
    reg_time_df = (
        df_region.select(["datetime", "region"])
        .with_columns(pl.col("datetime").dt.truncate("1h"))
        .group_by(["datetime", "region"])
        .agg(pl.len().alias("qsos"))
        .sort("datetime")
        .with_columns(pl.col("datetime").cast(pl.Utf8))
    )

    reg_time_chart = (
        alt.Chart(reg_time_df)
        .mark_bar()
        .encode(
            x=alt.X("datetime:T", title="Time (UTC)"),
            y=alt.Y("qsos:Q", title="QSOs"),
            color=alt.Color("region:N", sort=region_order, title="Region"),
            order=alt.Order("region:N", sort="ascending"),
            tooltip=["datetime:T", "region:N", "qsos:Q"],
        )
        .properties(
            title="Contacts by Region over Time (1-hour windows)", width=700, height=280
        )
    )

    mo.vstack(
        [
            mo.md("## Stats by Region"),
            mo.hstack([reg_bar, mo.ui.table(region_summary)]),
            mo.image(reg_buf.getvalue()),
            reg_time_chart,
        ]
    )
    return df_region, region_order


@app.cell
def _(alt, band_order, df_region, mo, pl, region_order):
    band_region_df = (
        df_region.group_by(["band", "region"])
        .agg(pl.len().alias("qsos"))
        .sort("band")
        .with_columns(pl.col("band").cast(pl.Utf8))
    )

    chart_abs = (
        alt.Chart(band_region_df)
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=band_order, title="Band"),
            y=alt.Y("qsos:Q", title="QSOs"),
            color=alt.Color("region:N", sort=region_order, title="Region"),
            order=alt.Order("region:N", sort="ascending"),
            tooltip=["band:N", "region:N", "qsos:Q"],
        )
        .properties(title="Count", width=280, height=300)
    )

    chart_norm = (
        alt.Chart(band_region_df)
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=band_order, title="Band"),
            y=alt.Y(
                "qsos:Q", stack="normalize", title="Proportion", axis=alt.Axis(format="%")
            ),
            color=alt.Color("region:N", sort=region_order, legend=None),
            order=alt.Order("region:N", sort="ascending"),
            tooltip=["band:N", "region:N", "qsos:Q"],
        )
        .properties(title="Proportion", width=280, height=300)
    )

    mo.vstack(
        [
            mo.md("## Region by Band"),
            (chart_abs | chart_norm).resolve_scale(color="shared"),
        ]
    )
    return


@app.cell
def _(alt, band_order, df_full, df_region, dt, mo, pl, region_order):
    contest_start = df_full["datetime"].min()


    def elapsed_hours(col):
        return ((pl.col(col) - contest_start).dt.total_seconds() / 3600).cast(pl.Int32)


    # ── 1. km/QSO heatmap: elapsed hour × band ───────────────────────────────────
    strat_heatmap_df = (
        df_region.with_columns(elapsed_hours("datetime").alias("hour"))
        .group_by(["hour", "band"])
        .agg(
            (pl.col("dist_km").sum() / pl.len()).alias("km_per_qso"),
            pl.len().alias("qsos"),
        )
        .sort("hour")
    )

    chart_heatmap = (
        alt.Chart(strat_heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", axis=alt.Axis(labels=False, title=None)),
            y=alt.Y("band:N", sort=band_order, title="Band"),
            color=alt.Color(
                "km_per_qso:Q", title="km / QSO", scale=alt.Scale(scheme="viridis")
            ),
            tooltip=[
                "band:N",
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("km_per_qso:Q", format=".0f", title="km/QSO"),
                alt.Tooltip("qsos:Q", title="QSOs"),
            ],
        )
        .properties(
            title="km/QSO by band and contest hour — empty cells = not operated",
            width=620,
            height=180,
        )
    )

    # ── 2. DX propagation windows: contacts per elapsed hour by region ────────────
    strat_region_time_df = (
        df_region.filter(pl.col("region") != "North America")
        .with_columns(elapsed_hours("datetime").alias("hour"))
        .group_by(["hour", "region"])
        .agg(pl.len().alias("qsos"))
        .sort("hour")
    )

    chart_windows = (
        alt.Chart(strat_region_time_df)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hours since contest start (18:00 UTC Sat)"),
            y=alt.Y("region:N", sort=region_order, title=None),
            color=alt.Color("qsos:Q", title="QSOs", scale=alt.Scale(scheme="oranges")),
            tooltip=[
                "region:N",
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("qsos:Q", title="QSOs"),
            ],
        )
        .properties(
            title="Contact density by region and contest hour — shows propagation windows",
            width=620,
            height=120,
        )
    )

    # ── 3. Band switch quality: km/QSO before vs after each switch ───────────────
    strat_switches = (
        df_full.sort("datetime")
        .with_columns(pl.col("band").shift(1).alias("prev_band"))
        .filter(pl.col("band").ne(pl.col("prev_band")) & pl.col("prev_band").is_not_null())
        .select(["datetime", "band", "prev_band"])
    )

    window = dt.timedelta(minutes=20)
    strat_results = []
    for sw in strat_switches.to_dicts():
        t = sw["datetime"]
        before = df_full.filter(
            (pl.col("datetime") < t) & (pl.col("datetime") >= t - window)
        )
        after = df_full.filter(
            (pl.col("datetime") >= t) & (pl.col("datetime") < t + window)
        )
        if len(before) < 2 or len(after) < 2:
            continue
        strat_results.append(
            {
                "switch_time": str(t),
                "from_band": sw["prev_band"],
                "to_band": sw["band"],
                "before": before["dist_km"].mean(),
                "after": after["dist_km"].mean(),
                "improved": after["dist_km"].mean() > before["dist_km"].mean(),
                "transition": sw["prev_band"] + " → " + sw["band"],
            }
        )

    strat_switch_df = pl.DataFrame(strat_results)
    n_improved = strat_switch_df["improved"].sum()
    n_total = len(strat_switch_df)

    max_val = max(strat_switch_df["before"].max(), strat_switch_df["after"].max()) * 1.05
    diag = pl.DataFrame({"x": [0.0, max_val], "y": [0.0, max_val]})

    chart_switch = (
        alt.Chart(strat_switch_df)
        .mark_point(size=80, filled=True)
        .encode(
            x=alt.X(
                "before:Q",
                title="km/QSO in 20 min before switch",
                scale=alt.Scale(domain=[0, max_val]),
            ),
            y=alt.Y(
                "after:Q",
                title="km/QSO in 20 min after switch",
                scale=alt.Scale(domain=[0, max_val]),
            ),
            color=alt.Color("transition:N", title="Transition"),
            tooltip=[
                "switch_time:N",
                "from_band:N",
                "to_band:N",
                "transition:N",
                alt.Tooltip("before:Q", format=".0f", title="Before km/QSO"),
                alt.Tooltip("after:Q", format=".0f", title="After km/QSO"),
            ],
        )
        .properties(
            title=f"Band switch quality — {n_improved}/{n_total} switches improved km/QSO (above diagonal = better)",
            width=380,
            height=340,
        )
    )

    chart_diag = (
        alt.Chart(diag).mark_line(strokeDash=[4, 4], color="grey").encode(x="x:Q", y="y:Q")
    )

    mo.vstack(
        [
            mo.md("## Band strategy for next year"),
            mo.md("### When was each band productive?"),
            alt.vconcat(chart_heatmap, chart_windows, spacing=0).resolve_scale(
                color="independent"
            ),
            mo.md("### Did band switches pay off?"),
            (chart_switch + chart_diag),
            mo.md(
                f"*{n_improved} of {n_total} switches ({100 * n_improved // n_total}%) improved km/QSO "
                f"in the 20 minutes after switching — points above the diagonal are improvements.*"
            ),
        ]
    )
    return strat_switch_df, strat_switches


@app.cell
def _(alt, band_order, df_full, mo, pl, strat_switch_df, strat_switches):
    # Build segments from all band transitions
    switch_rows_all = (
        df_full.sort("datetime")
        .with_columns(pl.col("band").shift(1).alias("prev_band"))
        .filter(pl.col("band").ne(pl.col("prev_band")) | pl.col("prev_band").is_null())
        .select(["datetime", "band"])
        .to_dicts()
    )

    switch_max_dt = df_full["datetime"].max()
    switch_segs = []
    for switch_idx, switch_row in enumerate(switch_rows_all):
        seg_end = (
            switch_rows_all[switch_idx + 1]["datetime"]
            if switch_idx + 1 < len(switch_rows_all)
            else switch_max_dt
        )
        switch_segs.append(
            {
                "band": switch_row["band"],
                "start": str(switch_row["datetime"]),
                "end": str(seg_end),
                "switch_time": str(switch_row["datetime"]),
            }
        )

    switch_seg_df = pl.DataFrame(switch_segs)

    switch_ratio_df = strat_switch_df.select(
        ["switch_time", "before", "after"]
    ).with_columns(
        (pl.col("after") / pl.col("before")).alias("ratio"),
        (pl.col("after") / pl.col("before")).log(2).alias("log2_ratio"),
    )

    switch_seg_full = switch_seg_df.join(switch_ratio_df, on="switch_time", how="left")

    switch_max_abs = float(switch_seg_full["log2_ratio"].drop_nulls().abs().max())

    chart_switch_timeline = (
        alt.Chart(switch_seg_full)
        .mark_bar(height=30)
        .encode(
            x=alt.X("start:T", title="Time (UTC)"),
            x2="end:T",
            y=alt.Y("band:N", sort=band_order, title="Band"),
            color=alt.condition(
                "datum.log2_ratio !== null",
                alt.Color(
                    "log2_ratio:Q",
                    scale=alt.Scale(
                        scheme="redyellowgreen", domain=[-switch_max_abs, switch_max_abs]
                    ),
                    title="log₂(after / before)",
                ),
                alt.value("lightgrey"),
            ),
            tooltip=[
                "band:N",
                "start:T",
                "end:T",
                alt.Tooltip("ratio:Q", format=".2f", title="km/QSO ratio (after/before)"),
                alt.Tooltip("before:Q", format=".0f", title="Before km/QSO"),
                alt.Tooltip("after:Q", format=".0f", title="After km/QSO"),
            ],
        )
        .properties(
            title="Band activity colored by switch quality (green = improved km/QSO, red = declined)",
            width=700,
            height=200,
        )
    )

    # Build vertical arrow lines connecting from_band → to_band at each transition
    arrow_point_rows = []
    for arr_idx, arr_sw in enumerate(
        strat_switches.with_columns(pl.col("datetime").cast(pl.Utf8)).to_dicts()
    ):
        arrow_point_rows.append(
            {"t_id": str(arr_idx), "time": arr_sw["datetime"], "band": arr_sw["prev_band"]}
        )
        arrow_point_rows.append(
            {"t_id": str(arr_idx), "time": arr_sw["datetime"], "band": arr_sw["band"]}
        )

    arrow_df = pl.DataFrame(arrow_point_rows)

    arrow_layer = (
        alt.Chart(arrow_df)
        .mark_line(color="black", opacity=0.5, strokeWidth=1.5)
        .encode(
            x=alt.X("time:T"),
            y=alt.Y("band:N", sort=band_order),
            detail="t_id:N",
        )
    )

    arrow_heads = (
        alt.Chart(
            arrow_df.filter(
                pl.col("t_id").is_in([str(i) for i in range(len(strat_switches))])
            )
            .group_by("t_id")
            .agg(
                pl.col("time").first(),
                pl.col("band").last(),
            )
        )
        .mark_point(shape="triangle", size=40, filled=True, color="black", opacity=0.6)
        .encode(
            x=alt.X("time:T"),
            y=alt.Y("band:N", sort=band_order),
        )
    )

    mo.vstack(
        [
            mo.md("## Band Switch Quality Timeline"),
            (chart_switch_timeline + arrow_layer + arrow_heads),
            mo.md(
                "*Grey = first segment or too few QSOs to score. Arrows show direction of each band switch. Color shows log₂(km/QSO after ÷ before): green > 1, red < 1.*"
            ),
        ]
    )
    return arrow_heads, arrow_layer, switch_seg_df


@app.cell
def _(
    alt,
    arrow_heads,
    arrow_layer,
    band_order,
    df_full,
    dt,
    mo,
    pl,
    strat_switches,
    switch_seg_df,
):
    # Recompute switch quality using total km instead of mean km/QSO
    totalkm_window = dt.timedelta(minutes=20)
    totalkm_results = []
    for totalkm_sw in strat_switches.with_columns(
        pl.col("datetime").cast(pl.Utf8)
    ).to_dicts():
        totalkm_t_str = totalkm_sw["datetime"][
            :19
        ]  # strip microseconds to match switch_seg_df format
        totalkm_t = df_full.filter(
            pl.col("datetime").cast(pl.Utf8).str.slice(0, 19) == totalkm_t_str
        )["datetime"][0]
        totalkm_before = df_full.filter(
            (pl.col("datetime") < totalkm_t)
            & (pl.col("datetime") >= totalkm_t - totalkm_window)
        )
        totalkm_after = df_full.filter(
            (pl.col("datetime") >= totalkm_t)
            & (pl.col("datetime") < totalkm_t + totalkm_window)
        )
        if len(totalkm_before) < 2 or len(totalkm_after) < 2:
            continue
        totalkm_results.append(
            {
                "switch_time": totalkm_t_str,
                "from_band": totalkm_sw["prev_band"],
                "to_band": totalkm_sw["band"],
                "before_total": totalkm_before["dist_km"].sum(),
                "after_total": totalkm_after["dist_km"].sum(),
            }
        )

    totalkm_switch_df = pl.DataFrame(totalkm_results)

    totalkm_ratio_df = totalkm_switch_df.select(
        ["switch_time", "before_total", "after_total"]
    ).with_columns(
        (pl.col("after_total") / pl.col("before_total")).alias("ratio"),
        (pl.col("after_total") / pl.col("before_total")).log(2).alias("log2_ratio"),
    )

    totalkm_seg_full = switch_seg_df.join(totalkm_ratio_df, on="switch_time", how="left")
    totalkm_max_abs = float(totalkm_seg_full["log2_ratio"].drop_nulls().abs().max() or 1.0)

    chart_totalkm_timeline = (
        alt.Chart(totalkm_seg_full)
        .mark_bar(height=30)
        .encode(
            x=alt.X("start:T", title="Time (UTC)"),
            x2="end:T",
            y=alt.Y("band:N", sort=band_order, title="Band"),
            color=alt.condition(
                "datum.log2_ratio !== null",
                alt.Color(
                    "log2_ratio:Q",
                    scale=alt.Scale(
                        scheme="redyellowgreen", domain=[-totalkm_max_abs, totalkm_max_abs]
                    ),
                    title="log₂(after / before)",
                ),
                alt.value("lightgrey"),
            ),
            tooltip=[
                "band:N",
                "start:T",
                "end:T",
                alt.Tooltip("ratio:Q", format=".2f", title="Total km ratio (after/before)"),
                alt.Tooltip("before_total:Q", format=".0f", title="Before total km"),
                alt.Tooltip("after_total:Q", format=".0f", title="After total km"),
            ],
        )
        .properties(
            title="Band activity colored by total km ratio (green = more total km after switch, red = less)",
            width=700,
            height=200,
        )
    )

    mo.vstack(
        [
            mo.md("## Band Switch Quality Timeline — Total km"),
            (chart_totalkm_timeline + arrow_layer + arrow_heads),
            mo.md(
                "*Grey = first segment or too few QSOs to score. Color shows log₂(total km after ÷ before): green > 1, red < 1.*"
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
