# ARRL Digital Contest Band Advisor

Streamlit dashboard for estimating current HF propagation from Quincy, Illinois
for an FT8/FT4 contest.

Run from the repository root:

```bash
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

With uv:

```bash
uv run --with-requirements dashboard/requirements.txt streamlit run dashboard/app.py
```

## Model

The app estimates `P(contact | band, distance, azimuth, now)` with a WSPR
opportunity model.

1. Fetch recent WSPR spots for 160, 80, 40, 30, 20, 17, 15, 12, and 10 meters.
   The app tries the documented WSPRnet JSON endpoint first and falls back to
   WSPR.live when WSPRnet rejects unauthenticated public access.
2. Infer active WSPR transmitters and active WSPR receivers by band and
   15-minute time bucket.
3. Select local WSPR stations in a 5x5 block of 4-character Maidenhead squares
   centered on the home grid (`EM49` by default for Quincy), using +/-2
   grid-square offsets in latitude and longitude. The dashboard displays the
   exact 25 local propagation grids it generated.
4. Build local-to-remote opportunities: local receiver vs active remote
   transmitter and local transmitter vs active remote receiver. Observed WSPR
   spots are successes.
5. Aggregate successes/exposures by band, 500 km distance bin, and 30 degree
   azimuth sector. The displayed probability is a beta-smoothed binomial
   estimate, not a log-normal fit.
6. Fetch PSKReporter FT8 reception reports and score current active stations
   from the exact home-grid center using the WSPR-derived probability surface.

The first tab, `Band Comparison`, is for choosing the band. It shows the
expected-points band ranking and a distance-profile line chart with one line per
selected band. The y-axis can be switched between `P(contact)` and `E[points]`,
where `E[points]` is the distance-bin midpoint multiplied by the WSPR-derived
contact probability for one available station at that distance.

The second tab, `Band Direction`, is for choosing where to point once a band is
selected. It shows a global propagation map for the selected band by projecting
the smoothed WSPR distance/azimuth probability surface from the exact home-grid
center onto a latitude/longitude grid and weighting it with a PSKReporter
station-density layer. It also shows the selected band's raw
distance/azimuth probability heatmap, expected points by azimuth sector, and the
underlying direction score table.

Expected points are computed as:

```text
sum over active PSKReporter stations:
    distance_from_home_km * P_wspr(contact | band, distance_bin, azimuth_bin)
```

## Caveat

Public spot data can infer only stations observed in at least one WSPR decode
during the lookback. Completely silent or never-decoded stations are not visible
in this denominator.
