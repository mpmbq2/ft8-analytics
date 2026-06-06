# Quincy WSPR Band Conditions

Standalone Streamlit dashboard for current WSPR conditions near Quincy, IL.

Run from the repository root:

```bash
uv run --with streamlit streamlit run dashboard/app.py
```

Or install the standalone requirements:

```bash
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The dashboard queries WSPR.live only when the refresh button is clicked. The default view checks the last 20 minutes of WSPR-2 spots touching the Quincy-adjacent 4-character Maidenhead grids.

## Scoring

The dashboard uses three complementary measures:

- Reach + confidence: separates observed reach from amount of independent evidence.
- Activity-normalized openness: weights unique paths over 500, 1000, and 3000 km, divided by active local WSPR stations.
- Historical percentile: compares the current normalized score with cached hourly observations for the same UTC hour over the last month.

Use **Update month baseline cache** in the sidebar to populate or refresh the local cache in `dashboard/cache/`. Current-condition refreshes read that cache from disk when it exists; they do not rebuild it automatically.
