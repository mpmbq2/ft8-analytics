# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Analysis of FT8 data (particularly ham radio contest data) shared via a Quarto website. Exploratory analysis lives in marimo notebooks; finished analyses are converted into Quarto reports for the website.

## Environment

Dependencies are managed with `uv`. Always use `uv run` to execute Python tools within the project environment.

## Common Commands

```bash
# Open a marimo notebook for exploratory analysis
uv run marimo edit notebooks/<notebook>.py

# Preview the Quarto website locally
quarto preview

# Render the Quarto website
quarto render
```

## Architecture

```
notebooks/         # Marimo notebooks — exploratory canvas, human/AI collaboration
  __marimo__/      # Marimo app state/cache (auto-generated, don't edit)
*.qmd              # Quarto pages — finished analyses and website content
_quarto.yml        # Quarto site config (navbar, theme, format)
pyproject.toml     # Python deps (pandas, polars, altair, plotly, marimo, etc.)
```

## Workflow

1. **Explore** in a marimo notebook (`notebooks/YYYY-MM-DD_Description.py`)
2. **Publish** by converting notebook findings into a `.qmd` file and wiring it into `_quarto.yml`

Notebook filenames follow the convention `YYYY-MM-DD_Short-Description.py`.

## Data

Data is stored in Google Cloud Storage: `gs://project-forest-chicken/ft8-analytics/` (GCP project `mccabemp`).

Currently available: `N9QY.txt`

Access via `gcloud storage cp` or the `gcsfs`/`fsspec` libraries.

## Data Stack

- **DataFrames**: polars (preferred for performance) or pandas
- **Visualization**: altair, plotly, plotnine, matplotlib/seaborn
- **Stats**: statsmodels, numpy
