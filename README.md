# Dispatch Dashboard

Streamlit dashboard over AFC/BSS ticket-sales and route-operations data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dispatch_dashboard.py
```

## Data

Tables live in `data/` as zstd-compressed Parquet and are queried through DuckDB.
`build_data.py` regenerates them from the source `AFC_BSS_Analytics.db`, which is
not tracked in this repo:

```bash
python build_data.py
```
