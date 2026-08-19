"""Rebuild data/*.parquet from the source SQLite database.

stops_metric is deliberately omitted: it is 217 MB of the 528 MB source and no
query in dispatch_dashboard.py touches it.
"""
from pathlib import Path

import duckdb

SRC = Path(__file__).parent / "AFC_BSS_Analytics.db"
OUT = Path(__file__).parent / "data"

TABLES = [
    "ticket_sales", "route_kiosks", "travel_time", "siso_files",
    "routes_information", "route_versions", "route_stops",
    "route_distances_tidy", "loaded_files",
]


def main():
    OUT.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{SRC.as_posix()}' AS src (TYPE sqlite, READ_ONLY);")
    for t in TABLES:
        dst = OUT / f"{t}.parquet"
        con.execute(
            f'COPY (SELECT * FROM src."{t}") TO \'{dst.as_posix()}\' '
            "(FORMAT parquet, COMPRESSION zstd);")
        print(f"{dst.stat().st_size / 1e6:8.2f} MB  {t}")


if __name__ == "__main__":
    main()
