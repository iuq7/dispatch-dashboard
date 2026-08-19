import io
import json
import math
import datetime as dt
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

DATA_DIR = Path(__file__).parent / "data"
BUS_CAPACITY = 45
TOWARDS_ISB = 0.65
SOC_THRESHOLD_KM = 230.0

sns.set_theme(style="whitegrid", palette="muted")


@st.cache_resource
def get_engine():
    """DuckDB over the Parquet tables in ./data.

    StaticPool keeps every checkout on the one in-memory DuckDB connection, so
    the views registered here stay visible to later queries.
    """
    engine = create_engine("duckdb:///:memory:", poolclass=StaticPool)
    with engine.connect() as conn:
        for pq in sorted(DATA_DIR.glob("*.parquet")):
            conn.execute(text(
                f'CREATE OR REPLACE VIEW "{pq.stem}" AS '
                f"SELECT * FROM read_parquet('{pq.as_posix()}')"))
        conn.commit()
    return engine


def table_exists(conn, name: str) -> bool:
    return not pd.read_sql(
        text("SELECT 1 FROM information_schema.tables WHERE table_name = :n"),
        conn, params={"n": name}).empty


# =====================================================================
# 1. SALES & RIDERSHIP (notebook: static analysis cell)
# =====================================================================

@st.cache_data(show_spinner=False)
def load_sales(start: str, end: str):
    s, e = f"{start} 00:00:00", f"{end} 23:59:59"
    engine = get_engine()
    with engine.connect() as conn:
        # only the three columns the dashboard reads: SELECT t.* pulled 35
        # columns and 1.15 GB for a row count, a mode, and an hourly pivot
        df_join = pd.read_sql(
            text('''
                SELECT t.cleaned_date, t.PRODUCT_DESCRIPTION, r.route_number
                FROM ticket_sales t
                LEFT JOIN route_kiosks r ON t."Terminal No" = r.terminal_no
                WHERE t.cleaned_date BETWEEN :s AND :e
            '''), conn, params={"s": s, "e": e})
        for col in ("PRODUCT_DESCRIPTION", "route_number"):
            df_join[col] = df_join[col].astype("category")

        query_agg = """
            SELECT "{group_col}", PRODUCT_DESCRIPTION,
                   COUNT(*) AS Ridership,
                   SUM(CAST(REPLACE(Amount, ',', '') AS REAL)) AS total_sales
            FROM ticket_sales
            WHERE cleaned_date BETWEEN :s AND :e
            GROUP BY "{group_col}", PRODUCT_DESCRIPTION
        """
        df_terminals = pd.read_sql(text(query_agg.format(group_col="Terminal No")),
                                   conn, params={"s": s, "e": e})
        df_routes = pd.read_sql(
            text(query_agg.format(group_col="route_number").replace(
                'ticket_sales', 'ticket_sales t LEFT JOIN route_kiosks r ON t."Terminal No" = r.terminal_no'
            )), conn, params={"s": s, "e": e})
        df_users = pd.read_sql(text(query_agg.format(group_col="Personnel Name")),
                               conn, params={"s": s, "e": e})

        df_regional = pd.read_sql(text("""
            SELECT "Terminal No" AS Terminal_No,
                   SUM(CASE WHEN CAST(REPLACE(Amount, ',', '') AS INTEGER) = 50 THEN CAST(REPLACE(Amount, ',', '') AS INTEGER) ELSE 0 END) AS total_sales_50,
                   SUM(CASE WHEN CAST(REPLACE(Amount, ',', '') AS INTEGER) = 50 THEN 1 ELSE 0 END) AS count_50,
                   SUM(CASE WHEN CAST(REPLACE(Amount, ',', '') AS INTEGER) = 90 THEN CAST(REPLACE(Amount, ',', '') AS INTEGER) ELSE 0 END) AS total_sales_90,
                   SUM(CASE WHEN CAST(REPLACE(Amount, ',', '') AS INTEGER) = 90 THEN 1 ELSE 0 END) AS count_90,
                   SUM(CASE WHEN CAST(REPLACE(Amount, ',', '') AS INTEGER) NOT IN (50, 90) THEN CAST(REPLACE(Amount, ',', '') AS INTEGER) ELSE 0 END) AS total_sales_other
            FROM ticket_sales
            WHERE "Terminal No" IS NOT NULL AND TRIM(CAST("Terminal No" AS VARCHAR)) <> ''
              AND "Kiosk No" <> 'Total'
              AND cleaned_date BETWEEN :s AND :e
            GROUP BY "Terminal No"
        """), conn, params={"s": s, "e": e})

    return df_join, df_terminals, df_routes, df_users, df_regional


def product_pivot_chart(df, group_col, title):
    if df.empty:
        st.info(f"No data for {title}.")
        return
    pivot = df.pivot_table(index=group_col, columns="PRODUCT_DESCRIPTION",
                           values="total_sales", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", width=0.8, edgecolor="black", linewidth=0.5, ax=ax)
    ax.set_title(title, pad=15)
    ax.set_ylabel("Total Sales")
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Product Description", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(pivot)


def hourly_matrix_from(df_join, product=None):
    if df_join.empty:
        return None, None
    product = product or df_join["PRODUCT_DESCRIPTION"].value_counts().idxmax()
    df_h = df_join[df_join["PRODUCT_DESCRIPTION"] == product].copy()
    df_h["hour"] = pd.to_datetime(df_h["cleaned_date"], errors="coerce").dt.hour
    df_h = df_h.dropna(subset=["hour"])
    hours = list(range(6, 24))
    # route_number is categorical; pin observed=False so a later pandas default
    # flip cannot silently drop unused routes from the matrix
    hm = df_h.pivot_table(index="route_number", columns="hour",
                          values="PRODUCT_DESCRIPTION", aggfunc="count", fill_value=0,
                          observed=False)
    hm = hm[~hm.index.duplicated(keep="first")]
    hm = hm.loc[:, ~hm.columns.duplicated(keep="first")]
    hm = hm.reindex(columns=hours, fill_value=0)
    return hm, product


# =====================================================================
# 2. SUPPLY MATRIX (notebook: build_supply_matrix cell)
# =====================================================================

@st.cache_data(show_spinner=False)
def build_supply_matrix(start: str, end: str, product=None):
    engine = get_engine()
    s, e = f"{start} 00:00:00", f"{end} 23:59:59"
    with engine.connect() as conn:
        if product is None:
            row = pd.read_sql(text("""
                SELECT PRODUCT_DESCRIPTION FROM ticket_sales
                WHERE cleaned_date BETWEEN :s AND :e
                  AND PRODUCT_DESCRIPTION IS NOT NULL
                GROUP BY PRODUCT_DESCRIPTION ORDER BY COUNT(*) DESC LIMIT 1
            """), conn, params={"s": s, "e": e})
            if row.empty:
                return None, None, None, None
            product = row.iloc[0, 0]

        df_filtered = pd.read_sql(text("""
            SELECT t.cleaned_date, r.route_number
            FROM ticket_sales t
            LEFT JOIN route_kiosks r ON t."Terminal No" = r.terminal_no
            WHERE t.cleaned_date BETWEEN :s AND :e
              AND t.PRODUCT_DESCRIPTION = :p
        """), conn, params={"s": s, "e": e, "p": product})

        df_tt = pd.read_sql_query("SELECT route_code, avg_tt_min FROM travel_time", conn)

    df_filtered["hour"] = pd.to_datetime(df_filtered["cleaned_date"], errors="coerce").dt.hour
    df_filtered = df_filtered.dropna(subset=["hour"])

    hours = list(range(6, 24))
    hm = df_filtered.pivot_table(index="route_number", columns="hour",
                                 values="cleaned_date", aggfunc="count", fill_value=0)
    hm = hm[~hm.index.duplicated(keep="first")]
    hm = hm.loc[:, ~hm.columns.duplicated(keep="first")]
    hm = hm.reindex(columns=hours, fill_value=0)

    peak_demand = hm.max(axis=1)
    calculated_buses = np.ceil((peak_demand * TOWARDS_ISB) / BUS_CAPACITY)

    df_tt_grouped = df_tt.groupby("route_code").mean(numeric_only=True)
    hours_required = df_tt_grouped["avg_tt_min"].reindex(calculated_buses.index, fill_value=60) / 60
    total_units = np.ceil(calculated_buses * hours_required)

    supply_matrix = pd.DataFrame({h: total_units for h in hours}, index=calculated_buses.index)
    return supply_matrix, df_tt, hours, product


# =====================================================================
# 3. DISPATCH ENGINE (notebook: integrated dispatch & analysis cell)
# =====================================================================

RAW_DISTANCE_DATA = """Template\tF\t11.68017
FR-01\tF\t22.95645
FR-02\tF\t20.057
FR-03A\tF\t11.68017
FR-04\tF\t13.5403
FR-04A\tF\t2.9796
FR-04B\tF\t9.491
FR-06\tF\t16.07475
FR-07\tF\t11.8395
FR-08A\tF\t18.07435
FR-08B\tF\t14.48
FR-08C\tF\t19.9517
FR-09\tF\t20.67775
FR-10\tF\t36.432
FR-11\tF\t13.4027
FR-12\tF\t17.314
FR-13\tF\t36.11317
FR-14\tF\t22.7193
FR-14A\tF\t6.5733
FR-15\tF\t14.4634
FR-16\tF\t24.42
FR-17\tF\t14.348
ST 01\tF\t13.2022
ST 02\tF\t7.78935
Template\tB\t13.73227
FR-01\tB\t24.56425
FR-02\tB\t19.312
FR-03A\tB\t13.73227
FR-04\tB\t13.7316
FR-04A\tB\t2.977
FR-04B\tB\t0
FR-06\tB\t16.45855
FR-07\tB\t12.7566
FR-08A\tB\t17.8574
FR-08B\tB\t15.21
FR-08C\tB\t17.9875
FR-09\tB\t21.3883
FR-10\tB\t39.939
FR-11\tB\t13.5839
FR-12\tB\t17.353
FR-13\tB\t34.43436
FR-14\tB\t19.18245
FR-14A\tB\t6.5989
FR-15\tB\t16.4475
FR-16\tB\t21.11
FR-17\tB\t13.885
ST 01\tB\t15.91795
ST 02\tB\t10.53395"""


def _distance_lookup():
    df = pd.read_csv(io.StringIO(RAW_DISTANCE_DATA), sep="\t", header=None,
                     names=["Route", "Dir", "Dist"])
    df["Dist"] = pd.to_numeric(df["Dist"], errors="coerce").fillna(0)
    return df


def _hourly_timestamps(supply_matrix, route, hours):
    timestamps, hour_blocks = [], {}
    for hour in hours:
        freq = supply_matrix.loc[route, hour]
        if freq <= 0:
            continue
        headway = timedelta(minutes=max(1, int(60.0 / freq)))
        current = datetime.strptime(f"{hour:02d}:00", "%H:%M")
        end_of_hour = current + timedelta(hours=1)
        while current < end_of_hour:
            timestamps.append(current)
            hour_blocks[current] = f"{hour:02d}:00"
            current += headway
    return timestamps, hour_blocks


@st.cache_data(show_spinner=False)
def run_dispatch(supply_matrix, df_tt, hours):
    """Baseline dispatch engine (Part 1 of the notebook's dispatch cell)."""
    all_trips = []
    df_tt_unique = df_tt.groupby("route_code").mean(numeric_only=True)
    next_bus = 1

    for route in supply_matrix.index:
        tt_min = int(df_tt_unique.loc[route, "avg_tt_min"]) if route in df_tt_unique.index else 60
        travel_time = timedelta(minutes=tt_min)
        timestamps, hour_blocks = _hourly_timestamps(supply_matrix, route, hours)

        pool_A, pool_B, arrivals = [], [], []
        a_count, b_count = 0, 0
        route_trips = []

        for T in sorted(timestamps):
            ready = [arr for arr in arrivals if arr[0] < T]
            arrivals = [arr for arr in arrivals if arr[0] >= T]
            ready.sort(key=lambda x: x[0])
            for _, term, b_id in ready:
                (pool_A if term == "A" else pool_B).append(b_id)

            if pool_A:
                bus_a = pool_A.pop(0)
            else:
                a_count += 1
                bus_a = f"A_{a_count}"
            route_trips.append({"Route": route, "Direction": "Forward", "Temp_Bus": bus_a,
                                "Start": T, "End": T + travel_time, "Block": hour_blocks[T]})
            arrivals.append((T + travel_time, "B", bus_a))

            if pool_B:
                bus_b = pool_B.pop(0)
            else:
                b_count += 1
                bus_b = f"B_{b_count}"
            route_trips.append({"Route": route, "Direction": "Backward", "Temp_Bus": bus_b,
                                "Start": T, "End": T + travel_time, "Block": hour_blocks[T]})
            arrivals.append((T + travel_time, "A", bus_b))

        id_map = {}
        for i in range(1, a_count + 1):
            id_map[f"A_{i}"] = f"EV-{str(next_bus).zfill(2)}"
            next_bus += 1
        for i in range(1, b_count + 1):
            id_map[f"B_{i}"] = f"EV-{str(next_bus).zfill(2)}"
            next_bus += 1

        for t in route_trips:
            all_trips.append({"Route": t["Route"], "Direction": t["Direction"],
                              "Bus": id_map[t["Temp_Bus"]],
                              "Start Time": t["Start"].strftime("%H:%M"),
                              "End Time": t["End"].strftime("%H:%M"),
                              "Hour Block": t["Block"]})

    df_dispatch = pd.DataFrame(all_trips)
    if df_dispatch.empty:
        return df_dispatch, pd.DataFrame(), pd.DataFrame()

    df_dist = _distance_lookup()
    rev = df_dist[df_dist["Dir"].isin(["F", "B"])].copy()
    rev["Direction"] = rev["Dir"].map({"F": "Forward", "B": "Backward"})
    df_km = pd.merge(df_dispatch, rev[["Route", "Direction", "Dist"]].rename(columns={"Dist": "Distance"}),
                     on=["Route", "Direction"], how="left").fillna(0)

    bus_route_kms = df_km.groupby(["Route", "Bus"])["Distance"].sum().reset_index()
    km_matrix = bus_route_kms.pivot(index="Route", columns="Bus", values="Distance").fillna(0)
    km_matrix["Total_Route_Km"] = km_matrix.sum(axis=1)
    active = (km_matrix.drop(columns=["Total_Route_Km"]) > 0).sum(axis=1)
    km_matrix["Avg_Km_Per_Bus_Daily"] = (km_matrix["Total_Route_Km"] / active.replace(0, np.nan)).round(2)
    return df_dispatch, bus_route_kms, km_matrix


@st.cache_data(show_spinner=False)
def run_soc_dispatch(supply_matrix, df_tt, hours):
    """SOC-adjusted dispatch engine (notebook's fleet analysis cell)."""
    df_dist = _distance_lookup()

    def route_dist(r, d):
        v = df_dist.loc[(df_dist["Route"] == r) & (df_dist["Dir"] == d), "Dist"]
        return v.iloc[0] if not v.empty else 0

    all_trips = []
    df_tt_unique = df_tt.groupby("route_code").mean(numeric_only=True)
    next_bus = 1

    for route in supply_matrix.index:
        tt_min = int(df_tt_unique.loc[route, "avg_tt_min"]) if route in df_tt_unique.index else 60
        travel_time = timedelta(minutes=tt_min)
        km_fwd, km_bwd = route_dist(route, "F"), route_dist(route, "B")
        timestamps, hour_blocks = _hourly_timestamps(supply_matrix, route, hours)

        def run_queue(fixed_a=0, fixed_b=0):
            pool_A = [f"A_{i}" for i in range(1, fixed_a + 1)]
            pool_B = [f"B_{i}" for i in range(1, fixed_b + 1)]
            a_count, b_count = max(fixed_a, 0), max(fixed_b, 0)
            arrivals, trips = [], []
            for T in sorted(timestamps):
                ready = [arr for arr in arrivals if arr[0] <= T]
                arrivals = [arr for arr in arrivals if arr[0] > T]
                ready.sort(key=lambda x: x[0])
                for _, term, b_id in ready:
                    (pool_A if term == "A" else pool_B).append(b_id)

                if pool_A:
                    bus_a = pool_A.pop(0)
                else:
                    a_count += 1
                    bus_a = f"A_{a_count}"
                trips.append({"Route": route, "Direction": "Forward", "Temp_Bus": bus_a,
                              "Start": T, "Block": hour_blocks[T], "KM": km_fwd})
                arrivals.append((T + travel_time, "B", bus_a))

                if pool_B:
                    bus_b = pool_B.pop(0)
                else:
                    b_count += 1
                    bus_b = f"B_{b_count}"
                trips.append({"Route": route, "Direction": "Backward", "Temp_Bus": bus_b,
                              "Start": T, "Block": hour_blocks[T], "KM": km_bwd})
                arrivals.append((T + travel_time, "A", bus_b))
            return trips, a_count, b_count

        base_trips, a_base, b_base = run_queue()
        base_fleet = a_base + b_base
        total_km = sum(t["KM"] for t in base_trips)

        if base_fleet > 0 and (total_km / base_fleet) > SOC_THRESHOLD_KM:
            target = math.ceil(total_km / SOC_THRESHOLD_KM)
            ratio_a = a_base / base_fleet
            a_t = max(math.ceil(target * ratio_a), a_base)
            b_t = max(target - a_t, b_base)
            final_trips, a_f, b_f = run_queue(fixed_a=a_t, fixed_b=b_t)
        else:
            final_trips, a_f, b_f = base_trips, a_base, b_base

        id_map = {}
        for i in range(1, a_f + 1):
            id_map[f"A_{i}"] = f"EV-{str(next_bus).zfill(2)}"
            next_bus += 1
        for i in range(1, b_f + 1):
            id_map[f"B_{i}"] = f"EV-{str(next_bus).zfill(2)}"
            next_bus += 1

        for t in final_trips:
            all_trips.append({"Route": t["Route"], "Direction": t["Direction"],
                              "Bus": id_map[t["Temp_Bus"]],
                              "Start Time": t["Start"].strftime("%H:%M"),
                              "Distance": t["KM"]})

    df = pd.DataFrame(all_trips)
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()
    bus_route_kms = df.groupby(["Route", "Bus"])["Distance"].sum().reset_index()
    km_matrix = bus_route_kms.pivot(index="Route", columns="Bus", values="Distance").fillna(0)
    km_matrix["Total_Route_Km"] = km_matrix.sum(axis=1)
    km_matrix["Avg_Km_Per_Bus"] = (km_matrix["Total_Route_Km"] /
                                   (km_matrix.drop(columns=["Total_Route_Km"]) > 0).sum(axis=1)).round(2)
    return df, bus_route_kms, km_matrix


# =====================================================================
# 4. ROUTE OPERATIONS (notebook: route summary + KM cells)
# =====================================================================

def fix_flipped_january_date(date_str):
    if pd.isna(date_str):
        return None
    date_part = str(date_str).split()[0]
    parts = date_part.split("-")
    if len(parts) == 3:
        year, middle, last = parts
        if last == "01" and middle != "01":
            return f"{year}-01-{middle.zfill(2)}"
    return date_part


@st.cache_data(show_spinner=False)
def route_operational_summary(start: str, end: str):
    engine = get_engine()
    with engine.connect() as conn:
        stops_raw = pd.read_sql_query("""
            SELECT r.route_code, rv.version_id, rv.file_timestamp, rs.stop_order, rs.stop_name
            FROM routes_information r
            JOIN route_versions rv ON r.route_id = rv.route_id
            JOIN route_stops rs ON rv.version_id = rs.version_id
            WHERE r.directional_id LIKE '%-F' OR r.directional_id LIKE '%-B'
            ORDER BY r.route_code, rv.file_timestamp, rv.version_id, rs.stop_order
        """, conn)

        routes_df = pd.read_sql("SELECT route_id, route_code FROM routes_information", conn)
        files_df = pd.read_sql(
            text("SELECT * FROM siso_files WHERE DATE(file_timestamp) BETWEEN :s AND :e"),
            conn, params={"s": start, "e": end})

        dist_raw = pd.read_sql("SELECT route_code, subcategory, distance FROM route_distances_tidy", conn)

    # --- latest stops per route ---
    stops_raw["clean_date"] = pd.to_datetime(stops_raw["file_timestamp"].apply(fix_flipped_january_date))
    latest = (stops_raw.sort_values(["clean_date", "version_id"])
              .groupby("route_code")["version_id"].last().reset_index())
    latest_stops = stops_raw.merge(latest, on=["route_code", "version_id"], how="inner")
    stops_df = (latest_stops.sort_values(["route_code", "stop_order"])
                .groupby("route_code").agg(
                    stop_count=("stop_name", "count"),
                    stops_name=("stop_name", lambda x: ", ".join(x.dropna().astype(str)))
                ).reset_index())

    # --- route active duration ---
    dur = stops_raw.groupby("route_code")["clean_date"].agg(["min", "max"]).reset_index()
    dur["days_active"] = (dur["max"] - dur["min"]).dt.days
    duration_df = dur[["route_code", "days_active"]].copy()

    # --- trip counts from siso_files in range ---
    def count_trips(raw_json):
        try:
            return pd.DataFrame(**json.loads(raw_json)).iloc[2:].shape[0]
        except Exception:
            return 0

    if files_df.empty:
        route_trips = pd.DataFrame(columns=["route_code", "f_trips", "b_trips", "total_trips"])
    else:
        files_df["trip_count"] = files_df["raw_json"].apply(count_trips)
        files_df = files_df.merge(routes_df, on="route_id", how="left")
        summary = (files_df.groupby(["route_code", "direction_info"])["trip_count"].sum().reset_index())
        pivot = summary.pivot(index="route_code", columns="direction_info", values="trip_count")
        pivot = pivot.reindex(columns=["F", "B"], fill_value=0).rename(
            columns={"F": "f_trips", "B": "b_trips"})
        pivot["total_trips"] = pivot.sum(axis=1)
        route_trips = pivot.reset_index()

    # --- distances incl. dead-KM bases ---
    pivot_dist = dist_raw.pivot(index="route_code", columns="subcategory", values="distance").fillna(0)
    for col in pivot_dist.columns:
        pivot_dist[col] = pd.to_numeric(pivot_dist[col], errors="coerce").fillna(0)
    pivot_dist = pivot_dist.rename(columns={"F": "distance_F", "B": "distance_B"})
    pivot_dist["min_ST"] = pivot_dist[["ST_H9", "ST_JCC", "ST_ZPoint"]].min(axis=1)
    pivot_dist["min_ET"] = pivot_dist[["ET_H9", "ET_JCC", "ET_ZPoint"]].min(axis=1)
    dist_df = pivot_dist.reset_index()

    # --- KM computation (revenue + dead) ---
    trips = route_trips.copy()
    trips["route_code"] = trips["route_code"].astype(str).str.strip()
    dist_df["route_code"] = dist_df["route_code"].astype(str).str.strip()
    trips["f_trips"] = pd.to_numeric(trips["f_trips"], errors="coerce").fillna(0)
    trips["b_trips"] = pd.to_numeric(trips["b_trips"], errors="coerce").fillna(0)
    merged = trips.merge(dist_df, on="route_code", how="left").fillna(0)
    merged["km_F"] = merged["f_trips"] * merged["distance_F"]
    merged["km_B"] = merged["b_trips"] * merged["distance_B"]
    merged["km_total"] = merged["km_F"] + merged["km_B"]
    merged["dead_KM_from_Depot"] = merged["f_trips"] * merged["min_ST"]
    merged["dead_KM_to_Depot"] = merged["b_trips"] * merged["min_ET"]
    merged["total_dead_KM"] = merged["dead_KM_from_Depot"] + merged["dead_KM_to_Depot"]

    for df in (stops_df, duration_df):
        df["route_code"] = df["route_code"].astype(str).str.strip()

    final = (stops_df
             .merge(duration_df, on="route_code", how="left")
             .merge(merged, on="route_code", how="left")).fillna(0)

    total_row = pd.DataFrame([{
        "route_code": "TOTAL",
        "stop_count": final["stop_count"].sum(),
        "days_active": final["days_active"].max(),
        "f_trips": final["f_trips"].sum(),
        "b_trips": final["b_trips"].sum(),
        "total_trips": final["total_trips"].sum(),
        "km_F": final["km_F"].sum(),
        "km_B": final["km_B"].sum(),
        "km_total": final["km_total"].sum(),
        "dead_KM_from_Depot": final["dead_KM_from_Depot"].sum(),
        "dead_KM_to_Depot": final["dead_KM_to_Depot"].sum(),
        "total_dead_KM": final["total_dead_KM"].sum(),
    }])
    return pd.concat([final, total_row], ignore_index=True)


# =====================================================================
# 5. PHASE-III SYNTHESIS (notebook: unified variance synthesis cell)
# =====================================================================

def _fix_flipped_jan_strict(date_str):
    if pd.isna(date_str):
        return None
    date_part = str(date_str).split()[0]
    parts = date_part.split("-")
    if len(parts) == 3:
        year, middle, last = parts
        if last == "01" and middle.isdigit() and int(middle) > 12:
            return f"{year}-01-{middle.zfill(2)}"
    return date_part


@st.cache_data(show_spinner=False)
def phase3_synthesis(start: str, end: str):
    engine = get_engine()
    with engine.connect() as conn:
        routes_df = pd.read_sql("SELECT route_id, route_code FROM routes_information", conn)
        files_df = pd.read_sql(
            text("SELECT * FROM siso_files WHERE DATE(file_timestamp) BETWEEN :s AND :e"),
            conn, params={"s": start, "e": end})

        stops_raw = pd.read_sql_query("""
            SELECT r.route_code, rv.version_id, rv.file_timestamp, rs.stop_order, rs.stop_name
            FROM routes_information r
            JOIN route_versions rv ON r.route_id = rv.route_id
            JOIN route_stops rs ON rv.version_id = rs.version_id
            WHERE r.directional_id LIKE '%-F' OR r.directional_id LIKE '%-B'
            ORDER BY r.route_code, rv.file_timestamp, rv.version_id, rs.stop_order
        """, conn)

        dur_raw = pd.read_sql_query("""
            SELECT route_code,
                   MIN(file_timestamp) as first_used_timestamp,
                   MAX(file_timestamp) as last_update_timestamp
            FROM routes_information r
            JOIN route_versions rv ON r.route_id = rv.route_id
            GROUP BY route_code
        """, conn)

        dist_raw = pd.read_sql("SELECT route_code, subcategory, distance FROM route_distances_tidy", conn)

        if table_exists(conn, "phase1_simulated_schedules"):
            sim_df = pd.read_sql(
                "SELECT route_code, sim_trips_F, sim_trips_B, sim_shift_count FROM phase1_simulated_schedules",
                conn)
        else:
            sim_df = pd.read_sql("SELECT DISTINCT route_code FROM routes_information", conn)
            sim_df["route_code"] = sim_df["route_code"].astype(str).str.strip()
            sim_df = sim_df[sim_df["route_code"] != "TOTAL"]
            sim_df["sim_trips_F"] = 12
            sim_df["sim_trips_B"] = 12
            sim_df["sim_shift_count"] = 2

    # actual trips per direction
    def count_trips(raw_json):
        try:
            return pd.DataFrame(**json.loads(raw_json)).iloc[2:].shape[0]
        except Exception:
            return 0

    if files_df.empty:
        actuals_df = pd.DataFrame(columns=["route_code", "act_trips_F", "act_trips_B", "act_shift_count"])
    else:
        files_df["trip_count"] = files_df["raw_json"].apply(count_trips)
        files_df = files_df.merge(routes_df, on="route_id", how="left")
        summary = (files_df.groupby(["route_code", "direction_info"])
                   .agg(total_trips_dir=("trip_count", "sum")).reset_index())
        pivot_trips = summary.pivot(index="route_code", columns="direction_info", values="total_trips_dir")
        pivot_trips = pivot_trips.reindex(columns=["F", "B"], fill_value=0).rename(
            columns={"F": "act_trips_F", "B": "act_trips_B"})
        pivot_shifts = files_df.groupby("route_code")["route_id"].count().to_frame("act_shift_count")
        actuals_df = pivot_trips.merge(pivot_shifts, on="route_code", how="left").reset_index()

    # distances
    pivot_dist = dist_raw.pivot(index="route_code", columns="subcategory", values="distance").fillna(0)
    for col in pivot_dist.columns:
        pivot_dist[col] = pd.to_numeric(pivot_dist[col], errors="coerce").fillna(0)
    pivot_dist = pivot_dist.rename(columns={"F": "distance_F", "B": "distance_B"})
    pivot_dist["min_ST"] = pivot_dist[["ST_H9", "ST_JCC", "ST_ZPoint"]].min(axis=1)
    pivot_dist["min_ET"] = pivot_dist[["ET_H9", "ET_JCC", "ET_ZPoint"]].min(axis=1)
    dist_df = pivot_dist.reset_index()

    # latest stops
    stops_raw["clean_date"] = pd.to_datetime(stops_raw["file_timestamp"].apply(_fix_flipped_jan_strict))
    latest = (stops_raw.sort_values(["clean_date", "version_id"])
              .groupby("route_code")["version_id"].last().reset_index())
    latest_stops = stops_raw.merge(latest, on=["route_code", "version_id"], how="inner")
    stops_df = (latest_stops.sort_values(["route_code", "stop_order"])
                .groupby("route_code").agg(
                    stop_count=("stop_name", "count"),
                    stops_name=("stop_name", lambda x: ", ".join(x.dropna().astype(str)))
                ).reset_index())

    # duration
    dur_raw["first_used_timestamp"] = pd.to_datetime(dur_raw["first_used_timestamp"].apply(_fix_flipped_jan_strict))
    dur_raw["last_update_timestamp"] = pd.to_datetime(dur_raw["last_update_timestamp"].apply(_fix_flipped_jan_strict))
    dur_raw["days_active"] = (dur_raw["last_update_timestamp"] - dur_raw["first_used_timestamp"]).dt.days + 1
    duration_df = dur_raw[["route_code", "days_active"]].copy()

    for df in (actuals_df, dist_df, stops_df, duration_df, sim_df):
        df["route_code"] = df["route_code"].astype(str).str.strip()

    master = (sim_df
              .merge(actuals_df, on="route_code", how="outer")
              .merge(dist_df, on="route_code", how="left")
              .merge(stops_df, on="route_code", how="left")
              .merge(duration_df, on="route_code", how="left")).fillna(0)
    master = master[master["route_code"] != "TOTAL"].copy()

    master["sim_km_revenue"] = master["sim_trips_F"] * master["distance_F"] + master["sim_trips_B"] * master["distance_B"]
    master["sim_km_deadhead"] = master["sim_shift_count"] * master["min_ST"] + master["sim_shift_count"] * master["min_ET"]
    master["sim_km_total"] = master["sim_km_revenue"] + master["sim_km_deadhead"]
    master["act_km_revenue"] = master["act_trips_F"] * master["distance_F"] + master["act_trips_B"] * master["distance_B"]
    master["act_km_deadhead"] = master["act_shift_count"] * master["min_ST"] + master["act_shift_count"] * master["min_ET"]
    master["act_km_total"] = master["act_km_revenue"] + master["act_km_deadhead"]

    master["variance_trips_F"] = master["act_trips_F"] - master["sim_trips_F"]
    master["variance_trips_B"] = master["act_trips_B"] - master["sim_trips_B"]
    master["variance_km_total"] = master["act_km_total"] - master["sim_km_total"]
    master["service_delivery_idx"] = np.where(master["sim_km_total"] > 0,
                                              master["act_km_total"] / master["sim_km_total"] * 100, 0.0)
    master["avg_km_per_actual_shift"] = np.where(master["act_shift_count"] > 0,
                                                 master["act_km_total"] / master["act_shift_count"], 0.0)
    master["soc_breach_risk"] = master["avg_km_per_actual_shift"].apply(
        lambda x: "CRITICAL RISK" if x > SOC_THRESHOLD_KM
        else "ELEVATED" if x > SOC_THRESHOLD_KM * 0.9 else "SAFE")

    total = pd.DataFrame([{
        "route_code": "TOTAL",
        "sim_trips_F": master["sim_trips_F"].sum(), "sim_trips_B": master["sim_trips_B"].sum(),
        "sim_shift_count": master["sim_shift_count"].sum(),
        "act_trips_F": master["act_trips_F"].sum(), "act_trips_B": master["act_trips_B"].sum(),
        "act_shift_count": master["act_shift_count"].sum(),
        "distance_F": 0.0, "distance_B": 0.0, "min_ST": 0.0, "min_ET": 0.0,
        "stop_count": master["stop_count"].sum(), "days_active": master["days_active"].max(),
        "sim_km_revenue": master["sim_km_revenue"].sum(),
        "sim_km_deadhead": master["sim_km_deadhead"].sum(),
        "sim_km_total": master["sim_km_total"].sum(),
        "act_km_revenue": master["act_km_revenue"].sum(),
        "act_km_deadhead": master["act_km_deadhead"].sum(),
        "act_km_total": master["act_km_total"].sum(),
        "variance_trips_F": master["variance_trips_F"].sum(),
        "variance_trips_B": master["variance_trips_B"].sum(),
        "variance_km_total": master["variance_km_total"].sum(),
        "service_delivery_idx": (master["act_km_total"].sum() / master["sim_km_total"].sum() * 100)
                                if master["sim_km_total"].sum() > 0 else 0.0,
        "avg_km_per_actual_shift": master["act_km_total"].sum() / master["act_shift_count"].sum()
                                   if master["act_shift_count"].sum() > 0 else 0.0,
        "soc_breach_risk": "SYSTEM AGGREGATE",
    }])
    return pd.concat([master, total], ignore_index=True)


# =====================================================================
# UI
# =====================================================================

def render(start: str, end: str):
    # ---------- Sales & ridership ----------
    df_join, df_terminals, df_routes, df_users, df_regional = load_sales(start, end)

    st.header("Sales & Ridership")
    st.metric("Total Ridership", f"{len(df_join):,}")
    st.subheader("Product Sales Analysis")
    product_pivot_chart(df_terminals, "Terminal No", "Terminal-wise Sales")
    product_pivot_chart(df_routes, "route_number", "Route-wise Sales")
    product_pivot_chart(df_users, "Personnel Name", "Personnel-wise Sales")

    st.subheader("Regional Sales (50 vs 90)")
    if not df_regional.empty:
        reg = df_regional.set_index("Terminal_No")[["total_sales_50", "total_sales_90", "total_sales_other"]]
        fig, ax = plt.subplots(figsize=(14, 6))
        reg.plot(kind="bar", stacked=True, ax=ax,
                 color=["#4C72B0", "#55A868", "#C44E52"], edgecolor="black")
        ax.set_title("Terminal-wise Revenue Breakdown (50 vs 90 vs Other)", pad=15)
        ax.set_ylabel("Total Revenue (PKR)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(["50-Rupee Sales", "90-Rupee Sales", "Other Sales"],
                  bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        t50, t90 = df_regional["count_50"].sum(), df_regional["count_90"].sum()
        if t50 > 0 or t90 > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie([t50, t90], labels=["50-Rupee Tickets", "90-Rupee Tickets"],
                   autopct="%1.1f%%", colors=["#4C72B0", "#55A868"], startangle=90,
                   wedgeprops={"edgecolor": "black", "width": 0.4})
            ax.set_title("Overall Ticket Count Volume (50 vs 90)", pad=15)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("No regional sales in range.")

    st.subheader("Hourly Ridership")
    hm, product = hourly_matrix_from(df_join)
    if hm is not None:
        st.caption(f"Product: **{product}**")
        st.dataframe(hm)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(hm, annot=True, fmt="d", cmap="Blues", linewidths=.5,
                    cbar_kws={"label": "Ridership Count"}, ax=ax)
        ax.set_title(f"Heatmap of {product} by Route and Hour", pad=15)
        ax.set_xlabel("Hour of the Day")
        ax.set_ylabel("Route Number")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No tickets in range.")

    # ---------- Supply & dispatch ----------
    supply_matrix, df_tt, hours, sm_product = build_supply_matrix(start, end)
    st.header("Supply & Dispatch")
    if supply_matrix is None:
        st.info("No ticket data in range — supply matrix unavailable.")
    else:
        st.subheader("Supply Matrix (Operating Units per Hour)")
        st.caption(f"Product: **{sm_product}** · capacity {BUS_CAPACITY} · directional factor {TOWARDS_ISB}")
        st.dataframe(supply_matrix)

        df_dispatch, bus_route_kms, km_matrix = run_dispatch(supply_matrix, df_tt, hours)
        st.subheader("Generated Dispatch Schedule")
        st.dataframe(df_dispatch, height=400)

        if not bus_route_kms.empty:
            fig = px.box(bus_route_kms, x="Route", y="Distance", points="all", color="Route",
                         title="EV Mileage Distribution by Route (Threshold 230 KM)")
            fig.add_hline(y=230, line_dash="dash", line_color="red",
                          annotation_text="Limit: 230 KM")
            st.plotly_chart(fig, use_container_width=True)

            workload = bus_route_kms.groupby("Bus")["Distance"].sum().reset_index()
            fig2 = px.bar(workload, x="Bus", y="Distance",
                          title="Total Daily KM per Bus", color="Distance")
            fig2.add_hline(y=230, line_dash="dot", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("KM Matrix")
            st.dataframe(km_matrix)

    # ---------- SOC fleet ----------
    st.header("SOC Fleet Analysis")
    if supply_matrix is None:
        st.info("No ticket data in range.")
    else:
        df_soc, soc_kms, soc_matrix = run_soc_dispatch(supply_matrix, df_tt, hours)
        if soc_kms.empty:
            st.info("No dispatch trips generated.")
        else:
            fig = px.box(soc_kms, x="Route", y="Distance", points="all", color="Route",
                         title=f"Route Performance (Threshold: {SOC_THRESHOLD_KM:.0f} KM)")
            fig.add_hline(y=SOC_THRESHOLD_KM, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

            totals = soc_kms.groupby("Bus")["Distance"].sum().reset_index()
            fig2 = px.bar(totals, x="Bus", y="Distance",
                          title="Individual Bus Daily Workload", color="Distance")
            fig2.add_hline(y=SOC_THRESHOLD_KM, line_dash="dot", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Route Summary: Average KM per Bus")
            st.dataframe(soc_matrix[["Avg_Km_Per_Bus"]])
            st.subheader("Full KM Matrix")
            st.dataframe(soc_matrix)

    # ---------- Route operations ----------
    st.header("Route Operations")
    result_df = route_operational_summary(start, end)
    st.subheader("Route Operational Summary (Revenue + Dead KM)")
    st.dataframe(result_df)

    plot_df = result_df[result_df["route_code"] != "TOTAL"]
    if not plot_df.empty and plot_df["total_trips"].sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(plot_df["f_trips"], plot_df["km_F"], color="blue", s=80,
                   label="Forward Trips vs KM")
        ax.scatter(plot_df["b_trips"], plot_df["km_B"], color="green", s=80,
                   label="Backward Trips vs KM")
        ax.scatter(plot_df["total_trips"], plot_df["km_total"], color="red", s=100,
                   marker="x", label="Total Trips vs KM")
        for _, row in plot_df.iterrows():
            ax.annotate(row["route_code"], (row["total_trips"], row["km_total"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_title("KM vs Trips Scatter Plot (Forward, Backward, Total)", fontsize=16)
        ax.set_xlabel("Trips Count", fontsize=14)
        ax.set_ylabel("Kilometers", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        plot_df = plot_df.sort_values("route_code")
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax1.set_title("Dual Axis Plot: Trips vs Kilometers", fontsize=18)
        ax1.plot(plot_df["route_code"], plot_df["f_trips"], marker="o", color="blue", label="Forward Trips")
        ax1.plot(plot_df["route_code"], plot_df["b_trips"], marker="o", color="green", label="Backward Trips")
        ax1.plot(plot_df["route_code"], plot_df["total_trips"], marker="o", color="red",
                 label="Total Trips", linewidth=2)
        ax1.set_xlabel("Route Code", fontsize=14)
        ax1.set_ylabel("Trips Count", fontsize=14)
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax2 = ax1.twinx()
        ax2.plot(plot_df["route_code"], plot_df["km_F"], marker="s", color="blue",
                 linestyle="--", label="Forward KM")
        ax2.plot(plot_df["route_code"], plot_df["km_B"], marker="s", color="green",
                 linestyle="--", label="Backward KM")
        ax2.plot(plot_df["route_code"], plot_df["km_total"], marker="s", color="red",
                 linestyle="--", label="Total KM", linewidth=2)
        ax2.set_ylabel("Kilometers", fontsize=14)
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No SISO trips in range — KM plots skipped.")

    # ---------- Phase-III ----------
    st.header("Phase-III Synthesis")
    st.subheader("Phase-III Unified Variance Synthesis Matrix")
    matrix = phase3_synthesis(start, end)
    st.dataframe(matrix)


def main():
    st.set_page_config(page_title="Dispatch", layout="wide")
    st.title("Dispatch")

    missing = not DATA_DIR.is_dir() or not any(DATA_DIR.glob("*.parquet"))
    if missing:
        st.error(f"No Parquet tables in {DATA_DIR}. Run build_data.py to generate them.")
        st.stop()

    with st.sidebar:
        st.header("Query parameters")
        start = st.date_input("Start date", value=dt.date(2025, 1, 17))
        end = st.date_input("End date", value=dt.date(2025, 1, 17))
        run = st.button("Run Query", type="primary", use_container_width=True)

    if run:
        if start > end:
            st.error("Start date must be on or before end date.")
            st.stop()
        st.session_state["query_range"] = (str(start), str(end))

    if "query_range" in st.session_state:
        s, e = st.session_state["query_range"]
        st.success(f"Showing results for **{s} → {e}**")
        render(s, e)
    else:
        st.info("Pick a date range in the sidebar and press **Run Query**.")


if __name__ == "__main__":
    main()
