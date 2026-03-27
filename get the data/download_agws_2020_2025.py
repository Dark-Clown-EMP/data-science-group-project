import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

BASE = "https://data.elexon.co.uk/bmrs/api/v1"

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 1, 1, tzinfo=timezone.utc)   # up to end of 2025
CHUNK_DAYS = 7

OUT_CSV = "wind_solar_outturn_2020_2025.csv"

def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def get_json(session: requests.Session, url: str, params: dict, retries=5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 429:
                # rate limited
                wait = 2 * attempt
                print(f"  ⏳ 429 rate limit, sleeping {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed: url={url} params={params} last_err={last_err}")

def normalize_records(js):
    """
    Elexon endpoints usually return {"data":[...]}.
    This function tries to safely extract a list of records.
    """
    if isinstance(js, dict):
        if "data" in js and isinstance(js["data"], list):
            return js["data"]
        if "items" in js and isinstance(js["items"], list):
            return js["items"]
    if isinstance(js, list):
        return js
    raise ValueError(f"Unexpected JSON shape: {type(js)} keys={list(js.keys()) if isinstance(js, dict) else None}")

def fetch_wind_solar_generation_api(session, start_dt, end_dt):
    # Primary: generation/actual/per-type/wind-and-solar
    url = f"{BASE}/generation/actual/per-type/wind-and-solar"
    params = {"from": iso(start_dt), "to": iso(end_dt), "format": "json"}
    js = get_json(session, url, params)
    recs = normalize_records(js)
    df = pd.json_normalize(recs)

    # Try to standardize common column names
    # (Different endpoints sometimes use different casing)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("settlementdate", "settlement_date"):
            rename_map[c] = "SETTLEMENT_DATE"
        elif lc in ("settlementperiod", "settlement_period"):
            rename_map[c] = "SETTLEMENT_PERIOD"
        elif lc in ("starttime", "start_time", "datetime", "timestamp"):
            rename_map[c] = "timestamp"
        elif "wind" in lc and ("mw" in lc or "generation" in lc or "value" in lc):
            # leave as-is; we'll aggregate later if needed
            pass

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

def fetch_fuelhh_fallback(session, start_dt, end_dt):
    # Fallback: datasets/FUELHH
    url = f"{BASE}/datasets/FUELHH"
    params = {"from": iso(start_dt), "to": iso(end_dt), "format": "json"}
    js = get_json(session, url, params)
    recs = normalize_records(js)
    df = pd.json_normalize(recs)

    # Standardize likely column names
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("settlementdate", "settlement_date"):
            colmap[c] = "SETTLEMENT_DATE"
        elif lc in ("settlementperiod", "settlement_period"):
            colmap[c] = "SETTLEMENT_PERIOD"
        elif lc in ("fueltype", "fuel_type"):
            colmap[c] = "fuelType"
        elif lc in ("quantity", "value", "generation", "output"):
            # choose first as MW value if not already mapped
            if "MW" not in colmap.values():
                colmap[c] = "MW"
    df = df.rename(columns=colmap)

    # If MW still missing, try to auto-detect a numeric column
    if "MW" not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        # remove obvious non-measure fields
        numeric_cols = [c for c in numeric_cols if c.lower() not in ("settlement_period", "settlementperiod", "year")]
        if not numeric_cols:
            raise RuntimeError(f"Could not find MW column in FUELHH response. Columns: {df.columns.tolist()}")
        df = df.rename(columns={numeric_cols[0]: "MW"})

    return df

def build_timestamp(df):
    # Build half-hour timestamp from SETTLEMENT_DATE + SETTLEMENT_PERIOD if possible
    if "SETTLEMENT_DATE" in df.columns and "SETTLEMENT_PERIOD" in df.columns:
        d = pd.to_datetime(df["SETTLEMENT_DATE"], errors="coerce", dayfirst=True)
        sp = pd.to_numeric(df["SETTLEMENT_PERIOD"], errors="coerce")
        ts = d + pd.to_timedelta((sp - 1) * 30, unit="min")
        df["timestamp"] = ts.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    return df

def main():
    session = requests.Session()

    # quick health check
    hc = session.get(f"{BASE}/health", timeout=20)
    print("Health:", hc.status_code, hc.url)

    all_rows = []
    cur = START
    i = 0

    while cur < END:
        i += 1
        nxt = min(cur + timedelta(days=CHUNK_DAYS), END)
        print(f"[{i:04d}] {iso(cur)} -> {iso(nxt)}")

        # Try primary endpoint first
        try:
            df = fetch_wind_solar_generation_api(session, cur, nxt)
            df = build_timestamp(df)
            # If endpoint returns already aggregated columns, just save raw
            all_rows.append(df)
            print(f"   ✅ generation/wind-and-solar rows: {len(df)}")
        except Exception as e:
            print(f"   ⚠️ primary failed ({type(e).__name__}): {e}")
            # Fallback to FUELHH and aggregate wind/solar
            df = fetch_fuelhh_fallback(session, cur, nxt)
            df = build_timestamp(df)

            # Keep wind + solar only (robust matching)
            if "fuelType" not in df.columns:
                raise RuntimeError("FUELHH fallback returned no fuelType column; cannot filter wind/solar.")

            mask = df["fuelType"].astype(str).str.upper().str.contains("WIND|SOLAR", regex=True)
            df = df.loc[mask, ["timestamp", "fuelType", "MW"]].copy()

            # Aggregate to one row per timestamp: WIND_MW and SOLAR_MW
            df["is_wind"] = df["fuelType"].astype(str).str.upper().str.contains("WIND")
            df["is_solar"] = df["fuelType"].astype(str).str.upper().str.contains("SOLAR")

            agg = df.groupby("timestamp").apply(
                lambda g: pd.Series({
                    "WIND_MW":  g.loc[g["is_wind"], "MW"].sum(),
                    "SOLAR_MW": g.loc[g["is_solar"], "MW"].sum(),
                })
            ).reset_index()

            all_rows.append(agg)
            print(f"   ✅ FUELHH aggregated rows: {len(agg)}")

        # be polite
        time.sleep(0.2)
        cur = nxt

    out = pd.concat(all_rows, ignore_index=True)

    # De-duplicate timestamps if chunks overlap
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    out.to_csv(OUT_CSV, index=False)
    print(f"\n🎉 Saved: {OUT_CSV}  rows={len(out)} cols={list(out.columns)}")

if __name__ == "__main__":
    main()
