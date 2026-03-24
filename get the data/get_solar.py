import argparse
import io
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import pandas as pd
import requests

API_BASE = "https://api.pvlive.uk/pvlive/api/v4"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_LEVEL = "gsp"  # gsp or pes
DEFAULT_OUT = "regional_solar_2020_2025.csv"
CHUNK_DAYS = 365  # PV-Live allows up to ~12 months per request


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _get_regions(session: requests.Session, level: str) -> List[Dict]:
    if level == "gsp":
        url = f"{API_BASE}/gsp_list"
    elif level == "pes":
        url = f"{API_BASE}/pes_list"
    else:
        raise ValueError("level must be 'gsp' or 'pes'")
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    # PV-Live returns {"data":[...], "meta":{...}}
    return payload.get("data", [])


def _fetch_region_data(session: requests.Session, level: str, region_id: int, start: datetime, end: datetime) -> pd.DataFrame:
    url = f"{API_BASE}/{level}/{region_id}"
    params = {
        "start": _iso(start),
        "end": _iso(end),
        "data_format": "csv",
    }
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.content.decode("utf-8")))


def _date_chunks(start: datetime, end: datetime, chunk_days: int) -> List[Tuple[datetime, datetime]]:
    chunks = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days) - timedelta(minutes=30), end)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(minutes=30)
    return chunks


def download_all(level: str, start_date: str, end_date: str, out_csv: str) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PVLive-downloader/1.0)"})

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    regions = _get_regions(session, level)
    if not regions:
        raise RuntimeError("No regions returned from PV-Live.")

    if level == "gsp":
        # format: [gsp_id, gsp_name, n_periods]
        region_key = "gsp_id"
        name_key = "gsp_name"
    else:
        # format: [pes_id, pes_code, pes_name]
        region_key = "pes_id"
        name_key = "pes_name"

    wrote_header = False
    chunks = _date_chunks(start_dt, end_dt, CHUNK_DAYS)

    for idx, reg in enumerate(regions, start=1):
        if not isinstance(reg, (list, tuple)) or len(reg) < 2:
            continue
        if level == "gsp":
            region_id, region_name = reg[0], reg[1]
        else:
            region_id, region_name = reg[0], reg[2]

        print(f"[{idx}/{len(regions)}] {level.upper()} {region_id} {region_name}")

        for (cstart, cend) in chunks:
            print(f"  {cstart.date()} -> {cend.date()}")
            df = _fetch_region_data(session, level, region_id, cstart, cend)
            if df.empty:
                continue
            df[region_key] = region_id
            df[name_key] = region_name
            df.to_csv(out_csv, index=False, mode="a", header=not wrote_header)
            wrote_header = True
            time.sleep(0.2)

    if not wrote_header:
        raise RuntimeError("No data downloaded.")
    print(f"Saved data to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download regional PV-Live solar generation.")
    parser.add_argument("--level", default=DEFAULT_LEVEL, choices=["gsp", "pes"], help="gsp or pes")
    parser.add_argument("--start", default=DEFAULT_START, help="start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="end date YYYY-MM-DD")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output CSV")
    args = parser.parse_args()

    download_all(args.level, args.start, args.end, args.out)


if __name__ == "__main__":
    main()
