import io
import re
import time
from typing import Dict, List

import pandas as pd
import requests

API_BASE = "https://api.neso.energy/api/3/action"
DATASET_ID = "embedded-wind-and-solar-forecasts"
OUTPUT_FILE = "regional_embedded_wind_2020_2025.csv"

START_YEAR = 2020
END_YEAR = 2025


def _fetch_resources(session: requests.Session) -> List[Dict]:
    url = f"{API_BASE}/datapackage_show"
    resp = session.get(url, params={"id": DATASET_ID}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"API error: {payload}")
    # Be polite to the CKAN API (1 req/sec guidance).
    time.sleep(1.1)
    return payload["result"]["resources"]


def _select_archive_resources(resources: List[Dict], start_year: int, end_year: int) -> List[Dict]:
    selected = []
    year_re = re.compile(r"(20\d{2})")

    for r in resources:
        name = (r.get("name") or "").lower()
        url = (r.get("url") or r.get("path") or "").lower()
        haystack = f"{name} {url}"
        if "archive" not in haystack:
            continue

        years = [int(y) for y in year_re.findall(haystack)]
        year = next((y for y in years if start_year <= y <= end_year), None)
        if year is None:
            continue

        download_url = r.get("url") or r.get("path")
        if not download_url:
            continue

        selected.append(
            {
                "year": year,
                "name": r.get("name"),
                "url": download_url,
            }
        )

    selected.sort(key=lambda x: x["year"])
    return selected


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]

    if "SETTLEMENT_DATE" in df.columns and "SETTLEMENT_PERIOD" in df.columns:
        settle_date = pd.to_datetime(df["SETTLEMENT_DATE"], errors="coerce", utc=True)
        settle_period = pd.to_numeric(df["SETTLEMENT_PERIOD"], errors="coerce")
        df["DATETIME"] = settle_date + pd.to_timedelta((settle_period - 1) * 30, unit="min")

    return df


def download_archives(start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    print("Fetching dataset resources from NESO API...")
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; NESO-data-script/1.0)",
        }
    )

    resources = _fetch_resources(session)
    archives = _select_archive_resources(resources, start_year, end_year)

    if not archives:
        raise RuntimeError("No archive resources found for requested years.")

    print(f"Found {len(archives)} archive files for {start_year}-{end_year}.")

    frames = []
    for item in archives:
        url = item["url"]
        name = item["name"] or url
        print(f"Downloading {item['year']} archive: {name}...")
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
        df = _normalize_columns(df)
        df["SOURCE_FILE"] = name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def main() -> None:
    df = download_archives()

    if "DATETIME" in df.columns:
        mask = (df["DATETIME"] >= "2020-01-01") & (df["DATETIME"] <= "2025-12-31")
        df = df.loc[mask]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
