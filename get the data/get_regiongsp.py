import argparse
import io
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

API_BASE = "https://api.neso.energy/api/3/action"
DEFAULT_PACKAGE_ID = "regional-breakdown-of-fes-data-electricity"
DEFAULT_RESOURCE_NAME = "fes_2024_grid_supply_point_info"
DEFAULT_OUT = "gsp_info.csv"


def _fetch_resources(session: requests.Session, package_id: str) -> List[Dict]:
    url = f"{API_BASE}/datapackage_show"
    resp = session.get(url, params={"id": package_id}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"API error: {payload}")
    # be polite to CKAN API
    time.sleep(1.1)
    return payload["result"]["resources"]


def _resource_download_url(resource: Dict) -> Optional[str]:
    return resource.get("url") or resource.get("path")


def _pick_csv_resource(resources: List[Dict], resource_name: str) -> Dict:
    for r in resources:
        name = (r.get("name") or "").lower()
        if name == resource_name.lower():
            return r
    for r in resources:
        fmt = (r.get("format") or "").lower()
        url = (_resource_download_url(r) or "").lower()
        name = (r.get("name") or "").lower()
        if fmt == "csv" or url.endswith(".csv") or ".csv" in url or "csv" in name:
            return r
    raise RuntimeError("No CSV resource found for GSP info dataset.")


def download_gsp_info(package_id: str, resource_name: str, out_csv: str) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; NESO-data-script/1.0)"})

    resources = _fetch_resources(session, package_id)
    resource = _pick_csv_resource(resources, resource_name)
    url = _resource_download_url(resource)
    if not url:
        raise RuntimeError("CSV resource has no download URL.")

    print(f"Downloading GSP info: {resource.get('name') or url}")
    resp = session.get(url, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NESO GSP info mapping file.")
    parser.add_argument("--package-id", default=DEFAULT_PACKAGE_ID, help="NESO CKAN package id")
    parser.add_argument("--resource-name", default=DEFAULT_RESOURCE_NAME, help="Resource name within the package")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output CSV filename")
    args = parser.parse_args()

    download_gsp_info(args.package_id, args.resource_name, args.out)


if __name__ == "__main__":
    main()
