"""Finoview commodity indicator crawler.

Downloads financial indicator time-series from Finoview's authenticated API in
two phases:

1. **Catalogue fetch** - for each category ID in ``INDEX_DIC``, retrieve the
   full indicator listing via ``queryCatalogueForIndex`` and cache it as
   ``{item}_{name}_lst.csv`` so a re-run does not hit the API twice.
2. **Series fetch** - for every indicator in the listing, pull its historical
   time-series via ``indexTableList`` and persist each as its own CSV. Any
   file already on disk is skipped, so partial runs resume cleanly.

Session credentials expire; refresh from the browser DevTools and export as
environment variables (``FINOVIEW_TOKEN``, ``FINOVIEW_COOKIE``). The script
politely rate-limits itself with a jittered ~10-15 s sleep between requests.
"""

from __future__ import annotations

import os
import random
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

CATALOGUE_URL = "https://www.finoview.com.cn/foreApi/queryCatalogueForIndex"
SERIES_URL = "https://www.finoview.com.cn/foreApi/indexTableList"

DEFAULT_START_DATE = "1900-01-01"
DEFAULT_END_DATE = "2025-11-30"

REQUEST_SLEEP_BASE = 10
REQUEST_SLEEP_JITTER = 5


def _build_headers() -> dict[str, str]:
    """Assemble the HTTP headers Finoview requires.

    Reads the JWT token and session cookie from environment variables so that
    credentials never live in source control:

        export FINOVIEW_TOKEN="eyJ..."
        export FINOVIEW_COOKIE="sensorsdata2015jssdkcross=...; fs_token_cookie=..."
    """
    token = os.environ.get("FINOVIEW_TOKEN")
    cookie = os.environ.get("FINOVIEW_COOKIE")
    if not token or not cookie:
        raise EnvironmentError(
            "Set FINOVIEW_TOKEN and FINOVIEW_COOKIE environment variables before running."
        )
    return {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json;charset=UTF-8",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        ),
        "Origin": "https://www.finoview.com.cn",
        "Referer": "https://www.finoview.com.cn/NewCommodity/newdashboard?isdetail=1",
        "Cookie": cookie,
        "Token": token,
    }


def _fetch_catalogue(catalogue_id: int, headers: dict[str, str]) -> pd.DataFrame:
    """Fetch the indicator listing for a single catalogue ID."""
    payload = {"catalogueId": catalogue_id, "model": 100}
    response = requests.post(CATALOGUE_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json()["data"])


def _fetch_series(
    indicator_id: str,
    field_id: str,
    headers: dict[str, str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch a single indicator time-series."""
    payload = {
        "id": indicator_id,
        "fid": field_id,
        "modelid": 101,
        "templatedataid": "",
        "templateid": "",
        "cycleMap": [],
        "endtime": end_date,
        "startTime": start_date,
        "type": 3,
    }
    response = requests.post(SERIES_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    payload = response.json()["data"][0]
    df = pd.DataFrame(payload["list"])
    for meta_key in ("unit", "name", "frequencyname", "source"):
        if meta_key in payload:
            df[meta_key] = payload[meta_key]
    return df


def download_indicators(
    catalogue_map: dict[str, int],
    output_root: Path,
    item_tag: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> None:
    """Download every indicator under every catalogue into ``output_root/item_tag``.

    Parameters
    ----------
    catalogue_map:
        Human-readable catalogue name -> Finoview ``catalogueId``.
    output_root:
        Directory that will hold the ``item_tag`` subfolder. Created if missing.
    item_tag:
        Subfolder name and prefix used for cached listing and series CSVs.
    start_date, end_date:
        Inclusive date range passed to the series endpoint.
    """
    headers = _build_headers()

    save_dir = output_root / item_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, catalogue_id in catalogue_map.items():
        listing_path = save_dir / f"{item_tag}_{name}_lst.csv"
        if listing_path.exists():
            index_df = pd.read_csv(listing_path, index_col=0)
        else:
            index_df = _fetch_catalogue(catalogue_id, headers)
            index_df.to_csv(listing_path)

        for _, row in index_df.iterrows():
            indicator_id = row["label"]
            series_path = save_dir / f"{item_tag}_{name}_{indicator_id}.csv"
            if series_path.exists():
                continue
            df = _fetch_series(
                indicator_id=indicator_id,
                field_id=row["id"],
                headers=headers,
                start_date=start_date,
                end_date=end_date,
            )
            df.to_csv(series_path)
            time.sleep(REQUEST_SLEEP_BASE + random.randint(1, REQUEST_SLEEP_JITTER))


if __name__ == "__main__":
    demo_catalogue = {
        "financial_market": 2012425,
    }
    download_indicators(
        catalogue_map=demo_catalogue,
        output_root=Path(__file__).resolve().parent / "raw",
        item_tag="financial_market_data",
    )
