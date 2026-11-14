"""NSE end-of-day file ("bhavcopy", UDiFF format) as a source for the newest India sessions.

Why: Yahoo's daily NSE candles arrive at least a day late (on 2026-11-14, 15+ hours after Friday's
close, Yahoo had no Friday bar for any large stock), while NSE publishes its own end-of-day file the
same evening. On a day both had (2026-11-12) the two agreed exactly for all 604 liquid stocks:
close, high, low and volume. So when Yahoo has not published a session yet, its bars are taken
from NSE's file; Yahoo's rows replace them once Yahoo catches up (fresh downloads overwrite cached
rows for the same date).

Caveat: the file is NOT adjusted for bonuses/splits. That only matters on a stock's ex-date. If one
falls on a filled day, the bar shows a jump (flagged by the one-day-move check) until Yahoo
publishes the day, at which point the 150-bar overlap check sees the re-adjusted history and
re-downloads it.

File: https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip
Columns used: TradDt, TckrSymb, SctySrs, OpnPric, HghPric, LwPric, ClsPric, TtlTradgVol.
No file (HTTP 404) means no session that day (holiday) or not published yet.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date
from typing import Callable, Iterable

import pandas as pd

from scanner.data.base import normalize_long

log = logging.getLogger(__name__)

URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"

FetchBytes = Callable[[str], "bytes | None"]  # None = 404 (no file)


def bhavcopy_url(day: date) -> str:
    return URL.format(ymd=day.strftime("%Y%m%d"))


def http_get_bytes(url: str) -> bytes | None:
    import requests

    resp = requests.get(url, timeout=30, headers={"User-Agent": "momentum-scanner/0.1"})
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


def read_zip_csv(content: bytes) -> str:
    """The CSV inside the downloaded zip, read in memory (nothing is extracted to disk)."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in the NSE zip, found {members}")
        return zf.read(members[0]).decode("utf-8-sig")


def parse_bhavcopy(text: str, series: Iterable[str] = ("EQ",), to_symbol: Callable[[str], str] = lambda s: f"{s}.NS"
                   ) -> pd.DataFrame:
    """Long OHLCV rows (symbol in Yahoo form, e.g. RELIANCE.NS) for the given series."""
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SctySrs"].str.strip().isin(list(series))]
    out = pd.DataFrame(
        {
            "symbol": df["TckrSymb"].str.strip().map(to_symbol),
            "date": pd.to_datetime(df["TradDt"].str.strip()),
            "open": df["OpnPric"],
            "high": df["HghPric"],
            "low": df["LwPric"],
            "close": df["ClsPric"],
            "volume": df["TtlTradgVol"],
        }
    )
    return normalize_long(out)


def fetch_bhavcopy(day: date, series: Iterable[str] = ("EQ",), fetch: FetchBytes = http_get_bytes) -> pd.DataFrame | None:
    """Rows for `day`, or None when NSE has no file for it (holiday, or not published yet)."""
    content = fetch(bhavcopy_url(day))
    if content is None:
        return None
    rows = parse_bhavcopy(read_zip_csv(content), series)
    rows = rows[rows["date"] == pd.Timestamp(day)]
    return rows
