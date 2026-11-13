"""US market adapter: listed-stock universe (Nasdaq Trader symbol directory) + daily OHLCV.

Universe source: https://www.nasdaqtrader.com/dynamic/SymDir/
  nasdaqlisted.txt  Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
  otherlisted.txt   ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
Both are pipe-delimited and end with a "File Creation Time: ..." line. otherlisted exchange codes:
  N = NYSE, A = NYSE American, P = NYSE Arca, Z = Cboe BZX, V = IEX, F = Texas Stock Exchange.

Prices: yfinance (see yfinance_provider.py), behind the PriceProvider interface.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Callable

import pandas as pd

from scanner.config import MarketConfig, UniverseConfig
from scanner.data.cache import PriceCache
from scanner.data.yfinance_provider import YFinanceProvider

log = logging.getLogger(__name__)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

OTHER_EXCHANGE_CODES = {
    "N": "NYSE",
    "A": "NYSE_AMERICAN",
    "P": "NYSE_ARCA",
    "Z": "CBOE_BZX",
    "V": "IEX",
    "F": "TXSE",
}

UNIVERSE_COLUMNS = ["symbol", "source_symbol", "name", "exchange", "is_etf", "is_test"]

FetchText = Callable[[str], str]


def _read_pipe_file(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
    first = df.columns[0]
    # Drop the trailer line ("File Creation Time: ...") and any blank rows.
    df = df[~df[first].str.startswith("File Creation Time") & (df[first].str.strip() != "")]
    return df.apply(lambda col: col.str.strip())


def to_yahoo_symbol(symbol: str) -> str:
    """Nasdaq/ACT class-share notation uses '.', Yahoo uses '-': BRK.B -> BRK-B."""
    return symbol.replace(".", "-")


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    df = _read_pipe_file(text)
    out = pd.DataFrame(
        {
            "source_symbol": df["Symbol"],
            "name": df["Security Name"],
            "exchange": "NASDAQ",
            "is_etf": df["ETF"].eq("Y"),
            "is_test": df["Test Issue"].eq("Y"),
        }
    )
    out.insert(0, "symbol", out["source_symbol"].map(to_yahoo_symbol))
    return out[UNIVERSE_COLUMNS].reset_index(drop=True)


def parse_other_listed(text: str) -> pd.DataFrame:
    df = _read_pipe_file(text)
    out = pd.DataFrame(
        {
            "source_symbol": df["ACT Symbol"],
            "name": df["Security Name"],
            "exchange": df["Exchange"].map(lambda c: OTHER_EXCHANGE_CODES.get(c, f"OTHER_{c}")),
            "is_etf": df["ETF"].eq("Y"),
            "is_test": df["Test Issue"].eq("Y"),
        }
    )
    out.insert(0, "symbol", out["source_symbol"].map(to_yahoo_symbol))
    return out[UNIVERSE_COLUMNS].reset_index(drop=True)


def filter_universe(df: pd.DataFrame, ucfg: UniverseConfig) -> pd.DataFrame:
    """Keep common stocks on the configured exchanges (SPEC Stage 1 'listed stocks only').

    Order: exchange -> ETF flag -> test-issue flag -> symbol regexes -> security-name regexes.
    """
    keep = df["exchange"].isin(ucfg.exchanges)
    if ucfg.exclude_etfs:
        keep &= ~df["is_etf"]
    if ucfg.exclude_test_issues:
        keep &= ~df["is_test"]
    for pat in ucfg.exclude_symbol_patterns:
        keep &= ~df["source_symbol"].str.contains(pat, regex=True)
    if ucfg.exclude_name_patterns:
        name_re = "|".join(f"(?:{p})" for p in ucfg.exclude_name_patterns)
        keep &= ~df["name"].str.contains(name_re, flags=re.IGNORECASE, regex=True)
    out = df[keep].drop_duplicates(subset="symbol", keep="first")
    return out.sort_values("symbol").reset_index(drop=True)


def _http_get_text(url: str) -> str:
    import requests

    resp = requests.get(url, timeout=30, headers={"User-Agent": "momentum-scanner/0.1"})
    resp.raise_for_status()
    return resp.text


def load_universe(
    cfg: MarketConfig,
    *,
    now: datetime | None = None,
    fetch_text: FetchText = _http_get_text,
    force: bool = False,
) -> pd.DataFrame:
    """Filtered US universe: columns symbol (Yahoo format), source_symbol, name, exchange.

    Cached as universe.csv in the market cache dir and refreshed when older than
    `data.universe_max_age_hours`. If the download fails, the cached copy is used with a warning.
    """
    now = now or datetime.now(timezone.utc)
    cache = PriceCache(cfg.data.cache_path())
    meta = cache.read_meta()
    fetched_at = meta.get("universe_fetched_at")
    fresh_enough = (
        fetched_at is not None
        and cache.universe_path.exists()
        and now - datetime.fromisoformat(fetched_at) < timedelta(hours=cfg.data.universe_max_age_hours)
    )
    if fresh_enough and not force:
        return pd.read_csv(cache.universe_path, dtype=str, keep_default_na=False)

    try:
        raw = pd.concat(
            [
                parse_nasdaq_listed(fetch_text(NASDAQ_LISTED_URL)),
                parse_other_listed(fetch_text(OTHER_LISTED_URL)),
            ],
            ignore_index=True,
        )
    except Exception as e:
        if cache.universe_path.exists():
            log.warning("universe download failed (%s); using cached list from %s", e, fetched_at)
            return pd.read_csv(cache.universe_path, dtype=str, keep_default_na=False)
        raise

    uni = filter_universe(raw, cfg.universe)[["symbol", "source_symbol", "name", "exchange"]]
    cache.dir.mkdir(parents=True, exist_ok=True)
    uni.to_csv(cache.universe_path, index=False)
    meta = cache.read_meta()
    meta["universe_fetched_at"] = now.isoformat()
    meta["universe_raw_count"] = int(len(raw))
    meta["universe_count"] = int(len(uni))
    cache.write_meta(meta)
    log.info("US universe: %d listed rows -> %d common stocks", len(raw), len(uni))
    return uni


TRADINGVIEW_EXCHANGE = {"NASDAQ": "NASDAQ", "NYSE": "NYSE", "NYSE_AMERICAN": "AMEX"}


def chart_url(symbol: str, exchange: str | None = None, source_symbol: str | None = None) -> str:
    """TradingView chart link. TradingView uses the exchange prefix and '.' class notation
    (NYSE:BRK.B), so prefer the Nasdaq Trader symbol over the Yahoo one (BRK-B)."""
    from urllib.parse import quote

    tv_symbol = source_symbol or symbol.replace("-", ".")
    prefix = TRADINGVIEW_EXCHANGE.get(exchange or "")
    full = f"{prefix}:{tv_symbol}" if prefix else tv_symbol
    return "https://www.tradingview.com/chart/?symbol=" + quote(full, safe=":.")


def make_provider(cfg: MarketConfig) -> YFinanceProvider:
    d = cfg.data
    return YFinanceProvider(
        batch_size=d.batch_size, pause_seconds=d.batch_pause_seconds, max_retries=d.max_retries
    )
