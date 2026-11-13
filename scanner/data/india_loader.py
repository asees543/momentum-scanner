"""India (NSE) market adapter: equity universe from NSE's equity list + daily OHLCV from Yahoo ".NS".

Universe source: https://archives.nseindia.com/content/equities/EQUITY_L.csv
  SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE
  (note the leading spaces in most column names). SPEC: keep series EQ only (BE, BZ, ... dropped).

Prices: yfinance with "<SYMBOL>.NS" tickers, which Yahoo adjusts for splits AND bonus issues.
Checked before building: Reliance's 1:1 bonus (2024-10-28) and Bajaj Finance's 4:1 bonus + 1:2 split
(2025-06-16) show no fake drop. Bhavcopy (unadjusted) is a later option; it would need corporate
actions applied by hand.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd

from scanner.config import MarketConfig, UniverseConfig
from scanner.data.cache import PriceCache, load_cached_universe
from scanner.data.us_loader import FetchText, _http_get_text
from scanner.data.yfinance_provider import YFinanceProvider

log = logging.getLogger(__name__)

EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


def to_yahoo_symbol(symbol: str) -> str:
    """NSE symbol -> Yahoo ticker: RELIANCE -> RELIANCE.NS, M&M -> M&M.NS, BAJAJ-AUTO -> BAJAJ-AUTO.NS."""
    return f"{symbol}.NS"


def parse_equity_list(text: str) -> pd.DataFrame:
    """Columns: symbol (Yahoo), source_symbol (NSE), name, exchange ("NSE"), series."""
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda col: col.str.strip())
    df = df[df["SYMBOL"] != ""]
    out = pd.DataFrame(
        {
            "source_symbol": df["SYMBOL"],
            "name": df["NAME OF COMPANY"],
            "exchange": "NSE",
            "series": df["SERIES"],
        }
    )
    out.insert(0, "symbol", out["source_symbol"].map(to_yahoo_symbol))
    return out.reset_index(drop=True)


def filter_universe(df: pd.DataFrame, ucfg: UniverseConfig) -> pd.DataFrame:
    """Keep the configured series (SPEC: EQ only)."""
    keep = df["exchange"].isin(ucfg.exchanges)
    if ucfg.series:
        keep &= df["series"].isin(ucfg.series)
    return df[keep].drop_duplicates(subset="symbol").sort_values("symbol").reset_index(drop=True)


def load_universe(
    cfg: MarketConfig,
    *,
    now: datetime | None = None,
    fetch_text: FetchText = _http_get_text,
    force: bool = False,
) -> pd.DataFrame:
    """Filtered NSE universe, cached as universe.csv like the US list."""

    def build() -> tuple[pd.DataFrame, int]:
        raw = parse_equity_list(fetch_text(EQUITY_LIST_URL))
        return filter_universe(raw, cfg.universe), len(raw)

    return load_cached_universe(
        PriceCache(cfg.data.cache_path()),
        cfg.data.universe_max_age_hours,
        build,
        now or datetime.now(timezone.utc),
        force,
        label="NSE",
    )


def chart_url(symbol: str, exchange: str | None = None, source_symbol: str | None = None) -> str:
    """TradingView chart link. TradingView writes NSE symbols with '_' for '&' and '-':
    M&M -> NSE:M_M, BAJAJ-AUTO -> NSE:BAJAJ_AUTO."""
    nse = source_symbol or symbol.removesuffix(".NS")
    tv = nse.replace("&", "_").replace("-", "_")
    return "https://www.tradingview.com/chart/?symbol=" + quote(f"NSE:{tv}", safe=":")


def make_provider(cfg: MarketConfig) -> YFinanceProvider:
    d = cfg.data
    return YFinanceProvider(batch_size=d.batch_size, pause_seconds=d.batch_pause_seconds, max_retries=d.max_retries)
