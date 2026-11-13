"""yfinance implementation of PriceProvider (used for US now, NSE `.NS` tickers in Phase 4).

yfinance is free but unofficial: Yahoo can rate-limit or change responses. This module keeps all
yfinance specifics in one place so a paid provider can replace it.

Adjustment: we call yf.download(auto_adjust=False) and use Open/High/Low/Close/Volume as returned.
Yahoo already adjusts those for splits (prices divided, volume multiplied by the split ratio) but NOT
for dividends; the dividend-adjusted series is the separate "Adj Close" column, which we ignore.
Verified on NVDA's 10-for-1 split (2024-06-10): closes run 120.89 -> 121.79 across the split day.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Callable, Iterator, Sequence

import pandas as pd

from scanner.data.base import FetchBatch, normalize_long

log = logging.getLogger(__name__)

_YF_FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}

DownloadFn = Callable[..., pd.DataFrame]


def _default_download(**kwargs) -> pd.DataFrame:
    import yfinance as yf

    # yfinance prints one line per failed ticker; we report failures ourselves.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    return yf.download(**kwargs)


def reshape_download(raw: pd.DataFrame | None, symbols: Sequence[str]) -> FetchBatch:
    """Turn yf.download(group_by='ticker') output into long rows + list of symbols with no data."""
    if raw is None or raw.empty:
        return FetchBatch(prices=normalize_long(pd.DataFrame(columns=["symbol", "date", *_YF_FIELDS.values()])),
                          failed=list(symbols))
    frames, failed = [], []
    if not isinstance(raw.columns, pd.MultiIndex):
        # Defensive: a single ticker without a ticker level.
        raw = pd.concat({symbols[0]: raw}, axis=1)
    present = set(raw.columns.get_level_values(0))
    for sym in symbols:
        if sym not in present:
            failed.append(sym)
            continue
        sub = raw[sym]
        cols = [c for c in _YF_FIELDS if c in sub.columns]
        sub = sub[cols].rename(columns=_YF_FIELDS).dropna(subset=["close"])
        if sub.empty:
            failed.append(sym)
            continue
        sub = sub.rename_axis("date").reset_index()
        sub.insert(0, "symbol", sym)
        frames.append(sub)
    prices = (
        normalize_long(pd.concat(frames, ignore_index=True))
        if frames
        else normalize_long(pd.DataFrame(columns=["symbol", "date", *_YF_FIELDS.values()]))
    )
    return FetchBatch(prices=prices, failed=failed)


class YFinanceProvider:
    name = "yfinance"

    def __init__(
        self,
        batch_size: int = 100,
        pause_seconds: float = 1.0,
        max_retries: int = 3,
        download: DownloadFn | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.batch_size = batch_size
        self.pause_seconds = pause_seconds
        self.max_retries = max_retries
        self._download = download or _default_download
        self._sleep = sleep

    def _download_batch(self, symbols: list[str], start: date, end: date | None) -> FetchBatch:
        kwargs = dict(
            tickers=symbols,
            start=start.isoformat(),
            # yfinance's `end` is exclusive; our `end` is inclusive.
            end=(end + timedelta(days=1)).isoformat() if end else None,
            interval="1d",
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        try:
            raw = self._download(**kwargs)
        except Exception as e:  # network errors, YFRateLimitError, JSON hiccups
            log.warning("yfinance batch of %d failed: %s", len(symbols), e)
            raw = None
        return reshape_download(raw, symbols)

    def fetch_batches(
        self, symbols: Sequence[str], start: date, end: date | None = None
    ) -> Iterator[FetchBatch]:
        """Download in batches. A batch where EVERY symbol fails usually means rate limiting or a
        network problem, so it is retried with exponential backoff. Partial failures are usually
        genuinely missing tickers; they are collected and retried once at the end."""
        symbols = list(dict.fromkeys(symbols))
        leftovers: list[str] = []
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i : i + self.batch_size]
            if i:
                self._sleep(self.pause_seconds)
            result = self._download_batch(batch, start, end)
            attempt = 0
            while len(result.failed) == len(batch) and attempt < self.max_retries:
                attempt += 1
                wait = self.pause_seconds * 5 * 2**attempt
                log.warning("whole batch failed; retry %d/%d in %.0fs", attempt, self.max_retries, wait)
                self._sleep(wait)
                result = self._download_batch(batch, start, end)
            leftovers.extend(result.failed)
            yield FetchBatch(prices=result.prices, failed=[])
        # One more pass over symbols that returned nothing.
        for i in range(0, len(leftovers), self.batch_size):
            batch = leftovers[i : i + self.batch_size]
            self._sleep(self.pause_seconds)
            yield self._download_batch(batch, start, end)
