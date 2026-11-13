"""Market-agnostic data shapes shared by every loader and by the scans.

Prices travel through the code in two shapes:

* **long**: one row per (symbol, date) with columns `symbol, date, open, high, low, close, volume`.
  This is what providers return and what the parquet cache stores.
* **panels**: one wide DataFrame per field (index = trading date, columns = symbol). This is what
  the indicators and scans use, so they compute every symbol at once.

Prices are split-adjusted (and bonus-adjusted for India). Scans never know which market they run on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterator, Protocol, Sequence

import pandas as pd

from scanner.config import DataConfig

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
LONG_COLUMNS = ["symbol", "date", *OHLCV_COLUMNS]


def empty_long() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in LONG_COLUMNS})
    df["symbol"] = df["symbol"].astype("str")
    df["date"] = pd.to_datetime(df["date"])
    return df


def normalize_long(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the canonical long schema: typed columns, rows with a close only, de-duplicated
    on (symbol, date) keeping the LAST occurrence, sorted by symbol then date."""
    missing = set(LONG_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"price frame missing columns: {sorted(missing)}")
    out = df[LONG_COLUMNS].copy()
    out["symbol"] = out["symbol"].astype("str")
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    for c in OHLCV_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = out[out["close"].notna()]
    out = out.drop_duplicates(subset=["symbol", "date"], keep="last")
    return out.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def last_complete_session_date(now: datetime, dcfg: DataConfig) -> date:
    """Latest calendar date whose daily bar can be treated as final.

    Before (market close + buffer) in the exchange's timezone, today's bar is still forming, so the
    cutoff is yesterday. Weekends/holidays need no special case: there is simply no bar that day.
    """
    local = now.astimezone(dcfg.tz)
    close_dt = datetime.combine(local.date(), dcfg.close_time, tzinfo=dcfg.tz)
    if local < close_dt + timedelta(minutes=dcfg.close_buffer_minutes):
        return local.date() - timedelta(days=1)
    return local.date()


def drop_incomplete_bars(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Remove bars dated after `cutoff` (a still-forming intraday bar)."""
    return df[df["date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)


@dataclass
class FetchBatch:
    """One provider batch: successfully fetched long-format rows, plus symbols that returned nothing."""

    prices: pd.DataFrame
    failed: list[str] = field(default_factory=list)


class PriceProvider(Protocol):
    """Anything that can download daily split-adjusted OHLCV. yfinance today; Polygon/Tiingo/Alpaca
    can be dropped in by implementing this one method."""

    name: str

    def fetch_batches(
        self, symbols: Sequence[str], start: date, end: date | None = None
    ) -> Iterator[FetchBatch]:
        """Yield batches covering `symbols` for dates in [start, end] (end inclusive; None = latest)."""
        ...


@dataclass(frozen=True)
class Panels:
    """Wide OHLCV frames (index = trading date, columns = symbol), all with identical axes."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> pd.Index:
        return self.close.columns

    @classmethod
    def from_long(cls, df: pd.DataFrame, min_coverage: float = 0.5) -> Panels:
        """Pivot long rows into panels aligned to the market's trading calendar.

        The calendar is every date on which at least `min_coverage` x (median number of symbols
        per date) symbols have a bar. This drops stray dates where one bad ticker reports a bar on
        a holiday, which would otherwise shift every "n bars ago" by one for all symbols.
        Missing bars inside the calendar stay NaN (no forward-fill): indicators that need them
        return NaN and the symbol drops out of that scan rather than being scored on invented data.
        """
        if df.empty:
            e = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
            return cls(e, e, e, e, e)
        counts = df.groupby("date")["symbol"].size()
        calendar = counts.index[counts >= min_coverage * counts.median()].sort_values()
        fields = {}
        for c in OHLCV_COLUMNS:
            wide = df.pivot(index="date", columns="symbol", values=c)
            fields[c] = wide.reindex(calendar).sort_index(axis=1)
            fields[c].index.name = "date"
        return cls(**fields)

    def as_of(self, when: date | pd.Timestamp) -> Panels:
        """Panels truncated to dates <= `when` (point-in-time view, no lookahead)."""
        ts = pd.Timestamp(when)
        return Panels(**{c: getattr(self, c).loc[:ts] for c in OHLCV_COLUMNS})

    def symbol_frame(self, symbol: str) -> pd.DataFrame:
        """SPEC shape for one symbol: columns date, open, high, low, close, volume."""
        out = pd.DataFrame({c: getattr(self, c)[symbol] for c in OHLCV_COLUMNS})
        out = out.dropna(subset=["close"]).reset_index()
        return out
