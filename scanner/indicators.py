"""Price/volume indicators used by the scans.

Every function accepts either
  * a pandas Series indexed by date (one symbol), or
  * a wide DataFrame indexed by date with one column per symbol (a "panel"),
and returns the same shape. Time-series functions work down the date axis; `rs_rank_pct` works
across symbols. Rolling windows require a full window of data (min_periods = n): values are NaN
until n bars exist, so a stock with too little history is never scored on a partial window.

"n bars ago" means n rows back in the input. Callers align panels to the market's trading calendar
so one row = one trading session.
"""

from __future__ import annotations

from typing import TypeVar

import pandas as pd

T = TypeVar("T", pd.Series, pd.DataFrame)


def _check_window(n: int) -> None:
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"window must be a positive integer, got {n!r}")


def pct_return(close: T, n: int) -> T:
    """Simple return over n bars.

    Formula: return_t = close_t / close_{t-n} - 1
    Example: SPEC's 1-month leader rule "close >= 1.25 x close 22 trading days ago" is
    pct_return(close, 22) >= 0.25.
    """
    _check_window(n)
    return close / close.shift(n) - 1


def adr_pct(high: T, low: T, n: int = 20) -> T:
    """Average Daily Range in percent (SPEC definition).

    Formula: ADR%_t = 100 x ( mean over the last n bars of (high / low) - 1 )
    e.g. a stock whose high is on average 6% above its low over 20 days has ADR% = 6.
    Bars with a non-positive low are treated as missing (they would divide by zero).
    """
    _check_window(n)
    ratio = high / low.where(low > 0)
    return 100 * (ratio.rolling(n, min_periods=n).mean() - 1)


def sma(series: T, n: int) -> T:
    """Simple moving average.

    Formula: SMA_t = (x_t + x_{t-1} + ... + x_{t-n+1}) / n
    """
    _check_window(n)
    return series.rolling(n, min_periods=n).mean()


def traded_value(close: T, volume: T) -> T:
    """Value traded per bar in the market's currency: close x volume.

    This is "dollar volume" in the US and "turnover" in India. It uses the close as a proxy for
    the day's average traded price (the usual screener approximation).
    """
    return close * volume


def avg_traded_value(close: T, volume: T, n: int = 20) -> T:
    """n-day average of close x volume (SPEC: 20-day average dollar volume).

    Formula: AvgTV_t = mean over the last n bars of (close x volume)
    """
    _check_window(n)
    return traded_value(close, volume).rolling(n, min_periods=n).mean()


# SPEC.md uses US naming; keep aliases so either name reads naturally.
dollar_volume = traded_value
avg_dollar_volume = avg_traded_value


def rs_rank_pct(values: T) -> T:
    """Relative-strength percentile rank across symbols, 0-100 (100 = strongest).

    Formula: pct_i = 100 x rank_i / N, where rank_i is symbol i's ascending rank among the N
    symbols with a value (ties share the highest rank of their group) and NaN values are left out.
    So with N = 100 distinct values the strongest gets 100 and the weakest gets 1.

    "Keep the top p%" is then: pct > 100 - p (with N = 100 and p = 7 that keeps exactly 7 names;
    tied names at the cut-off are kept together).

    Series input = one cross-section (index = symbols). DataFrame input = panel; each row (date)
    is ranked across its columns (symbols).
    """
    if isinstance(values, pd.DataFrame):
        return values.rank(axis=1, method="max", pct=True) * 100
    return values.rank(method="max", pct=True) * 100
