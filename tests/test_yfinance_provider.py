from datetime import date

import numpy as np
import pandas as pd

from scanner.data.yfinance_provider import YFinanceProvider, reshape_download


def fake_yf_frame(data: dict[str, list[float] | None], dates) -> pd.DataFrame:
    """Build a frame shaped like yf.download(group_by='ticker', auto_adjust=False).
    data[symbol] = list of closes, or None for a ticker Yahoo returned no data for (all NaN)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    parts = {}
    for sym, closes in data.items():
        c = np.array(closes if closes is not None else [np.nan] * len(idx), dtype=float)
        parts[sym] = pd.DataFrame(
            {
                "Open": c,
                "High": c * 1.02,
                "Low": c * 0.98,
                "Close": c,
                "Adj Close": c * 0.99,  # must be ignored
                "Volume": np.where(np.isnan(c), np.nan, 1000.0),
            },
            index=idx,
        )
    return pd.concat(parts, axis=1)


def test_reshape_uses_close_not_adj_close_and_flags_missing():
    raw = fake_yf_frame({"AAA": [10, 11], "BAD": None}, ["2026-11-12", "2026-11-13"])
    out = reshape_download(raw, ["AAA", "BAD", "GONE"])
    assert out.failed == ["BAD", "GONE"]
    assert out.prices["symbol"].unique().tolist() == ["AAA"]
    assert out.prices["close"].tolist() == [10, 11]
    assert list(out.prices.columns) == ["symbol", "date", "open", "high", "low", "close", "volume"]


def test_reshape_handles_none_and_empty():
    assert reshape_download(None, ["X"]).failed == ["X"]
    assert reshape_download(pd.DataFrame(), ["X", "Y"]).failed == ["X", "Y"]


def test_fetch_batches_retries_whole_batch_failures_then_leftovers():
    dates = ["2026-11-12", "2026-11-13"]
    calls = []

    def download(tickers, **kw):
        calls.append(list(tickers))
        n = len(calls)
        if n == 1:  # first batch: simulated rate limit
            raise RuntimeError("Too Many Requests")
        data = {t: ([5, 6] if t != "MISSING" or n >= 4 else None) for t in tickers}
        return fake_yf_frame(data, dates)

    p = YFinanceProvider(batch_size=2, pause_seconds=0, max_retries=2, download=download, sleep=lambda s: None)
    batches = list(p.fetch_batches(["A", "B", "C", "MISSING"], date(2026, 11, 1), date(2026, 11, 13)))
    got = pd.concat([b.prices for b in batches])
    assert sorted(got["symbol"].unique()) == ["A", "B", "C", "MISSING"]
    # call 1 fails (rate limit), call 2 retries [A, B], call 3 is [C, MISSING] (MISSING empty),
    # call 4 is the leftover pass for MISSING
    assert calls == [["A", "B"], ["A", "B"], ["C", "MISSING"], ["MISSING"]]
    assert all(not b.failed for b in batches)


def test_end_date_is_made_inclusive():
    seen = {}

    def download(**kw):
        seen.update(kw)
        return fake_yf_frame({"A": [1.0]}, ["2026-11-13"])

    p = YFinanceProvider(download=download, sleep=lambda s: None)
    list(p.fetch_batches(["A"], date(2026, 11, 1), date(2026, 11, 13)))
    assert seen["start"] == "2026-11-01"
    assert seen["end"] == "2026-11-14"  # yfinance end is exclusive
    assert seen["auto_adjust"] is False
