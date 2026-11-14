"""NSE end-of-day file (bhavcopy) parsing and filling sessions the price provider hasn't published."""

import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scanner.data.cache import PriceCache, fill_missing_sessions
from scanner.data.nse_bhavcopy import bhavcopy_url, fetch_bhavcopy, parse_bhavcopy, read_zip_csv

FIX = Path(__file__).parent / "fixtures" / "nse_bhavcopy_20261113_sample.csv"
TEXT = FIX.read_text()


def zipped(text: str, name="BhavCopy_NSE_CM_0_0_0_20261113_F_0000.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


# --- parsing ----------------------------------------------------------------------------------


def test_parse_keeps_eq_series_with_yahoo_symbols():
    rows = parse_bhavcopy(TEXT, ["EQ"])
    assert rows["symbol"].tolist() == ["BAJAJ-AUTO.NS", "M&M.NS", "NIFTYBEES.NS", "RELIANCE.NS"]  # BE, SM dropped
    r = rows.set_index("symbol").loc["RELIANCE.NS"]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (1336.60, 1350.70, 1324.00, 1339.70, 13756247)
    assert rows["date"].unique().tolist() == [pd.Timestamp("2026-11-13")]


def test_url_and_zip_roundtrip():
    assert bhavcopy_url(date(2026, 11, 13)).endswith("BhavCopy_NSE_CM_0_0_0_20261113_F_0000.csv.zip")
    assert read_zip_csv(zipped(TEXT)) == TEXT


def test_zip_with_unexpected_contents_is_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "y")
    with pytest.raises(ValueError):
        read_zip_csv(buf.getvalue())


def test_fetch_returns_none_on_missing_file():
    assert fetch_bhavcopy(date(2026, 11, 10), fetch=lambda url: None) is None
    rows = fetch_bhavcopy(date(2026, 11, 13), fetch=lambda url: zipped(TEXT))
    assert len(rows) == 4


# --- filling sessions -------------------------------------------------------------------------

THU = date(2026, 11, 12)
FRI = date(2026, 11, 13)
DATES = pd.bdate_range(end="2026-11-12", periods=30)


def history(symbol, dates, close=100.0):
    c = np.full(len(dates), close)
    return pd.DataFrame({"symbol": symbol, "date": dates, "open": c, "high": c * 1.02, "low": c * 0.98,
                         "close": c, "volume": 1_000.0})


def day_rows(day, symbols, close=101.0):
    return pd.DataFrame({"symbol": symbols, "date": pd.Timestamp(day), "open": close, "high": close,
                         "low": close, "close": close, "volume": 500.0})


@pytest.fixture
def cache_with_history(tmp_path):
    cache = PriceCache(tmp_path)
    prices = pd.concat(
        [history(s, DATES) for s in ("A", "B", "C")]
        + [history("A", pd.DatetimeIndex([pd.Timestamp(FRI)]), close=99.0)]  # provider already has A's Friday
    )
    cache.save(prices)
    return cache, cache.load()


def test_fills_only_missing_universe_symbols(cache_with_history):
    cache, prices = cache_with_history
    calls = []

    def fetch(day):
        calls.append(day)
        return day_rows(day, ["A", "B", "ETF"]) if day == FRI else None

    out, filled = fill_missing_sessions(cache, prices, ["A", "B", "C"], FRI, fetch, max_days=5)
    assert calls == [FRI]
    assert filled == {"2026-11-13": 1}  # only B: A already had Friday, ETF isn't in the universe, C not in file
    fri = out[out["date"] == pd.Timestamp(FRI)].set_index("symbol")["close"].to_dict()
    assert fri == {"A": 99.0, "B": 101.0}  # the provider's A bar is untouched
    assert cache.read_meta()["filled_from_fallback"] == {"2026-11-13": 1}
    assert len(cache.load()) == len(out)  # saved


def test_skips_weekends_and_holidays_and_respects_max_days(cache_with_history):
    cache, prices = cache_with_history
    calls = []

    def fetch(day):
        calls.append(day)
        return None  # pretend every weekday is a holiday

    mon = date(2026, 11, 16)
    fill_missing_sessions(cache, prices, ["A", "B", "C"], mon, fetch, max_days=5)
    assert calls == [FRI, mon]  # Sat/Sun never asked
    calls.clear()
    fill_missing_sessions(cache, prices, ["A", "B", "C"], mon, fetch, max_days=1)
    assert calls == [mon]  # only the cutoff day itself


def test_nothing_to_do_when_provider_is_current(cache_with_history):
    cache, prices = cache_with_history
    out, filled = fill_missing_sessions(cache, prices, ["A", "B", "C"], THU, lambda d: pytest.fail("called"), 5)
    assert filled == {} and out is prices


def test_failing_source_does_not_break_the_run(cache_with_history):
    cache, prices = cache_with_history

    def fetch(day):
        raise ConnectionError("NSE refused the connection")

    out, filled = fill_missing_sessions(cache, prices, ["A", "B", "C"], FRI, fetch, 5)
    assert filled == {} and len(out) == len(prices)
