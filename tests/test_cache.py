from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from scanner.config import load_config
from scanner.data.base import FetchBatch, Panels, drop_incomplete_bars, last_complete_session_date
from scanner.data.cache import PriceCache, find_mismatched, update_prices

SESSIONS = pd.bdate_range("2026-01-02", "2026-11-13")  # stand-in trading calendar


class FakeMarket:
    """A pretend provider holding the 'true' current Yahoo history for each symbol.
    Tests mutate `history` to simulate new bars, splits, and missing tickers."""

    name = "fake"

    def __init__(self, history: dict[str, pd.DataFrame]):
        self.history = history
        self.requests: list[tuple[tuple[str, ...], date, date]] = []

    def fetch_batches(self, symbols, start, end=None):
        self.requests.append((tuple(symbols), start, end))
        frames, failed = [], []
        for s in symbols:
            h = self.history.get(s)
            if h is None:
                failed.append(s)
                continue
            m = (h["date"] >= pd.Timestamp(start)) & (h["date"] <= pd.Timestamp(end or "2100-01-01"))
            frames.append(h[m])
        yield FetchBatch(prices=pd.concat(frames) if frames else pd.DataFrame(), failed=failed)


def make_history(symbol, closes, dates):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"symbol": symbol, "date": dates, "open": c, "high": c * 1.02, "low": c * 0.98, "close": c,
         "volume": 1_000.0}
    )


def run_update(cache, market, symbols, cutoff, **kw):
    params = dict(history_start=date(2021, 1, 1), cutoff=cutoff, overlap_bars=10, tolerance_pct=0.5)
    params.update(kw)
    return update_prices(cache, market, symbols, **params)


@pytest.fixture
def cache(tmp_path):
    return PriceCache(tmp_path)


def test_first_run_downloads_full_history_and_saves(cache):
    d = SESSIONS[:-5]
    market = FakeMarket({"AAA": make_history("AAA", np.linspace(10, 20, len(d)), d)})
    df, rep = run_update(cache, market, ["AAA"], d[-1].date())
    assert rep.new_symbols == 1 and not rep.failed
    assert len(df) == len(d)
    assert len(cache.load()) == len(d)
    assert cache.read_meta()["checked"] == {"AAA": d[-1].date().isoformat()}
    assert market.requests[0][1] == date(2021, 1, 1)  # full history from history_start


def test_incremental_update_fetches_only_overlap_and_appends(cache):
    d_all = SESSIONS
    closes = np.linspace(10, 20, len(d_all))
    first = d_all[:-5]
    market = FakeMarket({"AAA": make_history("AAA", closes[: len(first)], first)})
    run_update(cache, market, ["AAA"], first[-1].date())

    market.history["AAA"] = make_history("AAA", closes, d_all)  # 5 new sessions appear
    market.requests.clear()
    df, rep = run_update(cache, market, ["AAA"], d_all[-1].date())
    assert rep.incremental == 1 and rep.full_refetch == []
    (syms, start, end), = market.requests
    assert start == first[-10].date()  # 10-bar overlap, not full history
    assert len(df) == len(d_all)
    assert df["close"].tolist() == pytest.approx(closes.tolist())


def test_split_triggers_full_redownload(cache):
    d_all = SESSIONS
    first = d_all[:-5]
    pre = np.full(len(first), 100.0)
    market = FakeMarket({"AAA": make_history("AAA", pre, first), "BBB": make_history("BBB", pre, first)})
    run_update(cache, market, ["AAA", "BBB"], first[-1].date())

    # AAA does a 2-for-1 split: Yahoo now reports ALL history halved, plus new bars at ~50.
    market.history["AAA"] = make_history("AAA", np.full(len(d_all), 50.0), d_all)
    market.history["BBB"] = make_history("BBB", np.full(len(d_all), 100.0), d_all)
    df, rep = run_update(cache, market, ["AAA", "BBB"], d_all[-1].date())
    assert rep.full_refetch == ["AAA"]
    aaa = df[df["symbol"] == "AAA"]
    assert len(aaa) == len(d_all)
    assert (aaa["close"] == 50.0).all()  # no stale pre-split 100s left -> no fake -50% day
    assert (df.loc[df["symbol"] == "BBB", "close"] == 100.0).all()


def test_revised_last_bar_replaces_cached_value(cache):
    d = SESSIONS[:-3]
    h = make_history("AAA", np.full(len(d), 10.0), d)
    market = FakeMarket({"AAA": h.copy()})
    run_update(cache, market, ["AAA"], d[-1].date())
    revised = h.copy()
    revised.loc[revised.index[-1], "volume"] = 5_000.0  # Yahoo finalises the last bar's volume
    market.history["AAA"] = revised
    df, rep = run_update(cache, market, ["AAA"], d[-1].date(), refresh=True)
    assert df["volume"].iloc[-1] == 5_000.0
    assert rep.full_refetch == []


def test_same_cutoff_skips_known_symbols_unless_refresh(cache):
    d = SESSIONS[:-3]
    market = FakeMarket({"AAA": make_history("AAA", np.full(len(d), 10.0), d)})
    run_update(cache, market, ["AAA"], d[-1].date())
    market.requests.clear()
    _, rep = run_update(cache, market, ["AAA"], d[-1].date())
    assert rep.already_current == 1 and market.requests == []
    _, rep = run_update(cache, market, ["AAA"], d[-1].date(), refresh=True)
    assert rep.already_current == 0 and len(market.requests) == 1


def test_partial_run_does_not_make_full_run_skip_other_symbols(cache):
    d = SESSIONS
    market = FakeMarket({s: make_history(s, np.full(len(d) - 1, 10.0), d[:-1]) for s in "ABC"})
    run_update(cache, market, list("ABC"), d[-2].date())  # day 1: full run
    for s in "ABC":
        market.history[s] = make_history(s, np.full(len(d), 10.0), d)
    run_update(cache, market, ["A"], d[-1].date())  # day 2: quick run on one symbol
    market.requests.clear()
    df, rep = run_update(cache, market, list("ABC"), d[-1].date())  # day 2: full run
    assert rep.already_current == 1  # only A
    assert sorted(market.requests[0][0]) == ["B", "C"]
    assert df.groupby("symbol")["date"].max().eq(d[-1]).all()


def test_failed_symbol_keeps_cached_rows(cache):
    d = SESSIONS[:-5]
    market = FakeMarket({"AAA": make_history("AAA", np.full(len(d), 10.0), d),
                         "BBB": make_history("BBB", np.full(len(d), 10.0), d)})
    run_update(cache, market, ["AAA", "BBB"], d[-1].date())
    del market.history["BBB"]  # Yahoo stops returning BBB
    df, rep = run_update(cache, market, ["AAA", "BBB"], SESSIONS[-1].date())
    assert rep.failed == ["BBB"]
    assert (df["symbol"] == "BBB").sum() == len(d)


def test_failed_symbols_are_retried_next_run(cache):
    d = SESSIONS[:-5]
    full = {s: make_history(s, np.full(len(SESSIONS), 10.0), SESSIONS) for s in "ABC"}
    market = FakeMarket({s: make_history(s, np.full(len(d), 10.0), d) for s in "ABC"})
    run_update(cache, market, list("ABC"), d[-1].date())
    market.history = {"A": full["A"]}  # B and C fail (e.g. rate-limited)
    _, rep = run_update(cache, market, list("ABC"), SESSIONS[-1].date())
    assert rep.failed == ["B", "C"]
    market.history = full
    market.requests.clear()
    df, rep = run_update(cache, market, list("ABC"), SESSIONS[-1].date())
    assert rep.already_current == 1 and not rep.failed
    assert sorted(market.requests[0][0]) == ["B", "C"]
    assert df.groupby("symbol")["date"].max().eq(SESSIONS[-1]).all()


def test_stale_symbol_without_overlap_gets_full_redownload(cache):
    d = SESSIONS[:-5]
    market = FakeMarket({"AAA": make_history("AAA", np.full(len(d), 10.0), d),
                         "OLD": make_history("OLD", np.full(len(d) - 30, 10.0), d[:-30])})
    run_update(cache, market, ["AAA", "OLD"], d[-1].date())
    market.history["OLD"] = make_history("OLD", np.full(len(SESSIONS), 10.0), SESSIONS)
    market.history["AAA"] = make_history("AAA", np.full(len(SESSIONS), 10.0), SESSIONS)
    df, rep = run_update(cache, market, ["AAA", "OLD"], SESSIONS[-1].date())
    assert rep.full_refetch == ["OLD"]
    assert (df["symbol"] == "OLD").sum() == len(SESSIONS)


def test_find_mismatched_tolerance():
    d = SESSIONS[:3]
    cached = make_history("A", [100, 100, 100], d)
    fresh = make_history("A", [100, 100, 100.4], d)  # 0.4% < 0.5% tolerance
    assert find_mismatched(cached, fresh, 0.5) == []
    fresh = make_history("A", [100, 100, 100.6], d)
    assert find_mismatched(cached, fresh, 0.5) == ["A"]


def test_save_is_atomic_and_roundtrips(cache):
    d = SESSIONS[:5]
    df = make_history("AAA", [1, 2, 3, 4, 5], d)
    cache.save(df)
    assert not list(cache.dir.glob("*.tmp"))
    back = cache.load()
    assert back["close"].tolist() == [1, 2, 3, 4, 5]
    assert str(back["date"].dtype).startswith("datetime64")


# --- partial-bar handling and panels -----------------------------------------------------------


def test_last_complete_session_respects_close_and_buffer():
    dcfg = load_config("us").data  # 16:00 New York + 30 min buffer
    # 2026-11-13 is a Friday; 21:00 UTC = 16:00 EST -> still inside buffer -> yesterday
    assert last_complete_session_date(datetime(2026, 11, 13, 21, 0, tzinfo=timezone.utc), dcfg) == date(2026, 11, 12)
    # 22:00 UTC = 17:00 EST -> today's bar is final
    assert last_complete_session_date(datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc), dcfg) == date(2026, 11, 13)
    # In July (EDT, UTC-4) 20:31 UTC = 16:31 local -> final
    assert last_complete_session_date(datetime(2026, 7, 14, 20, 31, tzinfo=timezone.utc), dcfg) == date(2026, 7, 14)


def test_drop_incomplete_bars():
    d = SESSIONS[-3:]
    df = make_history("A", [1, 2, 3], d)
    out = drop_incomplete_bars(df, d[1].date())
    assert out["date"].max() == d[1]


def test_panels_drop_stray_holiday_bar_and_keep_gaps_nan():
    d = SESSIONS[:10]
    rows = [make_history(s, np.arange(1, 11), d) for s in ["A", "B", "C", "D"]]
    rows.append(make_history("A", [99.0], [pd.Timestamp("2026-01-10")]))  # a Saturday bar from one ticker
    long = pd.concat(rows)
    long = long[~((long["symbol"] == "B") & (long["date"] == d[4]))]  # B missing one session
    p = Panels.from_long(long)
    assert pd.Timestamp("2026-01-10") not in p.dates
    assert len(p.dates) == 10
    assert np.isnan(p.close.loc[d[4], "B"])
    assert p.symbol_frame("A").columns.tolist() == ["date", "open", "high", "low", "close", "volume"]
    assert p.as_of(d[5]).dates.max() == d[5]


def test_session_not_yet_published_is_downloaded_again(cache):
    # A run right after the close: Yahoo has bars only up to yesterday for AAA, but today for BBB.
    d = SESSIONS
    market = FakeMarket({"AAA": make_history("AAA", np.full(len(d) - 1, 10.0), d[:-1]),
                         "BBB": make_history("BBB", np.full(len(d), 10.0), d)})
    run_update(cache, market, ["AAA", "BBB"], d[-1].date())
    assert cache.read_meta()["checked"] == {"BBB": d[-1].date().isoformat()}
    # The retry later that day: Yahoo has now published AAA's bar.
    market.history["AAA"] = make_history("AAA", np.full(len(d), 10.0), d)
    market.requests.clear()
    df, rep = run_update(cache, market, ["AAA", "BBB"], d[-1].date())
    assert rep.already_current == 1  # BBB skipped, AAA downloaded again
    assert market.requests[0][0] == ("AAA",)
    assert df.groupby("symbol")["date"].max().eq(d[-1]).all()
