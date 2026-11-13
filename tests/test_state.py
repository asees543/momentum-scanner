from datetime import date, datetime, timezone

import pandas as pd
import pytest

from scanner.config import AlertsConfig
from scanner.scans import BreakoutResult, LeadersResult, ScanResults, WatchlistResult
from scanner.state import MarketState, diff_results, state_path

ACFG = AlertsConfig(min_active_ratio=0.9, baseline_first_run=True)
NOW = datetime(2026, 11, 13, 22, 5, tzinfo=timezone.utc)


def results(as_of="2026-11-12", watch=("A", "B"), breakouts=(), active=5000) -> ScanResults:
    ts = pd.Timestamp(as_of)
    leaders = LeadersResult(as_of=ts, mode="fixed", table=pd.DataFrame(), counts={}, active_count=active,
                            stale_count=0, liquid_count=0, windows=[])
    wl = pd.DataFrame({"trigger": [10.0 + i for i in range(len(watch))]},
                      index=pd.Index(list(watch), name="symbol"))
    bo = pd.DataFrame({"status": [st for _, st in breakouts]},
                      index=pd.Index([s for s, _ in breakouts], name="symbol"))
    ok = sum(st == "ok" for _, st in breakouts)
    return ScanResults(leaders, WatchlistResult(ts, wl, len(watch)), BreakoutResult(ts, bo, ok, len(breakouts) - ok))


def run(prev, res, acfg=ACFG):
    return diff_results(prev, res, "us", acfg, now=NOW)


def test_first_run_records_baseline_and_alerts_only_breakouts():
    d = run(None, results(watch=("A", "B"), breakouts=(("C", "ok"), ("D", "too wide"))))
    assert d.status == "send" and d.first_run
    assert d.new_watchlist == []  # baseline, no flood
    assert d.breakouts == ["C"]
    assert d.rejected_breakouts == 1
    assert d.watchlist_size == 2
    assert set(d.next_state.watchlist) == {"A", "B"}
    assert d.next_state.watchlist["A"] == {"trigger": 10.0, "first_seen": "2026-11-12"}


def test_first_run_without_baseline_lists_everything():
    d = run(None, results(watch=("A", "B")), AlertsConfig(baseline_first_run=False))
    assert d.new_watchlist == ["A", "B"]


def test_new_names_only_and_order_kept():
    prev = run(None, results("2026-11-11", watch=("A", "B"))).next_state
    d = run(prev, results("2026-11-12", watch=("C", "A", "D")))
    assert d.status == "send" and not d.first_run
    assert d.new_watchlist == ["C", "D"]  # watchlist-table order (closest to trigger first)
    assert d.next_state.watchlist["A"]["first_seen"] == "2026-11-11"
    assert d.next_state.watchlist["C"]["first_seen"] == "2026-11-12"
    assert "B" not in d.next_state.watchlist


def test_name_that_returns_is_new_again():
    s1 = run(None, results("2026-11-10", watch=("A",))).next_state
    s2 = run(s1, results("2026-11-11", watch=())).next_state
    d = run(s2, results("2026-11-12", watch=("A",)))
    assert d.new_watchlist == ["A"]


def test_all_ok_breakouts_are_new_when_date_moves_on():
    prev = run(None, results("2026-11-11", breakouts=(("X", "ok"),))).next_state
    d = run(prev, results("2026-11-12", breakouts=(("X", "ok"), ("Y", "ok"))))
    assert d.breakouts == ["X", "Y"]


def test_same_day_is_skipped():
    prev = run(None, results("2026-11-12")).next_state
    d = run(prev, results("2026-11-12", watch=("Z",)))
    assert d.status == "skip_same_day"
    assert d.next_state is None


def test_older_date_is_refused():
    prev = run(None, results("2026-11-12")).next_state
    d = run(prev, results("2026-11-11"))
    assert d.status == "refuse_older"


def test_incomplete_data_is_unhealthy():
    prev = run(None, results("2026-11-11", active=5000)).next_state
    assert run(prev, results("2026-11-12", active=4500)).status == "send"  # exactly 90%
    d = run(prev, results("2026-11-12", active=4499))
    assert d.status == "unhealthy"
    assert "4,499" in d.reason and d.next_state is None


def test_save_load_roundtrip(tmp_path):
    st = run(None, results(watch=("A", "B"), breakouts=(("C", "ok"),))).next_state
    path = state_path("us", tmp_path)
    st.save(path)
    assert not list(tmp_path.glob("*.tmp"))
    back = MarketState.load(path)
    assert back == st
    assert back.as_of == date(2026, 11, 12)
    assert MarketState.load(tmp_path / "missing.json") is None
