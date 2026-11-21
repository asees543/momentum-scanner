"""Trigger Bar scan: a tight bar on a key moving average in an uptrend (Blueprint Trading Course, p.36-37)."""

import dataclasses
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scanner import indicators as ind
from scanner import scans
from scanner.config import load_config
from scanner.dashboard import build_site, snapshot, write_snapshot
from scanner.data.us_loader import chart_url
from scanner.notify import TB_DIVIDER, format_alerts, trigger_bar_entry, trigger_bar_flags
from scanner.state import MarketState, diff_results
from tests.synth import Bars, panels_from, us_cfg

UNIVERSE = pd.DataFrame({"symbol": ["X"], "source_symbol": ["X"], "name": ["X Corp"], "exchange": ["NASDAQ"]})
N = 260


def uptrend(close_last=None, range_last=None, volume_last=None, n=N, slope=0.2, start=50.0, volume=1_000_000.0,
            range_pct=3.0) -> Bars:
    """A steady uptrend (close = start + slope x bar, 3% daily range, constant volume) whose last bar can be
    changed. The 10-day EMA trails the close by about 1.8 points (slope x 9), the 21-day by about 4.

    On the last bar the high/low straddle the close, so range% = range_last exactly."""
    b = Bars.from_close(start + slope * np.arange(n), range_pct, volume)
    if close_last is not None or range_last is not None or volume_last is not None:
        b.set_bar(n - 1, close=close_last, range_pct=range_last, volume=volume_last)
    return b


def on_ema(range_pct: float, close_pos: float = 0.5, volume: float | None = None, previous: str = "",
           **kw) -> Bars:
    """The uptrend with a last bar sitting on its 10-day EMA: `range_pct` high-to-low exactly, and the close at
    `close_pos` of the bar (0 = at the low, 1 = at the high). Solves low = ema / (1 + pos x k), high = low x (1 + k)
    with k = range_pct / 100, so range% and the close position are exact."""
    ema = float(ind.ema(pd.Series(uptrend(**kw).close), 10).iloc[-1])
    k = range_pct / 100
    low = ema / (1 + close_pos * k)
    b = uptrend(close_last=ema, volume_last=volume, **kw)
    b.low[-1], b.high[-1] = low, low * (1 + k)
    if previous == "wider":  # yesterday's range contains today's: an inside bar
        b.high[-2], b.low[-2] = b.high[-1] + 2, b.low[-1] - 2
    elif previous == "lower":  # yesterday's high is under today's: not an inside bar
        b.high[-2], b.low[-2] = b.high[-1] - 0.05, b.low[-1] - 0.05
    return b


def run(bars: dict[str, Bars], cfg=None) -> scans.TriggerBarResult:
    cfg = cfg or us_cfg()
    p = panels_from(bars)
    return scans.run_scans(p, cfg).trigger_bars


def row(bars: Bars, cfg=None) -> pd.Series | None:
    t = run({"X": bars}, cfg).table
    return t.loc["X"] if "X" in t.index else None


# --- the rules ----------------------------------------------------------------------------------------


def test_tight_bar_on_the_10_ema_qualifies_and_the_numbers_are_hand_checked():
    b = uptrend()
    c = pd.Series(b.close)
    # last close 50 + 0.2 x 259 = 101.8; make the bar tight (1%, ADR is 3%) and sitting on the 10-day EMA
    ema10 = ind.ema(c, 10).iloc[-1]
    b = uptrend(close_last=float(ema10), range_last=1.0)
    r = row(b)
    assert r is not None and r["ma"] == "10 EMA"
    c = pd.Series(b.close)
    hi, lo = b.high[-1], b.low[-1]
    assert r["buy_above"] == pytest.approx(hi) and r["stop"] == pytest.approx(lo - 0.01)
    assert r["risk_pct"] == pytest.approx(100 * (hi - (lo - 0.01)) / hi)
    assert r["range_pct"] == pytest.approx(1.0) and r["adr_pct"] == pytest.approx((19 * 3.0 + 1.0) / 20, abs=0.05)
    assert r["range_vs_adr"] == pytest.approx(r["range_pct"] / r["adr_pct"])
    assert r["risk_vs_adr"] == pytest.approx(r["risk_pct"] / r["adr_pct"])
    assert r["ma_dist_pct"] == pytest.approx(0.0)           # the EMA is inside the bar
    assert r["close"] == pytest.approx(ema10)


def test_bonuses_are_counted_one_by_one():
    cfg = us_cfg()
    # volume 40% of average, close at 60% of the bar, 1% range against an ADR of about 2.9%
    r = row(on_ema(1.0, close_pos=0.6, volume=400_000), cfg)
    assert r["low_volume"] and r["vol_ratio"] == pytest.approx(0.4)
    assert r["strong_close"] and r["half_adr"]               # 0.6 >= 0.5; 1% <= half of 2.9%
    assert r["tightest"]                                     # tighter than each of the previous 10 bars (3%)
    # the close in the lower part of the bar loses only the strong-close sign
    r1 = row(on_ema(1.0, close_pos=0.3, volume=400_000), cfg)
    assert not r1["strong_close"] and r1["score"] == r["score"] - 1
    # a heavy-volume day loses only the low-volume sign
    r2 = row(on_ema(1.0, close_pos=0.6, volume=3_000_000), cfg)
    assert not r2["low_volume"] and r2["score"] == r["score"] - 1
    # a 1.8% bar (0.62 ADR: allowed, but over half an ADR) loses the half-ADR sign, and being wider than the day before
    # it also stops being an inside bar; the other four signs hold
    assert r["inside"] and r["score"] == 6
    r3 = row(on_ema(1.8, close_pos=0.6, volume=400_000), cfg)
    assert not r3["half_adr"] and not r3["inside"] and r3["tightest"] and r3["score"] == r["score"] - 2


def test_inside_bar_needs_a_higher_high_and_lower_low_the_day_before():
    r = row(on_ema(1.0, close_pos=0.6, volume=400_000, previous="wider"))
    assert r["inside"]
    # the same bar after a bar whose high is lower is not inside
    assert not row(on_ema(1.0, close_pos=0.6, volume=400_000, previous="lower"))["inside"]


def quiet_stock() -> Bars:
    """ADR 0.9%: a steady uptrend with 0.9% daily ranges, last bar tight (0.3%) on its 10-day EMA."""
    ema = float(ind.ema(pd.Series(uptrend(range_pct=0.9).close), 10).iloc[-1])
    return uptrend(range_pct=0.9, close_last=ema, range_last=0.3)


def thin_stock() -> Bars:
    """1,000 shares a day at about $100: $100,000 a day against the $3,000,000 floor."""
    ema = float(ind.ema(pd.Series(uptrend(volume=1_000.0).close), 10).iloc[-1])
    return uptrend(volume=1_000.0, close_last=ema, range_last=1.0, volume_last=1_000.0)


def test_each_hard_rule_can_fail_alone():
    assert row(on_ema(1.0)) is not None
    # too wide: a 3% bar against a 3% ADR (cap is 2/3 of ADR)
    assert row(on_ema(3.0)) is None
    # far from every key average: the whole bar 15% above the 10-day EMA (still above all the others)
    far = on_ema(1.0)
    far.close[-1], far.high[-1], far.low[-1] = far.close[-1] * 1.15, far.high[-1] * 1.15, far.low[-1] * 1.15
    assert row(far) is None
    assert row(quiet_stock()) is None          # volatility floor: ADR 0.9% < 1.5%
    assert row(thin_stock()) is None           # liquidity floor


def test_uptrend_is_required():
    # a downtrend: close below its 50 and 200-day SMAs even though the bar is tight and on the 10-day EMA
    down = Bars.from_close(250.0 - 0.5 * np.arange(N), 3.0, 1_000_000.0)
    ema10 = float(ind.ema(pd.Series(down.close), 10).iloc[-1])
    down.set_bar(N - 1, close=ema10, range_pct=1.0)
    assert row(down) is None
    # above the 200-day but below the 50-day SMA (a deep pullback inside a long uptrend) also fails
    c = np.r_[np.linspace(50, 150, 220), np.linspace(150, 120, 40)]
    b = Bars.from_close(c, 3.0, 1_000_000.0)
    s = pd.Series(b.close)
    assert s.iloc[-1] > ind.sma(s, 200).iloc[-1] and s.iloc[-1] < ind.sma(s, 50).iloc[-1]
    b.set_bar(N - 1, close=float(ind.ema(s, 10).iloc[-1]), range_pct=1.0)
    assert row(b) is None


def test_short_history_and_missing_bars_do_not_qualify():
    assert row(uptrend(n=150)) is None                       # no 200-day average yet
    b = uptrend()
    b.close[-1] = np.nan
    assert row(b) is None


def test_stop_must_not_be_wider_than_one_adr():
    cfg = us_cfg()
    ema10 = float(ind.ema(pd.Series(uptrend().close), 10).iloc[-1])
    # a 1% bar passes; with a stop buffer of 5 points (about 5% below the low) the stop is wider than the 2.9% ADR
    wide = cfg.model_copy(update={"trigger_bar": cfg.trigger_bar.model_copy(update={"stop_buffer": 5.0})})
    assert row(uptrend(close_last=ema10, range_last=1.0), cfg) is not None
    assert row(uptrend(close_last=ema10, range_last=1.0), wide) is None


def test_names_on_our_lists_come_first_then_score():
    cfg = us_cfg()
    ema10 = float(ind.ema(pd.Series(uptrend().close), 10).iloc[-1])
    tight = uptrend(close_last=ema10, range_last=1.0, volume_last=400_000)       # high score
    plain = uptrend(close_last=ema10, range_last=1.9, volume_last=3_000_000)     # lower score
    p = panels_from({"A": plain, "B": tight})
    res = scans.run_scans(p, cfg)
    assert list(res.trigger_bars.table.index) == ["B", "A"]                       # by score
    # mark A as a watchlist name: it moves first
    wl = pd.DataFrame({"close": [1.0], "trigger": [2.0], "stop": [1.0], "flag_day_move": [False]}, index=["A"])
    lead = scans.LeadersResult(res.leaders.as_of, "fixed", res.leaders.table.iloc[0:0], {}, 2, 0, 2, ["1m", "3m", "6m"])
    tb = scans.trigger_bars(p, cfg, lead, scans.WatchlistResult(res.leaders.as_of, wl, 1),
                            scans.BreakoutResult(res.leaders.as_of, wl.iloc[0:0], 0, 0))
    assert list(tb.table.index) == ["A", "B"] and tb.table.loc["A", "ours"] == "watchlist" and tb.table.loc["B", "ours"] == ""


def test_both_markets_configure_the_scan():
    for market, tick in (("us", 0.01), ("india", 0.05)):
        t = load_config(market).trigger_bar
        assert (t.ema_fast, t.ema_mid, t.sma_mid, t.sma_long, t.volume_window) == (10, 21, 50, 200, 30)
        assert t.max_range_vs_adr == pytest.approx(0.67) and t.half_adr_mult == 0.5 and t.stop_buffer == tick
        assert t.message_max <= t.max_listed


# --- message and dashboard ------------------------------------------------------------------------------


def results_with_a_bar(**kw):
    cfg = us_cfg(trigger_bar={"message_max": 1, **kw})
    ema10 = float(ind.ema(pd.Series(uptrend().close), 10).iloc[-1])
    p = panels_from({"X": uptrend(close_last=ema10, range_last=1.0, volume_last=400_000),
                     "Y": uptrend(close_last=ema10, range_last=1.8, volume_last=3_000_000)})
    return cfg, scans.run_scans(p, cfg)


def alert_text(res, cfg):
    prev = MarketState(market="us", as_of=date(2026, 1, 1), active_count=1, watchlist={})
    return "\n".join(format_alerts(diff_results(prev, res, "us", cfg.alerts), res, cfg, UNIVERSE, chart_url))


def test_alert_message_lists_the_plan_for_each_bar():
    cfg, res = results_with_a_bar()
    r = res.trigger_bars.table.loc["X"]
    text = alert_text(res, cfg)
    assert "🔔 <b>Trigger bars today</b> · 2 names, 1 shown" in text
    assert "<b>X</b> · X Corp\n" in text
    assert f"/6 · on 10 EMA · buy above {r['buy_above']:.2f} · stop {r['stop']:.2f} (" in text
    assert "low volume 0.4×" in text and "range under half an ADR" in text
    assert "…and 1 more on the dashboard." in text                         # message_max = 1; Y is not listed
    assert "<b>Y</b>" not in text
    # and with nothing, it says so
    empty = dataclasses.replace(res, trigger_bars=dataclasses.replace(res.trigger_bars, table=res.trigger_bars.table.iloc[0:0]))
    assert "Trigger bars today</b> · 0 names, 0 shown" in alert_text(empty, cfg) and "None today." in alert_text(empty, cfg)
    assert "Trigger bars" not in alert_text(dataclasses.replace(res, trigger_bars=None), cfg)


def test_entry_marks_wide_stops_and_our_lists():
    cfg, res = results_with_a_bar()
    r = res.trigger_bars.table.loc["X"].copy()
    r["risk_pct"], r["ours"] = 6.5, "watchlist"
    text = trigger_bar_entry("X", r, ["1m", "3m", "6m"], "USD", "https://x", "X Corp")
    assert "6.5%" in text and " · wide)" in text and "on our watchlist" in text
    only_inside = r.copy()
    for col in ("inside", "tightest", "low_volume", "strong_close", "half_adr", "confluence"):
        only_inside[col] = col == "inside"
    assert trigger_bar_flags(only_inside) == ["inside bar"]


def test_dashboard_panel_and_old_snapshots(tmp_path):
    cfg, res = results_with_a_bar(max_listed=1)
    snap = snapshot(res, cfg, UNIVERSE, chart_url)
    tb = snap["trigger_bars"]
    assert tb["count"] == 2 and len(tb["items"]) == 1                      # max_listed = 1
    item = tb["items"][0]
    assert item["symbol"] == "X" and item["ma"] == "10 EMA" and "low volume 0.4×" in item["signs"]
    data, reports, out = tmp_path / "data", tmp_path / "reports", tmp_path / "site"
    reports.mkdir()
    write_snapshot(snap, data)
    build_site(out, data, reports)
    page = (out / "index.html").read_text()
    assert "<h2>Trigger bars (2)</h2>" in page and "The best 1 of 2 are listed." in page and ">X</a>" in page
    assert "Buy above" in page and "hover a score" in page
    old = json.loads((data / "us.json").read_text())
    del old["trigger_bars"]
    (data / "us.json").write_text(json.dumps(old))
    build_site(out, data, reports)
    assert "Trigger bars (" not in (out / "index.html").read_text()


# --- the line where names on our lists end ------------------------------------------------------------------


def with_ours(res, ours: list[str]):
    """Both names X (better) and Y, with X and Y marked as on our lists (or not) in that order."""
    table = res.trigger_bars.table.assign(ours=ours)
    return dataclasses.replace(res, trigger_bars=dataclasses.replace(res.trigger_bars, table=table))


def test_message_draws_a_line_only_where_our_names_end():
    cfg, res = results_with_a_bar(message_max=10)
    assert list(res.trigger_bars.table.index) == ["X", "Y"]
    mixed = alert_text(with_ours(res, ["watchlist", ""]), cfg)
    assert mixed.count(TB_DIVIDER) == 1 and mixed.index("<b>X</b>") < mixed.index(TB_DIVIDER) < mixed.index("<b>Y</b>")
    assert alert_text(with_ours(res, ["watchlist", "leader"]), cfg).count(TB_DIVIDER) == 0   # all on our lists
    assert alert_text(with_ours(res, ["", ""]), cfg).count(TB_DIVIDER) == 0                  # none on our lists


def test_dashboard_draws_the_line_and_drops_it_when_re_sorted(tmp_path):
    cfg, res = results_with_a_bar()
    reports = tmp_path / "reports"
    reports.mkdir()

    def page(ours):
        data = tmp_path / f"data_{'_'.join(o or 'none' for o in ours)}"
        write_snapshot(snapshot(with_ours(res, ours), cfg, UNIVERSE, chart_url), data)
        out = tmp_path / f"site_{data.name}"
        build_site(out, data, reports)
        return (out / "index.html").read_text()

    mixed = page(["watchlist", ""])
    assert mixed.count('<tr class="divider">') == 1
    assert mixed.index(">X</a>") < mixed.index('<tr class="divider">') < mixed.index(">Y</a>")
    assert page(["watchlist", "leader"]).count('<tr class="divider">') == 0
    assert page(["", ""]).count('<tr class="divider">') == 0
    assert "tr.divider').forEach(r => r.remove())" in mixed             # re-sorting removes it
