"""Stage 3 consolidation setup and the watchlist, on synthetic series (tests/synth.py).

Textbook setup, bar 159 (hand-checked):
  liquid      close 31.4 >= $1; 20-day avg close x volume ~ $32M >= $3M; ADR% 5.875 >= 5
  leader      3M: 31.4 / close 67 bars ago (20.0) = 1.57 >= 1.5, liquid that day
  trend       SMA50 ~ 25.9 < 31.4
  sma_order   SMA10 ~ 31.05 > SMA20 ~ 30.6
  near_high   20-day high = 32 x 1.04 = 33.28; 0.85 x 33.28 = 28.29 <= 31.4
  higher_lows min low bars 155-159 = 31.1 x 1.0125/1.025 = 30.72 > min low bars 150-154 = 30.6 x 1.025/1.05 = 29.87
  tight       3-day mean range 2.5% < ADR% = (10 x 8 + 5 x 5 + 5 x 2.5) / 20 = 5.875
  dryup       0.6M <= 0.70 x mean(11 x 1.5M + 9 x 0.6M) = 0.70 x 1.095M
  trigger     highest high of bars 150-159 = 31.4 x 1.0125 = 31.7925
"""

import numpy as np
import pandas as pd
import pytest

from scanner import scans
from scanner.config import WindowConfig
from tests.synth import Bars, frames, panels_from, textbook, us_cfg

CFG = us_cfg()
MAX_LB = 126
RULES = ["liquid", "leader", "trend", "sma_order", "near_high", "higher_lows", "tight", "dryup"]


def last_conditions(sf, sym="X"):
    return {name: bool(f[sym].iloc[-1]) for name, f in sf.conditions.items()}


def watchlist(bars_by_symbol, cfg=CFG):
    p = panels_from(bars_by_symbol)
    liq, lf, sf = frames(p, cfg)
    return scans.watchlist_from_frames(p, liq, lf, sf, cfg, MAX_LB), sf


def test_textbook_setup_qualifies_on_every_rule():
    _, _, sf = frames(panels_from({"X": textbook()}), CFG)
    assert last_conditions(sf) == {r: True for r in RULES}
    assert bool(sf.qualified["X"].iloc[-1])
    assert sf.trigger["X"].iloc[-1] == pytest.approx(31.4 * 1.0125)


def test_textbook_watchlist_row():
    b = textbook()
    wl, _ = watchlist({"X": b})
    row = wl.table.loc["X"]
    trigger = b.high[150:160].max()
    stop = b.low[159]
    assert row["trigger"] == pytest.approx(trigger)
    assert row["stop"] == pytest.approx(stop)  # latest bar's low
    assert row["risk_pct"] == pytest.approx(100 * (trigger - stop) / trigger)
    assert row["to_trigger_pct"] == pytest.approx(100 * (trigger / 31.4 - 1))
    assert row["setup_age"] == 0
    assert not row["risk_too_wide"]
    assert row["vol_ratio"] == pytest.approx(0.6 / 1.095)
    assert row["led_3m"] and row["led_1m"] and not row["led_6m"]
    assert row["ret_3m"] == pytest.approx(31.4 / 20.0 - 1)  # close 67 bars ago was 20.0
    assert row["ret_1m"] == pytest.approx(31.4 / (20 + 0.4 * 18) - 1)  # bar 137 on the run-up
    assert wl.qualified_today == 1


# --- break exactly one rule ---------------------------------------------------------------------


def only_failing(sf):
    return sorted(r for r, ok in last_conditions(sf).items() if not ok)


def test_volume_not_drying_up():
    b = textbook().set_bar(159, volume=0.8 * 1_095_000)
    _, _, sf = frames(panels_from({"X": b}), CFG)
    assert only_failing(sf) == ["dryup"]


def test_volume_exactly_70_pct_passes():
    b = textbook()
    b.volume[:159] = 1_000_000.0  # prior 20-session mean is exactly 1.0M
    b.set_bar(159, volume=700_000.0)
    _, _, sf = frames(panels_from({"X": b}), CFG)
    assert last_conditions(sf)["dryup"]
    b.set_bar(159, volume=700_001.0)
    _, _, sf = frames(panels_from({"X": b}), CFG)
    assert only_failing(sf) == ["dryup"]


def test_range_not_tight():
    b = textbook()
    for i in range(155, 160):
        b.set_bar(i, range_pct=7.5)  # ADR becomes (80 + 25 + 37.5) / 20 = 7.125 < every 3-5 day mean
    _, _, sf = frames(panels_from({"X": b}), CFG)
    assert only_failing(sf) == ["tight"]


def test_tight_needs_only_one_of_3_4_5_days():
    b = textbook()
    # Last 3 days tight, but days 155-156 wide: the 3-day mean passes even though the 5-day mean fails.
    b.set_bar(155, range_pct=14.0).set_bar(156, range_pct=14.0)
    _, _, sf = frames(panels_from({"X": b}), CFG)
    assert last_conditions(sf)["tight"]


def test_lower_low_breaks_higher_lows():
    b = textbook().set_bar(157, low=29.5)  # below the 29.87 low of bars 150-154
    _, _, sf = frames(panels_from({"X": b}), CFG)
    assert only_failing(sf) == ["higher_lows"]


def test_near_high_boundary_is_inclusive():
    b = textbook()
    h20 = b.high[140:160].max()  # 33.28, the run-up peak
    b.set_bar(159, close=(1 - 15 / 100) * h20)
    _, _, sf = frames(panels_from({"X": b}), CFG)
    assert last_conditions(sf)["near_high"]
    b.set_bar(159, close=0.999 * (1 - 15 / 100) * h20)
    _, _, sf = frames(panels_from({"X": b}), CFG)
    assert not last_conditions(sf)["near_high"]


def test_trend_and_sma_order_on_simple_series():
    n = 80
    up = Bars.from_close(np.linspace(10, 30, n), 8.0, 1e6)
    down = Bars.from_close(np.linspace(30, 10, n), 8.0, 1e6)
    _, _, sf = frames(panels_from({"UP": up, "DOWN": down}), CFG)
    assert bool(sf.conditions["trend"]["UP"].iloc[-1]) and bool(sf.conditions["sma_order"]["UP"].iloc[-1])
    assert not bool(sf.conditions["trend"]["DOWN"].iloc[-1])
    assert not bool(sf.conditions["sma_order"]["DOWN"].iloc[-1])


def test_leader_must_be_recent():
    # With only the 1M window, the last 1M-leader day is bar 152 (30.8 / 24.4 = 1.26); bar 153 is
    # 30.9 / 24.8 = 1.246. That is 7 sessions before bar 159.
    one_m = {"1m": WindowConfig(lookback=22, min_multiple=1.25)}
    cfg20 = us_cfg(leaders={"windows": one_m, "sort_by": "1m"})
    cfg5 = us_cfg(leaders={"windows": one_m, "sort_by": "1m"}, setup={"leader_lookback_sessions": 5})
    p = panels_from({"X": textbook()})
    _, lf, sf20 = frames(p, cfg20)
    assert bool(lf.passed["1m"]["X"].iloc[152]) and not lf.passed["1m"]["X"].iloc[153:].any()
    assert bool(sf20.qualified["X"].iloc[-1])
    _, _, sf5 = frames(p, cfg5)
    assert only_failing(sf5) == ["leader"]


def test_never_a_leader_never_qualifies():
    flat = Bars.from_close(np.full(160, 20.0), 8.0, 1e6)
    wl, sf = watchlist({"X": flat})
    assert not bool(sf.conditions["leader"]["X"].iloc[-1])
    assert wl.table.empty


# --- watchlist memory -------------------------------------------------------------------------


def busy_tail(b, from_bar):
    """Make bars from_bar..159 fail the dry-up rule (volume 1.2M) while staying under the trigger."""
    for i in range(from_bar, 160):
        b.set_bar(i, volume=1_200_000.0)
    return b


def test_setup_from_3_sessions_ago_stays_on_watchlist_with_its_trigger():
    b = busy_tail(textbook(), 157)
    wl, sf = watchlist({"X": b})
    q = sf.qualified["X"]
    assert bool(q.iloc[156]) and not q.iloc[157:].any()
    row = wl.table.loc["X"]
    assert row["setup_age"] == 3
    assert row["trigger"] == pytest.approx(sf.trigger["X"].iloc[156])  # 10-bar high as of bar 156
    assert row["stop"] == pytest.approx(b.low[159])
    assert wl.qualified_today == 0


def test_setup_older_than_memory_drops_off():
    b = busy_tail(textbook(), 155)
    wl, sf = watchlist({"X": b})
    q = sf.qualified["X"]
    assert bool(q.iloc[154]) and not q.iloc[155:].any()  # last setup is 5 sessions before bar 159
    assert "X" not in wl.table.index


def test_name_that_already_closed_above_trigger_is_not_on_watchlist():
    b = busy_tail(textbook(), 157)
    b.set_bar(158, close=34.0)  # above the bar-156 trigger (33.28)
    wl, sf = watchlist({"X": b})
    assert bool(sf.qualified["X"].iloc[156])
    assert "X" not in wl.table.index


def test_wide_risk_is_flagged():
    b = textbook()
    b.set_bar(159, low=27.0)  # bar 159 dips to 27: today's setup fails, bar 158's setup still counts
    wl, sf = watchlist({"X": b})
    assert bool(sf.qualified["X"].iloc[158]) and not bool(sf.qualified["X"].iloc[159])
    row = wl.table.loc["X"]
    assert row["setup_age"] == 1
    assert row["stop"] == 27.0
    assert row["risk_pct"] > row["adr_pct"]  # ~15% risk vs ~6.6% ADR
    assert row["risk_too_wide"]


def test_watchlist_is_point_in_time():
    b = textbook()
    p = panels_from({"X": b}).as_of(panels_from({"X": b}).dates[-2])  # stop at bar 158
    liq, lf, sf = frames(p, CFG)
    wl = scans.watchlist_from_frames(p, liq, lf, sf, CFG, MAX_LB)
    assert wl.as_of == p.dates[-1]
    assert wl.table.loc["X", "stop"] == pytest.approx(b.low[158])
    assert wl.table.loc["X", "trigger"] == pytest.approx(b.high[149:159].max())  # bar 159 unseen


def test_broken_base_drops_off_even_within_memory():
    b = busy_tail(textbook(), 157)  # setup on bar 156
    b.set_bar(159, close=27.0)  # now more than 15% below the 33.28 high
    wl, sf = watchlist({"X": b})
    assert bool(sf.qualified["X"].iloc[156])
    assert not bool(sf.conditions["near_high"]["X"].iloc[159])
    assert "X" not in wl.table.index


def collapsed_adr() -> "Bars":
    """Textbook, but the 10-bar base trades in a 1.5% range: ADR on bar 159 falls to
    (10 x 8 + 10 x 1.5) / 20 = 4.75 < 5, like a stock pinned to an acquisition price."""
    b = textbook()
    for i in range(150, 160):
        b.set_bar(i, range_pct=1.5)
    return b


def test_stage1_must_pass_on_the_day_itself():
    wl, sf = watchlist({"X": collapsed_adr()})
    assert only_failing(sf) == ["liquid"]  # every other rule, including "leader", still holds
    assert bool(sf.conditions["leader"]["X"].iloc[-1])
    assert "X" not in wl.table.index


def test_stage1_today_can_be_switched_off():
    wl, sf = watchlist({"X": collapsed_adr()}, us_cfg(setup={"require_liquid_today": False}))
    assert last_conditions(sf)["liquid"]
    assert "X" in wl.table.index
