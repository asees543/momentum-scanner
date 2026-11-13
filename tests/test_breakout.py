"""Stage 4 breakout trigger, on the textbook setup (tests/synth.py) plus one or two new bars.

On bar 159 the setup qualifies with trigger T = 31.4 x 1.0125 = 31.7925.
For a breakout on bar 160:
  mean volume of bars 140..159 = (10 x 1.5M + 10 x 0.6M) / 20 = 1.05M, so 1.5x = 1,575,000
  ADR% as of bar 159 = 5.875
"""

import pandas as pd
import pytest

from scanner import scans
from tests.synth import Bars, panels_from, textbook, us_cfg

CFG = us_cfg()
T = 31.4 * 1.0125
AVG_VOL = 1_050_000.0


def with_bar(b: Bars, close, high, low, volume) -> Bars:
    return b.copy().append(close, high, low, volume)


def run(b: Bars, sym="X"):
    return scans.run_scans(panels_from({sym: b}), CFG)


def test_clean_breakout():
    res = run(with_bar(textbook(), close=33.0, high=33.3, low=31.9, volume=2_000_000))
    bo = res.breakouts
    assert bo.ok_count == 1 and bo.rejected_count == 0
    row = bo.table.loc["X"]
    assert row["trigger"] == pytest.approx(T)
    assert row["stop"] == 31.9  # breakout day's low
    assert row["risk_pct"] == pytest.approx(100 * (33.0 - 31.9) / 33.0)
    assert row["adr_pct"] == pytest.approx(5.875)
    assert row["vol_ratio"] == pytest.approx(2_000_000 / AVG_VOL)
    assert row["status"] == "ok"
    assert row["setup_age"] == 1
    # It closed above its trigger, so it has left the watchlist.
    assert "X" not in res.watchlist.table.index


def test_high_above_trigger_but_close_below_is_not_a_breakout():
    res = run(with_bar(textbook(), close=31.7, high=32.5, low=31.2, volume=2_000_000))
    assert res.breakouts.table.empty


def test_volume_boundary_is_inclusive():
    res = run(with_bar(textbook(), close=33.0, high=33.3, low=31.9, volume=1.5 * AVG_VOL))
    assert res.breakouts.ok_count == 1
    res = run(with_bar(textbook(), close=33.0, high=33.3, low=31.9, volume=1.5 * AVG_VOL - 1))
    assert res.breakouts.table.empty


def test_low_volume_breakout_is_ignored():
    res = run(with_bar(textbook(), close=33.0, high=33.3, low=31.9, volume=1.4 * AVG_VOL))
    assert res.breakouts.table.empty


@pytest.mark.parametrize("risk_pct, status", [(5.875 - 0.01, "ok"), (5.875 + 0.01, "too wide")])
def test_stop_wider_than_adr_is_rejected_but_listed(risk_pct, status):
    close = 33.0
    low = close * (1 - risk_pct / 100)
    res = run(with_bar(textbook(), close=close, high=33.3, low=low, volume=2_000_000))
    row = res.breakouts.table.loc["X"]
    assert row["risk_pct"] == pytest.approx(risk_pct)
    assert row["status"] == status
    assert res.breakouts.ok_count + res.breakouts.rejected_count == 1


def busy_tail(b: Bars, from_bar: int) -> Bars:
    for i in range(from_bar, 160):
        b.set_bar(i, volume=1_200_000.0)
    return b


def test_breakout_from_setup_three_sessions_back_uses_that_trigger():
    b = busy_tail(textbook(), 157)  # last qualifying setup: bar 156, trigger = 33.28 (run-up peak)
    res = run(with_bar(b, close=34.0, high=34.4, low=33.0, volume=2_500_000))
    row = res.breakouts.table.loc["X"]
    assert row["trigger"] == pytest.approx(32 * 1.04)
    assert row["setup_age"] == 4  # bar 156 -> bar 160
    assert row["status"] == "ok"


def test_setup_older_than_memory_cannot_break_out():
    b = busy_tail(textbook(), 155)  # last setup bar 154: six sessions before bar 160
    res = run(with_bar(b, close=34.0, high=34.4, low=33.0, volume=2_500_000))
    assert res.breakouts.table.empty


def test_second_day_above_trigger_is_not_reported_again():
    b = with_bar(textbook(), close=33.0, high=33.3, low=31.9, volume=2_000_000)
    b = with_bar(b, close=33.6, high=33.9, low=32.8, volume=2_000_000)
    res = run(b)
    assert res.breakouts.table.empty


def test_breakout_is_point_in_time():
    b = with_bar(textbook(), close=33.0, high=33.3, low=31.9, volume=2_000_000)
    p = panels_from({"X": b})
    assert scans.run_scans(p, CFG).breakouts.ok_count == 1
    before = scans.run_scans(p.as_of(p.dates[-2]), CFG)
    assert before.breakouts.table.empty
    assert "X" in before.watchlist.table.index  # the day before, it was on the watchlist


def test_run_scans_leaders_match_momentum_leaders():
    p = panels_from({"X": textbook()})
    a = scans.run_scans(p, CFG).leaders
    b = scans.momentum_leaders(p, CFG.liquidity, CFG.leaders)
    assert a.counts == b.counts
    # Rolling means over different-length tails differ by ~1e-15, so compare with a tolerance.
    pd.testing.assert_frame_equal(a.table, b.table)
