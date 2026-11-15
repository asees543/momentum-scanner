"""Backtest mechanics on hand-built price paths (one symbol, explicit OHLC)."""

import numpy as np
import pandas as pd
import pytest

from scanner import backtest as bt
from scanner.config import load_config
from scanner.data.base import Panels
from tests.synth import panels_from, textbook, us_cfg

N = 30
DATES = pd.bdate_range("2026-01-05", periods=N)


def panel(opens, highs, lows, closes, sym="X") -> Panels:
    f = lambda a: pd.DataFrame({sym: np.asarray(a, dtype=float)}, index=DATES[: len(a)])
    return Panels(open=f(opens), high=f(highs), low=f(lows), close=f(closes), volume=f(np.full(len(closes), 1e6)))


def flat_then(path_from_11):
    """Closes 100 for rows 0..10, then the given closes; opens = previous close; high/low +-1%."""
    closes = np.r_[np.full(11, 100.0), path_from_11]
    opens = np.r_[100.0, closes[:-1]]
    return opens, closes * 1.01, closes * 0.99, closes


def trade(model="A", entry_row=10, entry=100.0, stop=95.0, n=N):
    return pd.DataFrame([{
        "model": model, "symbol": "X", "col": 0, "signal_date": DATES[entry_row - (model == "B")],
        "entry_row": entry_row, "entry_date": DATES[entry_row], "entry": entry, "stop": stop,
        "stop_from": entry_row + 1 if model == "A" else entry_row, "group": "ok",
        "risk_pct": 100 * (entry - stop) / entry, "vol_ratio": 2.0, "flag_day_move": False,
    }])


RULE = bt.ExitRule(fraction=1 / 3, after_days=3, trail_sma=10)


def test_stop_hit_intraday_exits_at_stop():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    lo[12] = 94.0  # trades through the 95 stop during day 12
    res = bt.simulate(trade(), panel(o, h, lo, c), RULE, cost_pct=0).iloc[0]
    assert res["ret"] == pytest.approx(-0.05) and res["r_multiple"] == pytest.approx(-1.0)
    assert res["exit_row"] == 12 and not res["open"]


def test_gap_below_stop_exits_at_open():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    o[12], lo[12] = 90.0, 89.0
    res = bt.simulate(trade(), panel(o, h, lo, c), RULE, cost_pct=0).iloc[0]
    assert res["ret"] == pytest.approx(-0.10) and res["r_multiple"] == pytest.approx(-2.0)


def test_partial_then_trailing_exit_hand_computed():
    # rows 11..15: 102, 104, 106, 107, 98 -> partial 1/3 at row 13 (entry 10 + 3) close 106;
    # row 15: SMA10 = (5 x 100 + 102 + 104 + 106 + 107 + 98) / 10 = 101.7 > 98 -> the rest exits at 98
    path = np.r_[102, 104, 106, 107, 98, np.full(N - 16, 98.0)]
    o, h, lo, c = flat_then(path)
    res = bt.simulate(trade(), panel(o, h, lo, c), RULE, cost_pct=0).iloc[0]
    assert res["ret"] == pytest.approx((106 / 3 + 98 * 2 / 3) / 100 - 1)
    assert res["exit_row"] == 15 and res["days"] == 5


def test_costs_on_both_sides():
    path = np.r_[102, 104, 106, 107, 98, np.full(N - 16, 98.0)]
    o, h, lo, c = flat_then(path)
    res = bt.simulate(trade(), panel(o, h, lo, c), RULE, cost_pct=0.15).iloc[0]
    gross = 106 / 3 + 98 * 2 / 3
    assert res["ret"] == pytest.approx(gross * (1 - 0.0015) / (100 * 1.0015) - 1)


def test_model_a_stop_not_checked_on_entry_day_but_model_b_is():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    lo[10] = 94.0  # entry day dips below the stop
    a = bt.simulate(trade("A"), panel(o, h, lo, c), RULE, 0).iloc[0]
    b = bt.simulate(trade("B"), panel(o, h, lo, c), RULE, 0).iloc[0]
    assert a["exit_row"] != 10  # A: the low may have come before the buy-stop filled
    assert b["exit_row"] == 10 and b["ret"] == pytest.approx(-0.05)  # B: bought at the open, then stopped


def test_still_open_at_end_is_valued_at_last_close():
    o, h, lo, c = flat_then(np.linspace(101, 130, N - 11))  # steady rise, never below the 10-SMA
    res = bt.simulate(trade(), panel(o, h, lo, c), RULE, cost_pct=0).iloc[0]
    assert res["open"] and res["exit_row"] == N - 1
    partial = c[13] / 3
    assert res["ret"] == pytest.approx((partial + c[-1] * 2 / 3) / 100 - 1)


def test_build_trades_models():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    o[11] = 94.0  # the day after the second signal opens below its stop
    p = panel(o, h, lo, c)
    ev = pd.DataFrame([
        {"date": DATES[5], "symbol": "X", "row": 5, "trigger": 99.5, "open": 99, "high": 101, "low": 98, "close": 100.5,
         "entry_a": 99.5, "stop": 98.0, "prev_low": 97.0, "status": "ok", "vol_ratio": 2.0, "adr_prev": 5, "setup_age": 1, "flag_day_move": False},
        {"date": DATES[10], "symbol": "X", "row": 10, "trigger": 99.0, "open": 100, "high": 101, "low": 96, "close": 98,
         "entry_a": 100.0, "stop": 96.0, "prev_low": 97.0, "status": "failed", "vol_ratio": 0.8, "adr_prev": 5, "setup_age": 2, "flag_day_move": False},
    ])
    a = bt.build_trades(ev, p, "A")
    assert a["entry"].tolist() == [99.5, 100.0]  # gap above the trigger fills at the open
    assert a["stop_from"].tolist() == [6, 11] and a["group"].tolist() == ["ok", "failed"]
    # A sizes on the previous day's low (known when the buy-stop is placed), not the breakout day's low
    assert a["risk_pct"].tolist() == pytest.approx([100 * (99.5 - 97.0) / 99.5, 100 * (100.0 - 97.0) / 100.0])
    b = bt.build_trades(ev, p, "B")
    assert len(b) == 1 and b.iloc[0]["entry"] == o[6] and b.iloc[0]["entry_row"] == 6  # failed one never in B
    assert b.iloc[0]["risk_pct"] == pytest.approx(100 * (o[6] - 98.0) / o[6])  # B: breakout-day low, known by then


def test_model_b_skips_a_gap_below_the_stop():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    o[11] = 95.0
    ev = pd.DataFrame([{"date": DATES[10], "symbol": "X", "row": 10, "trigger": 99.0, "open": 100, "high": 101,
                        "low": 96.0, "close": 100, "entry_a": 100.0, "stop": 96.0, "prev_low": 97.0, "status": "ok", "vol_ratio": 2.0,
                        "adr_prev": 5, "setup_age": 1, "flag_day_move": False}])
    assert bt.build_trades(ev, panel(o, h, lo, c), "B").empty


def test_forward_returns_vs_benchmark():
    o, h, lo, c = flat_then(np.linspace(101, 130, N - 11))
    p = panel(o, h, lo, c)
    bench = pd.DataFrame({"open": np.full(N, 50.0), "close": np.linspace(50, 53, N)}, index=DATES)
    f = bt.forward_returns(trade(), p, bench, [5]).iloc[0]
    assert f["fwd_5"] == pytest.approx(c[15] / 100 - 1)
    assert f["bench_5"] == pytest.approx(bench["close"].iloc[15] / 50 - 1)
    assert f["excess_5"] == pytest.approx(f["fwd_5"] - f["bench_5"])
    late = bt.forward_returns(trade(entry_row=N - 2), p, bench, [5]).iloc[0]
    assert np.isnan(late["fwd_5"])  # not enough data after entry


def test_trade_stats_and_curve_stats():
    t = pd.DataFrame({"ret": [0.10, -0.05, 0.20, -0.05], "r_multiple": [2, -1, 4, -1], "days": [5, 2, 9, 1],
                      "open": [False] * 4})
    s = bt.trade_stats(t)
    assert s["win_rate"] == 0.5 and s["mean_ret"] == pytest.approx(0.05)
    assert s["profit_factor"] == pytest.approx(0.30 / 0.10) and s["mean_r"] == 1.0
    curve = pd.Series([100, 120, 90, 110.0], index=pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01", "2026-01-01"]))
    cs = bt.curve_stats(curve)
    assert cs["max_drawdown"] == pytest.approx(90 / 120 - 1)
    assert cs["cagr"] == pytest.approx((110 / 100) ** (1 / ((curve.index[-1] - curve.index[0]).days / 365.25)) - 1)


def test_equity_curve_sizes_by_risk():
    # one trade: risk 5% of entry, +10% return; risking 0.5% of equity -> position 10% of equity -> +1%
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    p = panel(o, h, lo, c)
    sim = trade().assign(ret=0.10, r_multiple=2.0, exit_row=20, days=10, open=False)
    curve, stats = bt.equity_curve(sim, p, load_config("us").backtest)
    assert curve.iloc[-1] == pytest.approx(101.0) and stats["taken"] == 1


def test_events_on_textbook_breakout_match_live_scan():
    b = textbook().append(33.0, 33.3, 31.9, 2_000_000)  # bar 160: confirmed breakout
    p = panels_from({"X": b})
    cfg = us_cfg()
    hf = bt.history_frames(p, cfg, warmup=1)  # the textbook series is shorter than the live warm-up
    ev = bt.trigger_events(hf, cfg)
    row = ev[ev["date"] == p.dates[160]].iloc[0]
    assert row["status"] == "ok" and row["trigger"] == pytest.approx(31.4 * 1.0125)
    # synthetic bars open at their close (33.0), above the trigger: a buy-stop fills at the open
    assert row["entry_a"] == pytest.approx(max(row["open"], row["trigger"])) == 33.0
    assert bt.verify_against_live(p, cfg, ev, [p.dates[160], p.dates[159]]) == []


def test_equity_curve_never_levers_up_and_floors_tiny_stops():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    p = panel(o, h, lo, c)
    cfg = load_config("us").backtest
    # five same-day trades with a 0.1% stop: without the 1% floor each would want 500% of equity;
    # with it each wants 50%, capped at 25%; cash then runs out after four (4 x 25% = 100%)
    many = pd.concat([trade(stop=99.9).assign(symbol=f"S{i}") for i in range(5)], ignore_index=True)
    sim = many.assign(ret=0.02, r_multiple=0.0, exit_row=20, days=10, open=False)
    curve, stats = bt.equity_curve(sim, p, cfg)
    assert stats["taken"] == 4 and stats["skipped_full"] == 1
    assert curve.iloc[-1] == pytest.approx(100 * (1 + 4 * 0.25 * 0.02))


def test_r_multiple_uses_the_risk_floor():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    lo[12] = 99.0
    res = bt.simulate(trade(stop=99.95), panel(o, h, lo, c), RULE, cost_pct=0, min_risk_pct=1.0).iloc[0]
    assert res["ret"] == pytest.approx(99.95 / 100 - 1)
    assert res["r_multiple"] == pytest.approx((99.95 / 100 - 1) / 0.01)  # divided by the 1% floor, not 0.05%


# --- Model C: buy-stop, failed breakouts sold at that day's close -------------------------------


def test_model_c_sells_a_failed_fill_at_the_entry_day_close():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    c[10] = 98.5  # closed back below the 99.0 trigger
    tr = trade("A", entry=99.0, stop=97.0).assign(model="C", group="failed")
    res = bt.simulate(tr, panel(o, h, lo, c), RULE, cost_pct=0).iloc[0]
    assert res["exit_row"] == 10 and res["days"] == 0
    assert res["ret"] == pytest.approx(98.5 / 99.0 - 1)


def test_model_c_keeps_a_confirmed_fill_like_model_a():
    path = np.r_[102, 104, 106, 107, 98, np.full(N - 16, 98.0)]
    o, h, lo, c = flat_then(path)
    a = bt.simulate(trade("A"), panel(o, h, lo, c), RULE, 0).iloc[0]
    cc = bt.simulate(trade("A").assign(model="C"), panel(o, h, lo, c), RULE, 0).iloc[0]  # group "ok"
    assert cc["ret"] == pytest.approx(a["ret"]) and cc["exit_row"] == a["exit_row"]


def test_build_trades_model_c_has_model_a_entries():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    ev = pd.DataFrame([{"date": DATES[10], "symbol": "X", "row": 10, "trigger": 99.0, "open": 100, "high": 101,
                        "low": 96.0, "close": 98, "entry_a": 100.0, "stop": 96.0, "prev_low": 97.0, "status": "failed",
                        "vol_ratio": 0.8, "adr_prev": 5, "setup_age": 1, "flag_day_move": False}])
    a, cc = bt.build_trades(ev, panel(o, h, lo, c), "A"), bt.build_trades(ev, panel(o, h, lo, c), "C")
    assert cc["model"].tolist() == ["C"]
    pd.testing.assert_frame_equal(a.drop(columns="model"), cc.drop(columns="model"))


# --- market filter test --------------------------------------------------------------------------


def test_market_filter_hand_computed():
    # SMA3 of 10, 11, 12, 11, 9, 13: -, -, 11, 11.33, 10.67, 11 -> above it on rows 2 (12) and 5 (13) only
    bench_dates = DATES[:6]
    close = pd.Series([10, 11, 12, 11, 9, 13.0], index=bench_dates)
    on = bt.market_filter(close, DATES[:6], 3)
    assert on.tolist() == [False, False, True, False, False, True]


def test_market_filter_carries_forward_on_days_the_benchmark_did_not_trade():
    close = pd.Series([10, 11, 12.0], index=DATES[[1, 2, 3]])  # nothing on row 0 or 4
    on = bt.market_filter(close, DATES[:5], 2)
    # SMA2: -, 10.5, 11.5 -> on at rows 2 and 3; row 0 precedes the data (off); row 4 keeps row 3's value
    assert on.tolist() == [False, False, True, True, True]


def test_tag_market_filter_uses_the_close_before_entry():
    on = np.array([False, True, False, True, False])
    a = pd.concat([trade("A", entry_row=r) for r in (0, 1, 2, 4)], ignore_index=True)
    tagged = bt.tag_market_filter(a, on)
    # entry row 0 has no previous close (off); row 1 -> on[0]; row 2 -> on[1]; row 4 -> on[3]
    assert tagged["market_ok"].tolist() == [False, False, True, True]
    # Model B enters the session after the breakout, so its order day (the breakout) is also entry_row - 1
    b = bt.tag_market_filter(trade("B", entry_row=2), on)
    assert bool(b["market_ok"].iloc[0]) is True
    assert bt.tag_market_filter(trade().iloc[:0], on)["market_ok"].tolist() == []


def test_equity_curve_can_start_before_the_first_trade():
    o, h, lo, c = flat_then(np.full(N - 11, 100.0))
    sim = trade().assign(ret=0.10, r_multiple=2.0, exit_row=20, days=10, open=False)
    curve, _ = bt.equity_curve(sim, panel(o, h, lo, c), load_config("us").backtest, start_row=3)
    assert curve.index[0] == DATES[3] and curve.iloc[0] == 100.0 and curve.iloc[-1] == pytest.approx(101.0)


def test_market_filter_pass_mark():
    assert bt.market_filter_passes(0.002, 0.20, 0.16)
    assert not bt.market_filter_passes(-0.001, 0.20, 0.16)  # losing per trade
    assert not bt.market_filter_passes(0.002, 0.10, 0.16)  # behind the index
    assert not bt.market_filter_passes(np.nan, np.nan, 0.16)  # no trades


def test_market_filter_configured_for_both_markets():
    assert load_config("us").backtest.market_filter_sma == 50
    assert load_config("india").backtest.market_filter_sma == 50
