"""The other trader's screener (cfg.jack) and the names that pass it and are on one of our lists."""

import dataclasses
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scanner import indicators as ind
from scanner import scans
from scanner.config import JackConfig, load_config
from scanner.dashboard import build_site, snapshot, write_snapshot
from scanner.data.us_loader import chart_url
from scanner.notify import format_alerts, jack_rule_text
from scanner.state import MarketState, diff_results
from tests.synth import BASE_CLOSES, Bars, panels_from, textbook, us_cfg

J = JackConfig(ema_fast=21, sma_mid=50, sma_long=200, adr_window=20, min_adr_pct=4.0, adv_window=50,
               min_avg_traded_value=1_000_000)
UNIVERSE = pd.DataFrame({"symbol": ["X"], "source_symbol": ["X"], "name": ["X Corp"], "exchange": ["NASDAQ"]})


def ramp(n=260, slope=1.0, start=100.0, range_pct=5.0, volume=20_000.0) -> Bars:
    """close = start + slope x bar, so the 21-day EMA lags the close by about 10 points, the 50-day SMA by
    about 24.5 and the 200-day SMA by about 99.5 (an average lags by (n - 1) / 2 bars of slope)."""
    return Bars.from_close(start + slope * np.arange(n), range_pct, volume)


def with_last_close(b: Bars, drop: float) -> Bars:
    """The same series with the final close `drop` below its previous value (range and volume unchanged)."""
    return b.copy().set_bar(len(b) - 1, close=b.close[-2] - drop)


def passes(bars: dict[str, Bars], j: JackConfig = J) -> pd.DataFrame:
    return scans.jack_pass(panels_from(bars), j)


# --- the screener alone ----------------------------------------------------------------------------


def test_rising_liquid_stock_passes_and_numbers_are_hand_checked():
    r = passes({"UP": ramp()}).loc["UP"]
    assert r["passes"]
    assert r["adr_pct"] == pytest.approx(5.0)  # high / low - 1 = 5% every bar
    # last 50 bars: close = 100 + bar, bars 210..259, volume 20,000 -> mean close 334.5
    assert r["avg_traded_value"] == pytest.approx(334.5 * 20_000)


def test_each_trend_leg_can_fail_on_its_own():
    up = ramp()
    ema_drop = with_last_close(up, 15)      # below the 21-day EMA, still above both SMAs
    sma50_drop = with_last_close(up, 40)    # below the 50-day SMA too, still above the 200-day
    for b, below in ((ema_drop, ("ema",)), (sma50_drop, ("ema", "sma50"))):
        c = pd.Series(b.close)
        last = c.iloc[-1]
        assert (last < ind.ema(c, 21).iloc[-1]) == ("ema" in below)
        assert (last < ind.sma(c, 50).iloc[-1]) == ("sma50" in below)
        assert last > ind.sma(c, 200).iloc[-1]
    assert not passes({"A": ema_drop, "B": sma50_drop}).loc[["A", "B"], "passes"].any()
    assert passes({"UP": with_last_close(up, 5)}).loc["UP", "passes"]  # a small dip is still above the EMA

    # above the 21-day EMA and the 50-day SMA but under the 200-day: a bounce inside a longer downtrend
    bounce = np.r_[np.full(150, 200.0), np.full(60, 100.0), np.linspace(100, 150, 50)]
    b = Bars.from_close(bounce, 5.0, 20_000.0)
    c = pd.Series(b.close)
    assert c.iloc[-1] > ind.ema(c, 21).iloc[-1] and c.iloc[-1] > ind.sma(c, 50).iloc[-1]
    assert c.iloc[-1] < ind.sma(c, 200).iloc[-1]
    assert not passes({"BOUNCE": b}).loc["BOUNCE", "passes"]


def test_adr_and_dollar_volume_floors_are_strict():
    res = passes({"CALM": ramp(range_pct=3.0), "THIN": ramp(volume=1_000.0), "OK": ramp()})
    assert res["passes"].to_dict() == {"CALM": False, "THIN": False, "OK": True}
    # "greater than": a value exactly at the floor does not pass, one tick under it does
    adv, adr = res.loc["OK", "avg_traded_value"], res.loc["OK", "adr_pct"]
    assert not passes({"OK": ramp()}, J.model_copy(update={"min_avg_traded_value": adv})).loc["OK", "passes"]
    assert passes({"OK": ramp()}, J.model_copy(update={"min_avg_traded_value": adv - 1})).loc["OK", "passes"]
    assert not passes({"OK": ramp()}, J.model_copy(update={"min_adr_pct": adr})).loc["OK", "passes"]


def test_missing_data_fails_instead_of_passing():
    short = ramp(n=150)                      # no full 200-day average yet
    no_bar = ramp()
    for f in ("close", "high", "low", "volume"):
        getattr(no_bar, f)[-1] = np.nan      # no bar on the last session
    gap = ramp()
    gap.volume[-10] = np.nan                 # a missing volume inside the 50-day window
    res = passes({"SHORT": short, "NOBAR": no_bar, "GAP": gap, "OK": ramp()})
    assert res["passes"].to_dict() == {"SHORT": False, "NOBAR": False, "GAP": False, "OK": True}


def test_both_markets_configure_the_screen():
    us, india = load_config("us").jack, load_config("india").jack
    for j in (us, india):
        assert (j.ema_fast, j.sma_mid, j.sma_long, j.adr_window, j.min_adr_pct, j.adv_window) == (21, 50, 200, 20, 4.0, 50)
    assert us.min_avg_traded_value == 250_000_000   # the screenshot's figure, in USD
    assert india.min_avg_traded_value == 1_200_000_000  # Rs 120 crore: same rank, not the dollar figure converted
    assert "Rs" not in jack_rule_text(load_config("us")) and "$250.0M" in jack_rule_text(load_config("us"))
    assert "₹120.0 Cr" in jack_rule_text(load_config("india"))


# --- the screener against our lists ----------------------------------------------------------------


def results_by_hand(cfg, panels):
    """Our three results with hand-made tables: UP and DUP are leaders, DUP is also on the watchlist, and DOWN is a
    leader that fails the screener."""
    ts = panels.dates[-1]
    cols = {"close": 1.0, "ret_1m": 0.3, "ret_3m": 0.6, "ret_6m": 1.6, "flag_day_move": False}
    lead = pd.DataFrame({k: [v] * 3 for k, v in cols.items()}, index=["UP", "DUP", "DOWN"])
    lead.loc["UP", "ret_3m"], lead.loc["DUP", "ret_3m"] = 0.7, 0.9
    wl = pd.DataFrame({**{k: [v] for k, v in cols.items()}, "trigger": [11.0], "stop": [9.5]}, index=["DUP"])
    leaders = scans.LeadersResult(ts, "fixed", lead, {}, 4, 0, 4, list(cfg.leaders.windows))
    return (leaders, scans.WatchlistResult(ts, wl, 1), scans.BreakoutResult(ts, pd.DataFrame(), 0, 0))


def test_names_on_both_screens_strongest_list_first():
    cfg = us_cfg(jack=J.model_dump())
    p = panels_from({"UP": ramp(), "DUP": ramp(), "DOWN": with_last_close(ramp(), 15)})
    jr = scans.jack_screen(p, cfg, *results_by_hand(cfg, p))
    assert list(jr.table.index) == ["DUP", "UP"]                       # DOWN fails the screen
    assert jr.table["ours"].tolist() == ["watchlist", "leader"]        # DUP is on both lists: the watchlist wins
    assert jr.table.loc["DUP", ["trigger", "stop"]].tolist() == [11.0, 9.5]
    assert jr.table.loc["UP", ["trigger", "stop"]].isna().all()        # a leader only: no trigger yet
    assert (jr.passing_jack, jr.ours_count) == (2, 3)                  # UP and DUP pass; the three names are on our lists
    assert jr.table.loc["UP", "adr_pct"] == pytest.approx(5.0)
    assert jr.table.loc["UP", "avg_traded_value"] == pytest.approx(334.5 * 20_000)   # the 50-day figure, not 20


def test_a_screener_pass_that_is_on_none_of_our_lists_is_not_listed():
    cfg = us_cfg(jack=J.model_dump())
    p = panels_from({"UP": ramp(), "LONER": ramp()})
    ts = p.dates[-1]
    empty = pd.DataFrame(columns=["close", "ret_1m", "ret_3m", "ret_6m", "flag_day_move"])
    jr = scans.jack_screen(p, cfg, scans.LeadersResult(ts, "fixed", empty, {}, 2, 0, 2, ["1m", "3m", "6m"]),
                           scans.WatchlistResult(ts, empty, 0), scans.BreakoutResult(ts, empty, 0, 0))
    assert jr.table.empty and (jr.passing_jack, jr.ours_count) == (2, 0)


def textbook_cfg():
    # the textbook series is 160 bars long, so use windows it can fill
    return us_cfg(jack={"sma_long": 100, "adv_window": 20, "min_avg_traded_value": 1_000_000})


def test_run_scans_includes_the_screen_on_the_textbook_setup():
    cfg = textbook_cfg()
    res = scans.run_scans(panels_from({"X": textbook()}), cfg)
    row = res.jack.table.loc["X"]
    assert row["ours"] == "watchlist" and row["trigger"] == pytest.approx(31.4 * 1.0125)
    # last 20 bars: 140-149 run-up closes at 1.5M shares, 150-159 base closes at 0.6M shares
    run_up = np.linspace(20.0, 32.0, 31)[21:31]
    assert row["avg_traded_value"] == pytest.approx((run_up.sum() * 1.5e6 + sum(BASE_CLOSES) * 0.6e6) / 20)
    assert row["adr_pct"] == pytest.approx((10 * 8 + 5 * 5 + 5 * 2.5) / 20)   # 5.875
    assert res.jack.as_of == res.watchlist.as_of


def test_a_breakout_outranks_the_watchlist():
    b = textbook().append(33.0, 33.3, 31.9, 2_000_000)
    res = scans.run_scans(panels_from({"X": b}), textbook_cfg())
    assert res.jack.table.loc["X", "ours"] == "breakout" and res.jack.table.loc["X", "stop"] == pytest.approx(31.9)


# --- message and dashboard --------------------------------------------------------------------------


def alert_text(res, cfg):
    prev = MarketState(market="us", as_of=date(2026, 1, 1), active_count=1, watchlist={})
    d = diff_results(prev, res, "us", cfg.alerts)
    return "\n".join(format_alerts(d, res, cfg, UNIVERSE, chart_url))


def test_alert_message_names_the_companies_on_both_screens():
    cfg = textbook_cfg()
    res = scans.run_scans(panels_from({"X": textbook()}), cfg)
    text = alert_text(res, cfg)
    assert "🎯 <b>On both screens</b>" in text
    assert "above 21 EMA, 50 &amp; 100 SMA, ADR above 4%, 20-day value above $1.0M" in text    # escaped for HTML
    assert "1 of 1 names that pass their screen are on our lists." in text
    assert "<b>X</b> · X Corp\nwatchlist · 31.40 · trig 31.79 · stop " in text
    assert '<a href="https://www.tradingview.com/chart/?symbol=NASDAQ:X">chart</a>' in text


def test_alert_message_says_when_there_are_none_and_skips_it_when_not_computed():
    cfg = textbook_cfg()
    res = scans.run_scans(panels_from({"X": textbook()}), cfg)
    none = dataclasses.replace(res, jack=dataclasses.replace(res.jack, table=res.jack.table.iloc[0:0], passing_jack=4))
    assert "None today. 4 names pass their screen alone." in alert_text(none, cfg)
    assert "On both screens" not in alert_text(dataclasses.replace(res, jack=None), cfg)


def test_flagged_name_on_both_screens_brings_the_flag_note():
    cfg = textbook_cfg()
    res = scans.run_scans(panels_from({"X": textbook()}), cfg)
    table = res.jack.table.assign(flag_day_move=True)
    wl = res.watchlist.table  # the watchlist entry itself is not flagged, only the combined row is
    text = alert_text(dataclasses.replace(res, jack=dataclasses.replace(res.jack, table=table)), cfg)
    assert not wl["flag_day_move"].any() and "<b>X</b> ⚠ · X Corp\nwatchlist" in text and "⚠ = a one-day move" in text


def test_dashboard_section_and_old_snapshots(tmp_path):
    cfg = textbook_cfg()
    res = scans.run_scans(panels_from({"X": textbook()}), cfg)
    snap = snapshot(res, cfg, UNIVERSE, chart_url)
    item = snap["jack"]["items"][0]
    assert (item["symbol"], item["ours"], snap["jack"]["passing"], snap["jack"]["ours"]) == ("X", "watchlist", 1, 1)
    assert item["trigger"] == round(31.4 * 1.0125, 2) and snap["jack"]["rule"].startswith("above 21 EMA, 50 & 100 SMA")
    data, reports, out = tmp_path / "data", tmp_path / "reports", tmp_path / "site"
    reports.mkdir()
    write_snapshot(snap, data)
    build_site(out, data, reports)
    page = (out / "index.html").read_text()
    assert "On both screens (1)" in page and ">X</a>" in page and "50 &amp; 100 SMA" in page
    # a snapshot written before the section existed still builds, without the section
    old = json.loads((data / "us.json").read_text())
    del old["jack"]
    (data / "us.json").write_text(json.dumps(old))
    build_site(out, data, reports)
    assert "On both screens" not in (out / "index.html").read_text()
    # nothing on both screens: the section says so and gives the count passing the screener alone
    snap["jack"]["items"], snap["jack"]["passing"] = [], 9
    write_snapshot(snap, data)
    build_site(out, data, reports)
    assert "On both screens (0)" in (out / "index.html").read_text() and "9 names pass the other screener alone" in (out / "index.html").read_text()
