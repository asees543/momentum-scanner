"""Study set: which past triggers become cards, and what each card contains."""

import json

import numpy as np
import pandas as pd
import pytest

from scanner import study
from scanner.config import StudyConfig, load_config
from scanner.data.base import Panels
from tests.synth import panels_from, textbook, us_cfg


def sim_frame(rows) -> pd.DataFrame:
    """rows of (symbol, entry_row, ret, group); flags default to False and can be set afterwards."""
    df = pd.DataFrame(rows, columns=["symbol", "entry_row", "ret", "group"])
    return df.assign(flag_day_move=False, open=False)


def only(**counts) -> StudyConfig:
    base = {"winners": 0, "failures": 0, "confirmed_losers": 0, "random": 0}
    return StudyConfig(**{**base, **counts})


def tagged(cards: pd.DataFrame, tag: str) -> pd.DataFrame:
    return cards[[tag in t for t in cards["tags"]]]


# --- choosing the cards ---------------------------------------------------------------------------


def test_winners_are_the_best_returns_one_card_per_move():
    sim = sim_frame([
        ("A", 100, 0.50, "ok"),
        ("A", 110, 0.40, "ok"),      # 10 sessions after A@100: the same move, skipped
        ("A", 130, 0.30, "ok"),      # 30 sessions after the chosen A@100: a new move
        ("B", 100, 0.20, "too wide"),
        ("C", 100, 0.10, "failed"),
        ("D", 100, -0.10, "ok"),     # a loser is never a "winner", however few winners there are
    ])
    cards = study.pick_cards(sim, only(winners=3, winner_spacing=20), n_rows=500)
    assert sorted(zip(cards["symbol"], cards["entry_row"])) == [("A", 100), ("A", 130), ("B", 100)]
    assert all(t == ["winner"] for t in cards["tags"])
    assert len(study.pick_cards(sim, only(winners=10), n_rows=500)) == 4  # A@100, A@130, B, C


def test_flagged_open_recent_and_unpriced_trades_are_never_cards():
    sim = sim_frame([(f"S{i}", 100, 0.05, "ok") for i in range(5)] + [("LATE", 485, 0.05, "ok")])
    sim.loc[0, "flag_day_move"] = True      # possible unadjusted split
    sim.loc[1, "open"] = True               # still open at the end of the data: no result yet
    sim.loc[2, "ret"] = np.nan
    cfg = only(winners=10, random=10, min_bars_after=20)
    cards = study.pick_cards(sim, cfg, n_rows=500)
    # LATE: entry row 485 + 20 sessions = 505 is past the last row (499), so its chart would be cut short
    assert sorted(cards["symbol"]) == ["S3", "S4"]
    assert "LATE" in set(study.pick_cards(sim, cfg, n_rows=506)["symbol"])  # 485 + 20 = 505 = last row


def test_each_deck_draws_from_its_own_group():
    rows = [(f"F{i}", 100 + i, -0.02, "failed") for i in range(30)]
    rows += [(f"L{i}", 100 + i, -0.05, "ok" if i % 2 else "too wide") for i in range(10)]
    rows += [(f"W{i}", 100 + i, 0.10 + i / 100, "ok") for i in range(5)]
    cards = study.pick_cards(sim_frame(rows), only(winners=2, failures=4, confirmed_losers=3, random=6), n_rows=500)
    assert sorted(tagged(cards, "winner")["symbol"]) == ["W3", "W4"]          # the two best returns
    failed, lost = tagged(cards, "failed"), tagged(cards, "lost")
    assert len(failed) == 4 and set(failed["group"]) == {"failed"}
    assert len(lost) == 3 and "failed" not in set(lost["group"]) and (lost["ret"] < 0).all()
    assert len(tagged(cards, "random")) == 6
    assert cards.index.is_unique  # a trade picked by two decks is one card with two tags
    assert sum(len(t) for t in cards["tags"]) == 2 + 4 + 3 + 6


def test_a_deck_larger_than_its_group_takes_the_whole_group():
    rows = [(f"F{i}", 100, -0.02, "failed") for i in range(7)] + [("W", 100, 0.1, "ok")]
    cards = study.pick_cards(sim_frame(rows), only(failures=50, confirmed_losers=50), n_rows=500)
    assert len(tagged(cards, "failed")) == 7 and len(tagged(cards, "lost")) == 0


def test_same_seed_same_cards_in_the_same_order():
    rows = [(f"S{i}", 100 + i, (i % 7 - 3) / 100, "failed" if i % 3 else "ok") for i in range(200)]
    cfg = only(winners=5, failures=10, confirmed_losers=10, random=20)
    a, b = study.pick_cards(sim_frame(rows), cfg, 500), study.pick_cards(sim_frame(rows), cfg, 500)
    assert list(a.index) == list(b.index) and list(a["tags"]) == list(b["tags"])
    c = study.pick_cards(sim_frame(rows), cfg.model_copy(update={"seed": 8}), 500)
    assert list(c.index) != list(a.index)


def test_no_trades_gives_no_cards():
    empty = sim_frame([])
    assert study.pick_cards(empty, StudyConfig(), 100).empty


# --- one card -------------------------------------------------------------------------------------


def ramp(n: int) -> Panels:
    """One symbol: close = 100 + row, open = close - 0.5, high/low = close +- 1, volume = (row + 1) x 1000."""
    dates = pd.bdate_range("2026-01-05", periods=n)
    c = 100.0 + np.arange(n)
    f = lambda a: pd.DataFrame({"X": np.asarray(a, dtype=float)}, index=dates)
    return Panels(open=f(c - 0.5), high=f(c + 1), low=f(c - 1), close=f(c), volume=f((np.arange(n) + 1) * 1000.0))


def one_trade(**kw):
    base = dict(entry_row=30, col=0, exit_row=33, tags=["random"], entry=131.0, trigger=130.5, prev_low=128.0,
                group="ok", ret=0.0123, r_multiple=0.537, days=3, adr_prev=2.04, risk_pct=2.29, vol_ratio=1.8,
                setup_age=2)
    return next(pd.DataFrame([{**base, **kw}]).itertuples(index=False))


def make_card(p: Panels, tr, **study_cfg):
    cfg = us_cfg(study=study_cfg)
    arr = {f: getattr(p, f).to_numpy() for f in ("open", "high", "low", "close", "volume")}
    smas = {k: study.ind.sma(p.close, k).to_numpy() for k in study.CHART_SMAS}
    return study.card(tr, 7, p, arr, smas, cfg, {"symbol": "X", "name": "X Corp"})


def test_card_window_cut_and_values_hand_checked():
    p = ramp(45)
    for f in ("open", "high", "low", "close", "volume"):
        getattr(p, f).iloc[12] = np.nan                      # one missing bar inside the window
    k = make_card(p, one_trade(), bars_before=20, bars_after=5)
    # start = 30 - 20 = 10; end = max(30 + 5, exit 33 + 5 margin) = 38 -> 29 sessions, trigger day at index 20
    assert k["cut"] == 20 and len(k["c"]) == 29 and k["exit"] == 23
    assert k["d"][k["cut"]] == k["date"] == int(f"{p.dates[30]:%Y%m%d}") and k["d"][k["cut"] - 1] < k["date"]
    assert k["c"][0] == 110.0 and k["c"][20] == 130.0 and k["o"][20] == 129.5 and k["h"][20] == 131.0 and k["l"][20] == 129.0
    assert all(k[f][2] is None for f in ("o", "h", "l", "c", "v"))   # row 12 -> None, not NaN or zero
    # volume is relative to the window's largest bar (row 38: 39,000): row 10 -> 1000 x 11,000 / 39,000 = 282
    assert k["v"][-1] == 1000 and k["v"][0] == 282
    # SMA10 on the trigger day = mean of closes 121..130 = 125.5; rows 12..21 have the missing bar in their window
    assert k["s10"][20] == 125.5 and k["s10"][11] is None and k["s10"][12] == pytest.approx(117.5)
    assert all(x is None for x in k["s50"])                  # fewer than 50 sessions so far
    # 1M return the evening before: close_29 / close_(29-22) - 1 = 129 / 107 - 1; 3M and 6M need more history
    assert k["rets"] == {"1m": pytest.approx(100 * (129 / 107 - 1), abs=0.005), "3m": None, "6m": None}
    # forward returns from the entry (131): close_35 = 135, close_40 = 140; row 50 is past the data
    assert k["fwd"] == {"5": pytest.approx(100 * (135 / 131 - 1), abs=0.005),
                        "10": pytest.approx(100 * (140 / 131 - 1), abs=0.005), "20": None}
    assert k["gap"] == pytest.approx(100 * (130.5 / 129 - 1), abs=0.005)   # trigger above the last close
    assert (k["trigger"], k["stop"], k["entry"], k["status"]) == (130.5, 128.0, 131.0, "ok")
    assert (k["ret"], k["r"], k["days"], k["age"], k["tags"]) == (1.23, 0.537, 3, 2, ["random"])
    json.dumps(k, allow_nan=False)                            # nothing un-serialisable (NaN, numpy types)


def test_card_window_is_clipped_to_the_data():
    k = make_card(ramp(40), one_trade(entry_row=5, exit_row=39), bars_before=20, bars_after=5)
    assert k["cut"] == 5 and len(k["c"]) == 40 and k["exit"] == 39     # start 0, end = last row


def test_card_shows_a_late_exit_but_not_forever():
    p = ramp(400)
    k = make_card(p, one_trade(entry_row=50, exit_row=120), bars_before=20, bars_after=5)
    assert len(k["c"]) == (120 + study.EXIT_MARGIN) - 30 + 1 and k["exit"] == 90
    k = make_card(p, one_trade(entry_row=50, exit_row=360), bars_before=20, bars_after=5)
    assert len(k["c"]) == (50 + study.MAX_BARS_AFTER) - 30 + 1 and k["exit"] is None  # the sale is off the chart


# --- the whole study and the page ------------------------------------------------------------------

UNIVERSE = pd.DataFrame({"symbol": ["X"], "source_symbol": ["X.Y"], "name": ["X </script> Corp - Common Stock"],
                         "exchange": ["NASDAQ"]})


def textbook_breakout_study() -> tuple[dict, Panels]:
    b = textbook().append(33.0, 33.3, 31.9, 2_000_000)        # bar 160: confirmed breakout, entry 33.0 (open)
    for close in np.r_[np.linspace(33.5, 38.0, 10), np.full(15, 33.2)]:   # runs up, then falls under its 10-day average
        b.append(close, close * 1.01, close * 0.99, 1_000_000)
    p = panels_from({"X": b})
    cfg = us_cfg(study={"winners": 5, "failures": 5, "confirmed_losers": 5, "random": 10})
    return study.build_study(p, cfg, UNIVERSE, warmup=1), p


def test_build_study_on_the_textbook_breakout():
    st, p = textbook_breakout_study()
    k = next(c for c in st["cards"] if c["date"] == int(f"{p.dates[160]:%Y%m%d}"))
    assert k["status"] == "ok" and k["entry"] == 33.0 and k["trigger"] == pytest.approx(31.4 * 1.0125, abs=1e-3)
    assert k["cut"] == 130 and k["symbol"] == "X.Y"            # 130 sessions of context, the exchange's symbol
    assert k["stop"] == pytest.approx(p.low["X"].iloc[159], rel=1e-4)       # the previous day's low
    assert k["ret"] > 0 and set(k["tags"]) == {"winner", "random"}
    assert st["exit"] == "sell 1/3 after 3d, trail 10-SMA" and st["windows"] == ["1m", "3m", "6m"]
    assert st["all"]["n"] >= 1 and {d["tag"] for d in st["decks"]} == set(study.DECKS)
    assert next(d["n"] for d in st["decks"] if d["tag"] == "winner") >= 1


def test_page_is_self_contained_and_survives_a_hostile_name():
    st, _ = textbook_breakout_study()
    page = study.render_page(st)
    assert "<script src" not in page and "<link" not in page and "http" not in page.split("<script", 1)[0]
    assert page.count("</script>") == 2                        # the data block and the code block, nothing else
    raw = page.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(raw) == st                               # the embedded data is exactly the study
    assert study.render_page(st) == page                       # deterministic: no build time


def test_study_config_for_both_markets():
    for market in ("us", "india"):
        cfg = load_config(market)
        s = cfg.study
        assert min(s.winners, s.failures, s.confirmed_losers, s.random) > 0
        # the chart must reach back over the longest leader window, or the prior move is off-screen
        assert s.bars_before >= max(w.lookback for w in cfg.leaders.windows.values())
