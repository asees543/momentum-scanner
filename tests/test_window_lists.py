"""Per-window leader lists (1M / 3M / 6M, each ranked by its own gain) with a one-day value floor."""

import dataclasses
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scanner import scans
from scanner.config import LeadersConfig, load_config
from scanner.dashboard import build_site, snapshot, write_snapshot
from scanner.data.us_loader import chart_url
from scanner.notify import LIST_CHUNK, format_alerts, window_list_blocks
from scanner.state import MarketState, diff_results
from tests.synth import Bars, panels_from, textbook, us_cfg

UNIVERSE = pd.DataFrame({"symbol": ["X"], "source_symbol": ["X"], "name": ["X Corp"], "exchange": ["NASDAQ"]})


def bars(close: float, volume: float, earlier_volume: float | None = None) -> Bars:
    """Three flat bars; only the last one's close x volume matters to the lists."""
    v = np.array([volume if earlier_volume is None else earlier_volume] * 2 + [volume], dtype=float)
    return Bars.from_close([close] * 3, 5.0, v)


def leaders_table(rows: dict) -> scans.LeadersResult:
    """rows: symbol -> (ret_1m, ret_3m, ret_6m, pass_1m, pass_3m, pass_6m)."""
    t = pd.DataFrame(rows, index=["ret_1m", "ret_3m", "ret_6m", "pass_1m", "pass_3m", "pass_6m"]).T
    t[["pass_1m", "pass_3m", "pass_6m"]] = t[["pass_1m", "pass_3m", "pass_6m"]].astype(bool)
    t = t.assign(close=1.0, flag_day_move=False)
    return scans.LeadersResult(pd.Timestamp("2026-03-04"), "fixed", t, {}, len(t), 0, len(t), ["1m", "3m", "6m"])


def test_each_window_is_ranked_by_its_own_gain_with_a_strict_one_day_floor():
    cfg = us_cfg(leaders={"min_day_value": 1_000_000})
    nan_volume = bars(10, 500_000)
    nan_volume.volume[-1] = np.nan
    p = panels_from({
        "A": bars(10, 200_000),                       # 2,000,000 today: passes
        "F": bars(10, 300_000),                       # 3,000,000, same 1M gain as A
        "B": bars(10, 100_000),                       # exactly 1,000,000: "greater than" excludes it
        "C": bars(20, 100_000),                       # 2,000,000
        "D": nan_volume,                              # no volume today: fails instead of passing
        "E": bars(10, 10_000, earlier_volume=9e6),    # huge volume on earlier days, only 100,000 today
    })
    res = leaders_table({
        "A": (0.5, 0.4, 0.1, True, True, False),
        "F": (0.5, 0.2, 0.0, True, True, False),
        "B": (0.8, 0.8, 0.8, True, True, True),
        "C": (0.9, 0.1, 3.0, True, False, True),
        "D": (0.7, 0.7, 0.7, True, True, True),
        "E": (0.6, 0.6, 0.6, True, True, True),
    })
    lists = scans.window_lists(p, cfg, res)
    assert list(lists) == ["1m", "3m", "6m"]
    assert list(lists["1m"].index) == ["C", "A", "F"]    # 0.9, then the 0.5 tie in symbol order
    assert list(lists["3m"].index) == ["A", "F"]         # C does not pass 3M; sorted 0.4 then 0.2
    assert list(lists["6m"].index) == ["C"]              # A and F do not pass 6M
    assert lists["1m"].loc["C", "day_value"] == pytest.approx(2_000_000)
    assert lists["3m"].loc["A", ["ret_1m", "ret_3m", "ret_6m"]].tolist() == [0.5, 0.4, 0.1]   # every window's return
    # no floor: B (exactly 1,000,000) and E (100,000) join; D still has no volume today, so it is still out
    open_lists = scans.window_lists(p, us_cfg(leaders={"min_day_value": 0}), res)
    assert list(open_lists["1m"].index) == ["C", "B", "E", "A", "F"]      # 0.9, 0.8, 0.6, then the 0.5 tie


def test_a_leader_not_in_the_table_is_never_listed_and_an_empty_table_gives_empty_lists():
    cfg = us_cfg(leaders={"min_day_value": 1})
    p = panels_from({"A": bars(10, 200_000)})
    empty = leaders_table({"A": (0.5, 0.4, 0.1, False, False, False)})
    assert all(df.empty for df in scans.window_lists(p, cfg, empty).values())
    none = scans.LeadersResult(pd.Timestamp("2026-03-04"), "fixed", empty.table.iloc[0:0], {}, 0, 0, 0, ["1m", "3m", "6m"])
    assert all(df.empty and df.index.name == "symbol" for df in scans.window_lists(p, cfg, none).values())


def test_both_markets_set_the_user_floor():
    assert load_config("us").leaders.min_day_value == 5_000_000      # the user's C*V/100 > 50,000
    assert load_config("india").leaders.min_day_value == 465_000_000   # $5M at 93.3 rupees per dollar, not 5M rupees
    assert LeadersConfig.model_fields["min_day_value"].default == 0   # off unless configured


# --- message ----------------------------------------------------------------------------------------


def textbook_results(floor):
    cfg = us_cfg(leaders={"min_day_value": floor})
    return cfg, scans.run_scans(panels_from({"X": textbook()}), cfg)


def alert_text(res, cfg):
    prev = MarketState(market="us", as_of=date(2026, 1, 1), active_count=1, watchlist={})
    return "\n".join(format_alerts(diff_results(prev, res, "us", cfg.alerts), res, cfg, UNIVERSE, chart_url))


def test_alert_message_has_one_list_per_window():
    cfg, res = textbook_results(5_000_000)
    text = alert_text(res, cfg)
    # textbook: up 57% over 3 months (31.4 / 20), not enough for the 1M or 6M rule; value 31.4 x 600,000 = $18.8M
    assert "📈 <b>3M leaders</b> · 1 names, biggest 3-month gain first (last-session value above $5.0M)" in text
    assert "1. <b>X</b> · X Corp · +57% · $18.8M" in text
    assert "📈 <b>1M leaders</b> · 0 names, biggest 1-month gain first (last-session value above $5.0M)\nNone today." in text
    assert text.index("1M leaders") < text.index("3M leaders") < text.index("6M leaders")
    # a floor above the day's value empties the list
    cfg2, res2 = textbook_results(20_000_000)
    assert "3M leaders</b> · 0 names" in alert_text(res2, cfg2)
    assert "leaders</b>" not in alert_text(dataclasses.replace(res, window_lists=None), cfg)


def test_long_lists_split_between_names_with_the_header_on_the_first_chunk():
    n = 2 * LIST_CHUNK + 5
    table = pd.DataFrame({"close": 1.0, "ret_1m": np.linspace(2.0, 0.3, n), "day_value": 6e6, "flag_day_move": False},
                         index=[f"S{i:02d}" for i in range(n)])
    cfg = us_cfg()
    blocks = window_list_blocks("1m", table, cfg, lambda s: "", lambda s: s)
    assert len(blocks) == 3 and blocks[0].startswith("📈 <b>1M leaders</b> · 45 names")
    assert sum("1M leaders" in b for b in blocks) == 1
    lines = "\n".join(blocks).split("\n")
    assert [l for l in lines if l[:1].isdigit()][0].startswith("1. <b>S00</b> · +200% · $6.0M")
    assert [l for l in lines if l[:1].isdigit()][-1].startswith(f"{n}. <b>S{n - 1:02d}</b>")
    assert all(len(b) < 4000 for b in blocks)


def test_flagged_name_in_a_list_brings_the_flag_note():
    cfg, res = textbook_results(5_000_000)
    wl = {w: df.assign(flag_day_move=True) for w, df in res.window_lists.items()}
    text = alert_text(dataclasses.replace(res, window_lists=wl), cfg)
    assert "1. <b>X</b> ⚠ · X Corp · +57%" in text and "⚠ = a one-day move" in text


# --- dashboard --------------------------------------------------------------------------------------


def test_dashboard_panels_and_old_snapshots(tmp_path):
    cfg, res = textbook_results(5_000_000)
    snap = snapshot(res, cfg, UNIVERSE, chart_url)
    assert snap["leader_lists"]["min_day_value"] == "$5.0M"
    item = snap["leader_lists"]["lists"]["3m"][0]
    assert (item["symbol"], item["day_value"]) == ("X", "$18.8M") and item["returns"]["3m"] == pytest.approx(0.57, abs=0.005)
    assert snap["leader_lists"]["lists"]["1m"] == []
    data, reports, out = tmp_path / "data", tmp_path / "reports", tmp_path / "site"
    reports.mkdir()
    write_snapshot(snap, data)
    build_site(out, data, reports)
    page = (out / "index.html").read_text()
    assert "<h2>3M leaders (1)</h2>" in page and "<h2>1M leaders (0)</h2>" in page and "None on this session." in page
    assert "traded more than $5.0M in the last session" in page and ">X</a>" in page
    old = json.loads((data / "us.json").read_text())
    del old["leader_lists"]
    (data / "us.json").write_text(json.dumps(old))
    build_site(out, data, reports)
    assert "leaders (" not in (out / "index.html").read_text()
