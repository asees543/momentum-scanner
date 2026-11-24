"""Sector map (lookups, caching, failures) and sector strength (hand-built groups)."""

import dataclasses
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scanner import run as run_mod
from scanner import scans
from scanner import sectors as sec
from scanner.config import SectorConfig, load_config
from scanner.dashboard import build_site, snapshot, write_snapshot
from scanner.data.us_loader import chart_url
from scanner.notify import format_alerts, sector_lines
from scanner.state import MarketState, diff_results
from tests.synth import Bars, panels_from, textbook, us_cfg

TODAY = date(2026, 11, 24)
SC = SectorConfig(min_names_sector=5, min_names_industry=3, change_sessions=5, sma=50, message_sectors=2,
                  message_industries=2, dashboard_industries=25, max_new_per_run=3, stale_days=180, retry_days=30)


def mapping(rows: dict) -> pd.DataFrame:
    """rows: symbol -> (sector, industry, as_of, status)."""
    df = pd.DataFrame(rows, index=["sector", "industry", "as_of", "status"]).T
    df.index.name = "symbol"
    return df


# --- the map ---------------------------------------------------------------------------------------------


def test_map_round_trip_sorted_and_empty(tmp_path):
    assert sec.load_map("us", tmp_path).empty and list(sec.load_map("us", tmp_path).columns) == ["sector", "industry", "as_of", "status"]
    df = mapping({"B": ("Tech", "Software", "2026-11-01", "ok"), "A": ("", "", "2026-11-01", "none")})
    path = sec.save_map(df, "us", tmp_path)
    assert path.read_text().splitlines()[1].startswith("A,")           # sorted by symbol
    back = sec.load_map("us", tmp_path)
    assert back.loc["B", "sector"] == "Tech" and back.loc["A", "status"] == "none" and back.loc["A", "sector"] == ""


def test_needs_lookup_rules():
    ok = lambda d, s="ok": pd.Series({"sector": "T", "industry": "I", "as_of": d, "status": s})
    assert sec.needs_lookup(None, TODAY, SC)                                  # not in the map
    assert not sec.needs_lookup(ok("2026-11-01"), TODAY, SC)                  # fresh
    assert sec.needs_lookup(ok("2026-05-01"), TODAY, SC)                      # classified over 180 days ago
    assert not sec.needs_lookup(ok("2026-11-01", "none"), TODAY, SC)          # no sector, tried 23 days ago
    assert sec.needs_lookup(ok("2026-10-20", "none"), TODAY, SC)              # no sector, tried 35 days ago: retry
    assert sec.needs_lookup(ok("garbage"), TODAY, SC)


def test_update_looks_up_only_what_is_missing_in_the_given_order_up_to_the_cap():
    calls = []

    def fetch(s):
        calls.append(s)
        return None if s == "NOSEC" else ("Tech", f"Ind-{s}")

    current = mapping({"HAVE": ("Tech", "X", "2026-11-20", "ok")})
    new, st = sec.update_map(["HAVE", "C", "NOSEC", "A", "B", "D"], current, SC, fetch, today=TODAY)
    assert calls == ["C", "NOSEC", "A"]                                        # HAVE is fresh; cap of 3; order kept
    assert (st.looked_up, st.classified, st.none, st.skipped_for_cap) == (3, 2, 1, 2)   # B and D wait for the next run
    assert new.loc["C"].tolist() == ["Tech", "Ind-C", "2026-11-24", "ok"]
    assert new.loc["NOSEC"].tolist() == ["", "", "2026-11-24", "none"]
    assert "B" not in new.index and new.loc["HAVE", "as_of"] == "2026-11-20"   # untouched
    assert len(current) == 1                                                   # the input map is not modified


def test_update_survives_errors_and_stops_when_the_source_refuses():
    def flaky(s):
        if s in ("E1", "E2"):
            raise RuntimeError("429")
        return ("Tech", "I")

    new, st = sec.update_map(["E1", "A", "E2", "B"], pd.DataFrame(columns=sec.MAP_COLUMNS[1:]), SC.model_copy(update={"max_new_per_run": 10}),
                             flaky, today=TODAY)
    assert list(new.index) == ["A", "B"] and st.errors == 2 and not st.stopped_early   # errors are not stored: retried next run

    def refusing(s):
        raise RuntimeError("rate limited")

    new, st = sec.update_map([f"S{i}" for i in range(20)], pd.DataFrame(columns=sec.MAP_COLUMNS[1:]),
                             SC.model_copy(update={"max_new_per_run": 20}), refusing, today=TODAY, max_consecutive_errors=4)
    assert st.stopped_early and st.errors == 4 and new.empty                    # gave up after 4 in a row


def test_a_failing_update_never_breaks_the_scan(monkeypatch):
    cfg = us_cfg()
    p = panels_from({"X": textbook()})
    current = mapping({"X": ("Tech", "Software", "2026-11-01", "ok")})
    monkeypatch.setattr(run_mod.sectors_mod, "update_map", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert run_mod.update_sector_map(p, cfg, current, "us") is current


# --- the strength ----------------------------------------------------------------------------------------

N = 100


def flat_then(last: float, dip_at_minus5: float | None = None, base: float = 100.0, step_up: float | None = None) -> Bars:
    """100 sessions at `base`. Optionally `step_up` from session 90 on (so the stock is above its 50-day average well before
    the end), or a dip `dip_at_minus5` five sessions before the end and 95 for the four after it; the last close is `last`.
    (SMA50 stays near 100, so the dip is below it and a last close of 120 is above it.)"""
    c = np.full(N, base, dtype=float)
    if step_up is not None:
        c[90:] = step_up
    if dip_at_minus5 is not None:
        c[N - 6], c[N - 5:N - 1] = dip_at_minus5, 95.0
    c[-1] = last
    return Bars.from_close(c, 5.0, 1_000_000.0)


def groups():
    """Tech: 5 names that stepped up to 120 ten sessions ago and are at 150 (above their 50-day average then and now). Util: 5 names at 80 (below, below).
    Mixed: 5 names that dipped to 90 five sessions ago and are at 120 now (below then, above now). Tiny: 2 names.
    Plus three liquid names with no sector in the map."""
    bars, rows = {}, {}
    for i in range(5):
        bars[f"T{i}"] = flat_then(150.0, step_up=120.0); rows[f"T{i}"] = ("Tech", "Software" if i < 3 else "Hardware", "2026-11-01", "ok")
        bars[f"U{i}"] = flat_then(80.0); rows[f"U{i}"] = ("Utilities", "Power", "2026-11-01", "ok")
        bars[f"M{i}"] = flat_then(120.0, dip_at_minus5=90.0); rows[f"M{i}"] = ("Mixed", "Banks", "2026-11-01", "ok")
    for i in range(2):
        bars[f"Y{i}"] = flat_then(130.0); rows[f"Y{i}"] = ("Tiny", "Tiny", "2026-11-01", "ok")
    for i in range(3):
        bars[f"Z{i}"] = flat_then(130.0)                                          # liquid but not in the map
    rows["Z0"] = ("", "", "2026-11-01", "none")                                  # in the map, no sector
    return bars, mapping(rows)


def strength(**cfg_updates):
    bars, smap = groups()
    p = panels_from(bars)
    cfg = us_cfg(sectors={"min_names_industry": 3, **cfg_updates})
    leaders = pd.Index(["T0", "T1", "T2", "M0"])                                 # 3 of 5 Tech, 1 of 5 Mixed
    return sec.sector_strength(p, cfg, smap, p.close.columns, leaders, pd.Index(["T0", "U0"]), pd.Index(["T1", "T2", "M4"])), cfg


def test_group_measures_are_hand_checked():
    sr, _ = strength()
    s = sr.sectors
    assert list(s.index) == ["Tech", "Mixed", "Utilities"]                       # Tiny (2 names) is not ranked
    assert s.loc["Tech", ["names", "leaders"]].tolist() == [5, 3] and s.loc["Tech", "leader_pct"] == pytest.approx(60)
    assert s.loc["Mixed", "leader_pct"] == pytest.approx(20) and s.loc["Utilities", "leader_pct"] == 0
    assert s["above_pct"].to_dict() == pytest.approx({"Tech": 100, "Mixed": 100, "Utilities": 0})
    assert s["d_above"].to_dict() == pytest.approx({"Tech": 0, "Mixed": 100, "Utilities": 0})   # Mixed went from below to above
    # median return over the 3-month window: close_t / close_(t-67) - 1
    assert s["median_ret"].to_dict() == pytest.approx({"Tech": 0.5, "Mixed": 0.2, "Utilities": -0.2})
    assert s.loc["Tech", ["watchlist", "trigger_bars"]].tolist() == [1, 2] and s.loc["Utilities", "watchlist"] == 1
    assert s.loc["Mixed", "trigger_bars"] == 1
    # score = mean of the percentile ranks of leader_pct, above_pct, median_ret among the 3 ranked groups
    assert s["score"].to_dict() == pytest.approx({"Tech": (100 + 250 / 3 + 100) / 3, "Mixed": (200 / 3 + 250 / 3 + 200 / 3) / 3,
                                                  "Utilities": 100 / 3})


def test_coverage_counts_only_names_with_a_sector():
    sr, _ = strength()
    assert sr.liquid == 20 and sr.classified == 17                              # 15 + 2 Tiny; Z0 has no sector, Z1 and Z2 no entry
    assert sr.coverage == pytest.approx(17 / 20)
    assert sr.sector_of["T0"] == "Tech" and "Z0" not in sr.sector_of


def test_industries_need_enough_names_and_say_their_sector():
    sr, _ = strength()
    ind_ = sr.industries
    assert set(ind_.index) == {"Software", "Power", "Banks"}                     # Hardware (2) and Tiny (2) are under 3 names
    assert ind_.loc["Software", "sector"] == "Tech" and ind_.loc["Software", ["names", "leaders"]].tolist() == [3, 3]
    assert ind_.index[0] == "Software"


def test_no_group_big_enough_gives_empty_tables():
    sr, _ = strength(min_names_sector=50, min_names_industry=50)
    assert sr.sectors.empty and sr.industries.empty


def test_run_scans_includes_the_sectors_only_when_given_a_map():
    cfg = us_cfg()
    p = panels_from({"X": textbook()})
    assert scans.run_scans(p, cfg).sectors is None
    smap = mapping({"X": ("Tech", "Software", "2026-11-01", "ok")})
    res = scans.run_scans(p, cfg, sector_map=smap)
    assert res.sectors.liquid >= 1 and res.sectors.classified == res.sectors.liquid and res.sectors.sectors.empty   # 1 name < 5


def test_both_markets_configure_the_sectors():
    for market in ("us", "india"):
        s = load_config(market).sectors
        assert (s.min_names_sector, s.sma, s.change_sessions) == (5, 50, 5) and s.max_new_per_run > 0


# --- message and dashboard -------------------------------------------------------------------------------


def results_with_sectors():
    sr, cfg = strength()
    cfg = cfg.model_copy(update={"sectors": cfg.sectors.model_copy(update={"message_sectors": 2, "message_industries": 2})})
    base = scans.run_scans(panels_from({"X": textbook()}), us_cfg())
    return cfg, dataclasses.replace(base, sectors=sr)


def test_message_lists_the_strongest_sectors_the_weakest_and_the_industries():
    cfg, res = results_with_sectors()
    blocks = sector_lines(res.sectors, cfg)
    text = "\n".join(blocks)
    assert "🧭 <b>Sectors in strength</b>" in text and "sector known for 85% of 20 liquid names" in text
    assert "1. <b>Tech</b> · 3 of 5 leaders (60%) · 100% above 50d (+0) · median 3M +50% · 1 on watchlist · 2 trigger bars" in text
    assert "(+100) · median 3M +20% · 1 trigger bar\n" in text                 # singular for one
    assert "2. <b>Mixed</b>" in text and "(+100)" in text                        # money flowing in
    assert "Weakest: <b>Utilities</b> · 0 of 5 leaders · 0% above 50d (+0)" in text
    assert "<b>Strongest industries</b>" in text and "1. <b>Software</b> <i>(Tech)</i> · 3 of 3 leaders (100%)" in text
    assert "3. <b>Utilities</b>" not in text                                      # only message_sectors = 2 are listed
    prev = MarketState(market="us", as_of=date(2026, 1, 1), active_count=1, watchlist={})
    full = "\n".join(format_alerts(diff_results(prev, res, "us", cfg.alerts), res, cfg,
                                   pd.DataFrame({"symbol": ["X"], "source_symbol": ["X"], "name": ["X Corp"], "exchange": ["NASDAQ"]}), chart_url))
    assert "Sectors in strength" in full
    none = dataclasses.replace(res, sectors=None)
    assert "Sectors in strength" not in "\n".join(format_alerts(diff_results(prev, none, "us", cfg.alerts), none, cfg,
                                                                pd.DataFrame(columns=["symbol", "source_symbol", "name", "exchange"]), chart_url))


def test_dashboard_panel_and_old_snapshots(tmp_path):
    cfg, res = results_with_sectors()
    uni = pd.DataFrame({"symbol": ["X"], "source_symbol": ["X"], "name": ["X Corp"], "exchange": ["NASDAQ"]})
    snap = snapshot(res, cfg, uni, chart_url)
    assert snap["sectors"]["classified"] == 17 and snap["sectors"]["coverage"] == 85.0
    assert [r["name"] for r in snap["sectors"]["sectors"]] == ["Tech", "Mixed", "Utilities"]
    assert snap["sectors"]["industries"][0]["sector"] == "Tech"
    data, reports, out = tmp_path / "data", tmp_path / "reports", tmp_path / "site"
    reports.mkdir()
    write_snapshot(snap, data)
    build_site(out, data, reports)
    page = (out / "index.html").read_text()
    assert "<h2>Sectors in strength (3)</h2>" in page and "Strongest industries" in page and "known for 17 of 20 liquid names (85%)" in page
    old = json.loads((data / "us.json").read_text())
    del old["sectors"]
    (data / "us.json").write_text(json.dumps(old))
    build_site(out, data, reports)
    assert "Sectors in strength" not in (out / "index.html").read_text()
