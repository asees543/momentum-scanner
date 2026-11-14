"""Dashboard snapshot and static site."""

import json
from datetime import datetime, timezone

import pandas as pd

from scanner import scans
from scanner.dashboard import build_site, markdown_to_html, snapshot, write_snapshot
from scanner.data.us_loader import chart_url
from tests.synth import panels_from, textbook, us_cfg

CFG = us_cfg()
UNIVERSE = pd.DataFrame({"symbol": ["X"], "source_symbol": ["X"], "name": ["X Corp - Common Stock"], "exchange": ["NASDAQ"]})


def wide_day():
    b = textbook().append(33.0, 33.3, 30.0, 2_000_000)  # a "too wide" breakout on bar 160
    return scans.run_scans(panels_from({"X": b}), CFG)


def test_snapshot_contents_and_determinism(tmp_path):
    res = scans.run_scans(panels_from({"X": textbook()}), CFG)
    snap = snapshot(res, CFG, UNIVERSE, chart_url)
    assert snap["market"] == "us" and snap["summary"]["watchlist"] == 1
    w = snap["watchlist"][0]
    assert w["symbol"] == "X" and w["name"] == "X Corp" and w["chart"].endswith("NASDAQ:X")
    assert w["trigger"] == round(31.4 * 1.0125, 2) and w["leader"]["3m"] is True
    p1 = write_snapshot(snap, tmp_path).read_text()
    p2 = write_snapshot(snapshot(res, CFG, UNIVERSE, chart_url), tmp_path).read_text()
    assert p1 == p2  # same scan -> byte-identical file (no daily commit noise)
    json.loads(p1)


def test_site_lists_watchlist_breakouts_and_report(tmp_path):
    data, reports, out = tmp_path / "data", tmp_path / "reports", tmp_path / "site"
    write_snapshot(snapshot(wide_day(), CFG, UNIVERSE, chart_url), data)
    reports.mkdir()
    (reports / "backtest-us.md").write_text("# Backtest · US\n\n- **bold** item\n\n| a | b |\n|---|---|\n| 1 | <2> |\n")
    written = build_site(out, data, reports, now=datetime(2026, 11, 14, tzinfo=timezone.utc))
    assert {p.name for p in written} == {"index.html", "backtest-us.html"}
    page = (out / "index.html").read_text()
    assert "<h2>Breakouts</h2>" in page and "stop &gt; ADR" in page and ">X</a>" in page
    assert 'href="backtest-us.html"' in page and "not investment advice" in page
    assert "<script src" not in page and "<link" not in page  # self-contained
    rep = (out / "backtest-us.html").read_text()
    assert "<strong>bold</strong>" in rep and "<td>&lt;2&gt;</td>" in rep


def test_empty_site_builds(tmp_path):
    build_site(tmp_path / "site", tmp_path / "none", tmp_path / "none")
    assert "No data yet." in (tmp_path / "site" / "index.html").read_text()


def test_markdown_subset():
    h = markdown_to_html("## Title\n\nSome *italic* and `code`.\n\n- a\n- b")
    assert "<h2>Title</h2>" in h and "<em>italic</em>" in h and "<code>code</code>" in h and "<li>b</li>" in h
