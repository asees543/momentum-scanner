"""Static dashboard: today's watchlist and breakouts for each market, plus the backtest reports.

Each real daily alert run saves a snapshot, site/data/<market>.json (committed next to
state/<market>.json). `build_site` turns all snapshots and reports/backtest-<market>.md into a
self-contained static site (no external scripts, fonts or styles), which the GitHub Actions workflows
publish to GitHub Pages:

    uv run python -m scanner.dashboard --out _site          # build locally, open _site/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scanner.config import PROJECT_ROOT, MarketConfig
from scanner.notify import jack_rule_text, money, short_name, trigger_bar_flags
from scanner.scans import ScanResults

SITE_DATA = PROJECT_ROOT / "site" / "data"
REPORTS = PROJECT_ROOT / "reports"
MARKET_TITLE = {"us": "US", "india": "India"}


# --- snapshot (written by the daily run) ---------------------------------------------------------


def _f(x, digits=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else round(float(x), digits)


def snapshot(res: ScanResults, cfg: MarketConfig, universe: pd.DataFrame, chart_url) -> dict:
    """JSON-ready view of one scan. Deterministic (no run timestamp), so an unchanged day gives an
    unchanged file and no commit."""
    info = universe.set_index("symbol") if "symbol" in universe.columns and len(universe) else pd.DataFrame()
    windows = list(cfg.leaders.windows)

    def meta(sym: str) -> dict:
        row = info.loc[sym] if sym in info.index else None
        return {
            "symbol": str(row["source_symbol"]) if row is not None and row.get("source_symbol") else sym,
            "name": short_name(row["name"], 40) if row is not None else "",
            "chart": chart_url(sym, row.get("exchange") if row is not None else None,
                               row.get("source_symbol") if row is not None else None),
        }

    def rets(r) -> dict:
        return {w: _f(r[f"ret_{w}"], 4) for w in windows}

    wl, bo, lead = res.watchlist.table, res.breakouts.table, res.leaders
    jk = res.jack
    return {
        "market": cfg.market,
        "title": MARKET_TITLE.get(cfg.market, cfg.market.upper()),
        "currency": cfg.currency,
        "as_of": f"{lead.as_of:%Y-%m-%d}",
        "windows": windows,
        "summary": {
            "symbols_with_data": lead.active_count,
            "liquid": lead.liquid_count,
            "leaders": lead.counts,
            "watchlist": len(wl),
            "qualified_today": res.watchlist.qualified_today,
            "breakouts_ok": res.breakouts.ok_count,
            "breakouts_wide": res.breakouts.rejected_count,
        },
        "breakouts": [
            {**meta(s), "status": r["status"], "close": _f(r["close"]), "trigger": _f(r["trigger"]),
             "stop": _f(r["stop"]), "risk_pct": _f(r["risk_pct"]), "adr_pct": _f(r["adr_pct"]),
             "vol_ratio": _f(r["vol_ratio"]), "setup_age": int(r["setup_age"]), "returns": rets(r),
             "avg_value": money(r["avg_traded_value"], cfg.currency), "flag": bool(r["flag_day_move"])}
            for s, r in bo.iterrows()
        ],
        "watchlist": [
            {**meta(s), "close": _f(r["close"]), "trigger": _f(r["trigger"]), "to_trigger_pct": _f(r["to_trigger_pct"]),
             "stop": _f(r["stop"]), "risk_pct": _f(r["risk_pct"]), "risk_wide": bool(r["risk_too_wide"]),
             "adr_pct": _f(r["adr_pct"]), "vol_ratio": _f(r["vol_ratio"]), "setup_age": int(r["setup_age"]),
             "leader": {w: bool(r[f"led_{w}"]) for w in windows}, "returns": rets(r),
             "avg_value": money(r["avg_traded_value"], cfg.currency), "avg_value_raw": _f(r["avg_traded_value"], 0),
             "flag": bool(r["flag_day_move"])}
            for s, r in wl.iterrows()
        ],
        # Trigger bars on the latest session, best first, at most trigger_bar.max_listed (None where not computed).
        "trigger_bars": None if res.trigger_bars is None else {
            "count": len(res.trigger_bars.table),
            "items": [
                {**meta(s), "ours": r["ours"], "score": int(r["score"]), "ma": r["ma"], "close": _f(r["close"]),
                 "buy_above": _f(r["buy_above"]), "stop": _f(r["stop"]), "risk_pct": _f(r["risk_pct"]),
                 "risk_vs_adr": _f(r["risk_vs_adr"]), "adr_pct": _f(r["adr_pct"]), "vol_ratio": _f(r["vol_ratio"]),
                 "below_high_pct": _f(r["below_high_pct"], 1), "signs": trigger_bar_flags(r), "returns": rets(r),
                 "avg_value": money(r["avg_traded_value"], cfg.currency), "avg_value_raw": _f(r["avg_traded_value"], 0),
                 "flag": bool(r["flag_day_move"])}
                for s, r in res.trigger_bars.table.head(cfg.trigger_bar.max_listed).iterrows()
            ],
        },
        # The leaders of each window, ranked by that window's gain (None where not computed).
        "leader_lists": None if res.window_lists is None else {
            "min_day_value": money(cfg.leaders.min_day_value, cfg.currency),
            "lists": {
                w: [
                    {**meta(s), "close": _f(r["close"]), "returns": rets(r), "day_value": money(r["day_value"], cfg.currency),
                     "day_value_raw": _f(r["day_value"], 0), "flag": bool(r["flag_day_move"])}
                    for s, r in df.iterrows()
                ]
                for w, df in res.window_lists.items()
            },
        },
        # Names on one of our lists that also pass the other trader's screener (None where not computed).
        "jack": None if jk is None else {
            "rule": jack_rule_text(cfg),
            "passing": jk.passing_jack,
            "ours": jk.ours_count,
            "items": [
                {**meta(s), "ours": r["ours"], "close": _f(r["close"]), "trigger": _f(r["trigger"]),
                 "stop": _f(r["stop"]), "adr_pct": _f(r["adr_pct"]), "returns": rets(r),
                 "avg_value": money(r["avg_traded_value"], cfg.currency), "avg_value_raw": _f(r["avg_traded_value"], 0),
                 "flag": bool(r["flag_day_move"])}
                for s, r in jk.table.iterrows()
            ],
        },
    }


def write_snapshot(snap: dict, directory: Path = SITE_DATA) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{snap['market']}.json"
    path.write_text(json.dumps(snap, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return path


# --- tiny markdown -> HTML (only what the backtest report uses) ----------------------------------


def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t


def markdown_to_html(md: str) -> str:
    """Headings, paragraphs, '-' lists, pipe tables, **bold**, *italic*, `code`."""
    out, i = [], 0
    lines = md.splitlines()
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if m := re.match(r"^(#{1,3}) (.*)", line):
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in block]
            head, body = cells[0], [r for r in cells[2:]]
            out.append('<div class="scroll"><table><thead><tr>'
                       + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table></div>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:])
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ul>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- )", lines[i]):
                para.append(lines[i])
                i += 1
            out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


# --- page ---------------------------------------------------------------------------------------

CSS = """
:root { --bg:#f7f7f5; --panel:#ffffff; --text:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc; --accent:#1f6f50;
        --good:#1f7a44; --warn:#a15c00; --chip:#efeee8; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#141413; --panel:#1d1d1b; --text:#ecebe6; --muted:#a3a29b; --line:#33322e; --accent:#5fc095;
  --good:#5fc095; --warn:#e3a14a; --chip:#2a2926; color-scheme: dark; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 20px 16px 48px; }
header h1 { font-size: 22px; margin: 0 0 4px; }
header p, .note, footer { color: var(--muted); font-size: 13px; }
nav.tabs { display:flex; gap:8px; margin: 18px 0 12px; flex-wrap: wrap; }
nav.tabs button { font: inherit; padding: 6px 14px; border-radius: 999px; border:1px solid var(--line);
                  background: var(--panel); color: var(--text); cursor: pointer; }
nav.tabs button[aria-selected="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
section.market[hidden] { display:none; }
.panel { background: var(--panel); border:1px solid var(--line); border-radius: 12px; padding: 14px 16px; margin: 12px 0; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin: 6px 0 0; }
.chip { background: var(--chip); border-radius: 8px; padding: 4px 10px; font-size: 13px; }
.chip b { font-variant-numeric: tabular-nums; }
h2 { font-size: 17px; margin: 0 0 8px; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th:first-child, td:first-child, th.l, td.l { text-align: left; }
th { color: var(--muted); font-weight: 600; cursor: pointer; user-select: none; position: sticky; top: 0; background: var(--panel); }
th[aria-sort="ascending"]::after { content: " ▲"; } th[aria-sort="descending"]::after { content: " ▼"; }
td.name { max-width: 220px; overflow: hidden; text-overflow: ellipsis; text-align: left; color: var(--muted); }
a { color: var(--accent); }
.pos { color: var(--good); } .neg { color: var(--warn); }
.badge { font-size: 11px; padding: 1px 6px; border-radius: 6px; background: var(--chip); }
.badge.ok { color: var(--good); } .badge.wide { color: var(--warn); }
.flag { color: var(--warn); }
.tall { max-height: 460px; overflow-y: auto; }
.report h1 { font-size: 20px; } .report h2 { margin-top: 24px; } .report td, .report th { cursor: default; }
"""

JS = """
document.querySelectorAll('nav.tabs button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('nav.tabs button').forEach(x => x.setAttribute('aria-selected', x === b));
  document.querySelectorAll('section.market').forEach(s => s.hidden = s.id !== b.dataset.target);
  try { localStorage.setItem('tab', b.dataset.target); } catch (e) {}
}));
try { const t = localStorage.getItem('tab'); const b = t && document.querySelector(`nav.tabs button[data-target="${t}"]`); if (b) b.click(); } catch (e) {}
document.querySelectorAll('table.sortable').forEach(table => {
  table.querySelectorAll('th').forEach((th, i) => th.addEventListener('click', () => {
    const asc = th.getAttribute('aria-sort') !== 'ascending';
    table.querySelectorAll('th').forEach(x => x.removeAttribute('aria-sort'));
    th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
    const rows = [...table.tBodies[0].rows];
    const key = r => { const v = r.cells[i].dataset.v ?? r.cells[i].textContent; const n = parseFloat(v); return isNaN(n) ? v : n; };
    rows.sort((a, b) => { const x = key(a), y = key(b); return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1); });
    rows.forEach(r => table.tBodies[0].appendChild(r));
  }));
});
"""


def _num(x, fmt="{:.2f}"):
    return "–" if x is None else fmt.format(x)


def _ret(x):
    if x is None:
        return '<td data-v="">–</td>'
    cls = "pos" if x >= 0 else "neg"
    return f'<td data-v="{x}" class="{cls}">{x * 100:+.0f}%</td>'


def _sym_cell(r: dict) -> str:
    flag = ' <span class="flag" title="One-day move beyond the flag threshold in the last 6 months: check the chart">⚠</span>' if r["flag"] else ""
    return f'<td class="l" data-v="{html.escape(r["symbol"])}"><a href="{html.escape(r["chart"])}" target="_blank" rel="noopener">{html.escape(r["symbol"])}</a>{flag}</td>'


LEFT_COLUMNS = {"Symbol", "Name", "Status", "Leader"}


def _header(head: list[str]) -> str:
    return "".join(f'<th class="l">{h}</th>' if h in LEFT_COLUMNS else f"<th>{h}</th>" for h in head)


def _breakouts_table(snap: dict) -> str:
    rows = snap["breakouts"]
    if not rows:
        return '<p class="note">No breakouts on this session.</p>'
    ws = snap["windows"]
    head = ["Symbol", "Name", "Status", "Close", "Trigger", "Stop", "Risk %", "ADR %", "Vol ×", "Setup"] + [w.upper() for w in ws] + ["Avg value"]
    body = []
    for r in sorted(rows, key=lambda r: (r["status"] != "ok", -(r["vol_ratio"] or 0))):
        badge = '<span class="badge ok">ok</span>' if r["status"] == "ok" else '<span class="badge wide">stop &gt; ADR</span>'
        body.append(
            "<tr>" + _sym_cell(r) + f'<td class="name">{html.escape(r["name"])}</td>'
            + f'<td class="l" data-v="{r["status"]}">{badge}</td>'
            + "".join(f'<td data-v="{r[k]}">{_num(r[k])}</td>' for k in ("close", "trigger", "stop"))
            + f'<td data-v="{r["risk_pct"]}">{_num(r["risk_pct"], "{:.1f}")}</td><td data-v="{r["adr_pct"]}">{_num(r["adr_pct"], "{:.1f}")}</td>'
            + f'<td data-v="{r["vol_ratio"]}">{_num(r["vol_ratio"], "{:.1f}")}</td><td data-v="{r["setup_age"]}">{r["setup_age"]}d</td>'
            + "".join(_ret(r["returns"].get(w)) for w in ws) + f'<td>{html.escape(r["avg_value"])}</td></tr>'
        )
    return ('<div class="scroll"><table class="sortable"><thead><tr>' + _header(head)
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def _watchlist_table(snap: dict) -> str:
    rows = snap["watchlist"]
    if not rows:
        return '<p class="note">The watchlist is empty.</p>'
    ws = snap["windows"]
    head = ["Symbol", "Name", "Close", "Trigger", "To trig", "Stop", "Risk %", "ADR %", "Vol/avg", "Leader", "Age"] + [w.upper() for w in ws] + ["Avg value"]
    body = []
    for r in rows:
        leader = " ".join(w.upper() for w in ws if r["leader"].get(w)) or "–"
        risk = _num(r["risk_pct"], "{:.1f}") + (' <span class="badge wide">wide</span>' if r["risk_wide"] else "")
        body.append(
            "<tr>" + _sym_cell(r) + f'<td class="name">{html.escape(r["name"])}</td>'
            + f'<td data-v="{r["close"]}">{_num(r["close"])}</td><td data-v="{r["trigger"]}"><b>{_num(r["trigger"])}</b></td>'
            + f'<td data-v="{r["to_trigger_pct"]}">{_num(r["to_trigger_pct"], "{:+.1f}%")}</td>'
            + f'<td data-v="{r["stop"]}">{_num(r["stop"])}</td><td data-v="{r["risk_pct"]}">{risk}</td>'
            + f'<td data-v="{r["adr_pct"]}">{_num(r["adr_pct"], "{:.1f}")}</td><td data-v="{r["vol_ratio"]}">{_num(r["vol_ratio"])}</td>'
            + f'<td class="l">{leader}</td><td data-v="{r["setup_age"]}">{"today" if r["setup_age"] == 0 else str(r["setup_age"]) + "d"}</td>'
            + "".join(_ret(r["returns"].get(w)) for w in ws)
            + f'<td data-v="{r.get("avg_value_raw") or 0}">{html.escape(r["avg_value"])}</td></tr>'
        )
    return ('<div class="scroll"><table class="sortable"><thead><tr>' + _header(head)
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def _jack_panel(snap: dict) -> str:
    jk = snap.get("jack")
    if jk is None:  # a snapshot written before this section existed
        return ""
    ws = snap["windows"]
    if jk["items"]:
        head = ["Symbol", "Name", "Ours", "Close", "Trigger", "Stop", "ADR %"] + [w.upper() for w in ws] + ["Avg value"]
        body = []
        for r in jk["items"]:
            badge = f'<span class="badge {"ok" if r["ours"] != "leader" else ""}">{r["ours"]}</span>'
            body.append(
                "<tr>" + _sym_cell(r) + f'<td class="name">{html.escape(r["name"])}</td>'
                + f'<td class="l" data-v="{r["ours"]}">{badge}</td>'
                + f'<td data-v="{r["close"]}">{_num(r["close"])}</td>'
                + f'<td data-v="{r["trigger"] or 0}">{_num(r["trigger"])}</td><td data-v="{r["stop"] or 0}">{_num(r["stop"])}</td>'
                + f'<td data-v="{r["adr_pct"]}">{_num(r["adr_pct"], "{:.1f}")}</td>'
                + "".join(_ret(r["returns"].get(w)) for w in ws)
                + f'<td data-v="{r.get("avg_value_raw") or 0}">{html.escape(r["avg_value"])}</td></tr>'
            )
        table = ('<div class="scroll"><table class="sortable"><thead><tr>' + _header(head)
                 + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")
    else:
        table = f'<p class="note">None on this session. {jk["passing"]} names pass the other screener alone.</p>'
    return (
        f'<div class="panel"><h2>On both screens ({len(jk["items"])})</h2>{table}'
        f'<p class="note">On one of our lists (breakout, watchlist or today\'s leaders) <b>and</b> passing Jack '
        f"Corsellis's screener: {html.escape(jk['rule'])}. {jk['passing']} names pass that screener alone; {jk['ours']} are on our "
        "lists. Filters read from a screenshot of their screener; the table shows the strongest list a name is on. "
        "The average value is over the screener's 50 days, not our 20.</p></div>"
    )


def _trigger_bar_panel(snap: dict) -> str:
    tb = snap.get("trigger_bars")
    if tb is None:  # a snapshot written before this section existed
        return ""
    ws = snap["windows"]
    if tb["items"]:
        head = ["Symbol", "Name", "Ours", "Score", "On", "Close", "Buy above", "Stop", "Risk %", "Risk / ADR", "ADR %",
                "Vol ×", "Below 52w high"] + [w.upper() for w in ws] + ["Avg value"]
        body = []
        for r in tb["items"]:
            badge = f'<span class="badge {"ok" if r["ours"] else ""}">{r["ours"] or "–"}</span>'
            body.append(
                "<tr>" + _sym_cell(r) + f'<td class="name">{html.escape(r["name"])}</td>'
                + f'<td class="l" data-v="{r["ours"]}">{badge}</td>'
                + f'<td data-v="{r["score"]}" title="{html.escape(", ".join(r["signs"]) or "none")}">{r["score"]}/6</td>'
                + f'<td class="l">{html.escape(r["ma"])}</td>'
                + f'<td data-v="{r["close"]}">{_num(r["close"])}</td><td data-v="{r["buy_above"]}"><b>{_num(r["buy_above"])}</b></td>'
                + f'<td data-v="{r["stop"]}">{_num(r["stop"])}</td>'
                + f'<td data-v="{r["risk_pct"]}">{_num(r["risk_pct"], "{:.1f}")}</td>'
                + f'<td data-v="{r["risk_vs_adr"]}">{_num(r["risk_vs_adr"], "{:.2f}")}</td>'
                + f'<td data-v="{r["adr_pct"]}">{_num(r["adr_pct"], "{:.1f}")}</td>'
                + f'<td data-v="{r["vol_ratio"]}">{_num(r["vol_ratio"], "{:.1f}")}</td>'
                + f'<td data-v="{r["below_high_pct"]}">{_num(r["below_high_pct"], "{:.0f}%")}</td>'
                + "".join(_ret(r["returns"].get(w)) for w in ws)
                + f'<td data-v="{r.get("avg_value_raw") or 0}">{html.escape(r["avg_value"])}</td></tr>'
            )
        table = ('<div class="scroll tall"><table class="sortable"><thead><tr>' + _header(head)
                 + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")
    else:
        table = '<p class="note">None on this session.</p>'
    more = f" The best {len(tb['items'])} of {tb['count']} are listed." if tb["count"] > len(tb["items"]) else ""
    return (
        f'<div class="panel"><h2>Trigger bars ({tb["count"]})</h2>{table}'
        '<p class="note">A tight daily bar sitting on a key moving average (10 / 21-day EMA, 50 / 200-day SMA) in an uptrend, '
        "from the Blueprint Trading Course: be ready for a break above its high (<b>Buy above</b>) on the next session, with "
        "the stop just under its low. Score = how many of the course's six signs of a good one it has (inside bar, tightest of "
        "recent bars, low volume, strong close, range under half an ADR, two or more averages nearby); hover a score to "
        "see which. Names already on one of our lists are first." + more + "</p></div>"
    )


def _leader_lists_panels(snap: dict) -> str:
    ll = snap.get("leader_lists")
    if ll is None:  # a snapshot written before these lists existed
        return ""
    ws = snap["windows"]
    words = {"1m": "1-month", "3m": "3-month", "6m": "6-month"}
    out = []
    for w in ws:
        rows = ll["lists"].get(w, [])
        if rows:
            head = ["#", "Symbol", "Name", "Close"] + [x.upper() for x in ws] + ["Last-session value"]
            body = []
            for i, r in enumerate(rows, start=1):
                body.append(
                    f'<tr><td data-v="{i}">{i}</td>' + _sym_cell(r) + f'<td class="name">{html.escape(r["name"])}</td>'
                    + f'<td data-v="{r["close"]}">{_num(r["close"])}</td>'
                    + "".join(_ret(r["returns"].get(x)) for x in ws)
                    + f'<td data-v="{r["day_value_raw"] or 0}">{html.escape(r["day_value"])}</td></tr>'
                )
            table = ('<div class="scroll tall"><table class="sortable"><thead><tr>' + _header(head)
                     + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")
        else:
            table = '<p class="note">None on this session.</p>'
        out.append(f'<div class="panel"><h2>{w.upper()} leaders ({len(rows)})</h2>{table}'
                   f'<p class="note">Biggest {words.get(w, w)} gain first (click a column to re-sort). Passes the {w.upper()} '
                   f"leader rule and traded more than {html.escape(ll['min_day_value'])} in the last session "
                   "(close × volume).</p></div>")
    return "".join(out)


def _market_section(snap: dict, has_report: bool, first: bool) -> str:
    s = snap["summary"]
    day = datetime.strptime(snap["as_of"], "%Y-%m-%d").strftime("%a %d %b %Y")
    leaders = " / ".join(f"{w.upper()} {s['leaders'].get(w, 0)}" for w in snap["windows"])
    chips = [
        f"Session <b>{day}</b>",
        f"Watchlist <b>{s['watchlist']}</b> ({s['qualified_today']} qualified that day)",
        f"Breakouts <b>{s['breakouts_ok']}</b> ok · <b>{s['breakouts_wide']}</b> stop &gt; ADR",
        f"Leaders {leaders}",
        f"Liquid <b>{s['liquid']:,}</b> of {s['symbols_with_data']:,}",
    ]
    report = (f'<p class="note"><a href="backtest-{snap["market"]}.html">Backtest report</a>: how these signals did '
              "historically, with the biases spelled out.</p>") if has_report else ""
    return (
        f'<section class="market" id="{snap["market"]}"{"" if first else " hidden"}>'
        f'<div class="panel"><div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>" + report + "</div>"
        + _jack_panel(snap)
        + _trigger_bar_panel(snap)
        + f'<div class="panel"><h2>Breakouts</h2>{_breakouts_table(snap)}'
        + '<p class="note">Close above the trigger on at least 1.5× average volume, from a name on the previous '
          "session's watchlist. Stop = the breakout day's low. “stop &gt; ADR” = the stop is more than one average "
          "daily range away (SPEC would reject the trade).</p></div>"
        + f'<div class="panel"><h2>Watchlist</h2>{_watchlist_table(snap)}'
        + '<p class="note">Names that can break out next session, closest to their trigger first (click a column to '
          "sort). Trigger = highest high of the last 10 sessions on the setup day; stop = latest low; ⚠ = a one-day "
          "move beyond the flag threshold in the last 6 months.</p></div>"
        + _leader_lists_panels(snap) + "</section>"
    )


def _page(title: str, body: str, generated: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><main>{body}
<footer><p>Information only, not investment advice. Paper trade before using real money. Prices: Yahoo Finance,
NSE end-of-day files. Built {generated}.</p></footer></main><script>{JS}</script></body></html>
"""


def build_site(out: Path, data_dir: Path = SITE_DATA, reports_dir: Path = REPORTS, now: datetime | None = None) -> list[Path]:
    generated = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    out.mkdir(parents=True, exist_ok=True)
    order = {"us": 0, "india": 1}
    snaps = sorted((json.loads(p.read_text()) for p in Path(data_dir).glob("*.json")),
                   key=lambda s: order.get(s["market"], 9))
    written = []
    reports = {p.stem.removeprefix("backtest-"): p for p in Path(reports_dir).glob("backtest-*.md")}
    tabs = "".join(
        f'<button data-target="{s["market"]}" aria-selected="{str(i == 0).lower()}">{html.escape(s["title"])}</button>'
        for i, s in enumerate(snaps)
    )
    sections = "".join(_market_section(s, s["market"] in reports, i == 0) for i, s in enumerate(snaps))
    body = ('<header><h1>Momentum scanner</h1><p>Qullamaggie-style momentum setups, updated after each market '
            "close.</p></header>"
            + (f'<nav class="tabs">{tabs}</nav>{sections}' if snaps else '<p class="note">No data yet.</p>'))
    (out / "index.html").write_text(_page("Momentum scanner", body, generated))
    written.append(out / "index.html")
    for market, path in reports.items():
        rep = (f'<header><p><a href="index.html">← Dashboard</a></p></header><div class="panel report">'
               f"{markdown_to_html(path.read_text())}</div>")
        target = out / f"backtest-{market}.html"
        target.write_text(_page(f"Backtest · {MARKET_TITLE.get(market, market)}", rep, generated))
        written.append(target)
    (out / ".nojekyll").write_text("")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scanner.dashboard", description="Build the static dashboard site")
    ap.add_argument("--out", default="_site", help="output folder (default: _site)")
    args = ap.parse_args(argv)
    for p in build_site(Path(args.out)):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
