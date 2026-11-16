"""Study set: past trigger touches from the backtest, as one self-contained page of charts.

Qullamaggie's own advice for learning his setups is to look through a large number of past examples.
This module picks examples from the same point-in-time events the backtest uses (Model A: a name from
the previous evening's watchlist trades up to its trigger) and writes study/<market>.html:

    uv run python -m scanner.study --market us      # uses the cached prices, like the backtest

Decks (a card can be in several):
  random   a random sample of ALL trigger touches: the realistic mix, the one to practise on
  winner   the biggest winners by trade return, one card per symbol per move
  failed   random triggers that did not confirm by the close (the common case)
  lost     random confirmed breakouts (ok or too wide) that still lost money

Practice mode shows a chart only up to the evening before the trigger day, which is what the watchlist
alert showed, and reveals the rest after a take / skip choice. The page keeps score of the trades taken
against all the cards seen: on the random deck that is "my picks against taking everything".

Each card's result is the backtest's Model A trade with the headline exit, after costs. R divides it
by the distance from the entry to the previous day's low, because the trigger day's own low is not
known when a buy-stop fills. Entries come from daily bars; Qullamaggie's real entry (an opening-range
high with the stop at the low of the day so far) needs intraday data and would give other R values.

The page embeds price history, so it is written to study/ (gitignored) and is never published.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scanner import backtest as bt
from scanner import indicators as ind
from scanner.config import PROJECT_ROOT, MarketConfig, StudyConfig, load_config
from scanner.dashboard import CSS, MARKET_TITLE
from scanner.data.base import Panels
from scanner.notify import short_name

DECKS = {
    "random": "Random mix",
    "winner": "Biggest winners",
    "failed": "Failed triggers",
    "lost": "Confirmed, still lost",
}
CHART_SMAS = (10, 20, 50)  # chart overlays only: the averages the setup rules look at
EXIT_MARGIN = 5  # sessions charted after the trade's last sale
MAX_BARS_AFTER = 250  # a trade that runs longer than this is cut off on the chart


def _num(x, digits: int = 5):
    """JSON-ready number rounded to `digits` significant figures; None for a missing value."""
    if x is None or pd.isna(x) or not np.isfinite(x):
        return None
    return float(f"{float(x):.{digits}g}")


def _pct(x, digits: int = 2):
    """A fraction as a percentage rounded to `digits` decimals; None for a missing value."""
    if x is None or pd.isna(x) or not np.isfinite(x):
        return None
    return round(100 * float(x), digits)


def _yyyymmdd(ts: pd.Timestamp) -> int:
    return ts.year * 10000 + ts.month * 100 + ts.day


# --- which trades ---------------------------------------------------------------------------------


def study_trades(panels: Panels, cfg: MarketConfig, warmup: int | None = None) -> tuple[pd.DataFrame, bt.ExitRule]:
    """Every Model A trade of the backtest under the headline exit, plus the event facts a card shows
    (trigger, previous day's low, ADR% the evening before, setup age). Same code path as the backtest
    report, so a card's result is the number behind "A · all" there."""
    b = cfg.backtest
    rule = bt.ExitRule(b.headline_partial_fraction, b.headline_partial_after_days, b.headline_trail_sma)
    events = bt.trigger_events(bt.history_frames(panels, cfg, warmup=warmup), cfg)
    trades = bt.build_trades(events, panels, "A")
    sim = bt.simulate(trades, panels, rule, b.cost_pct_per_side + b.slippage_pct_per_side, b.min_risk_pct)
    # Model A takes every event, in order, so the event's own columns line up row for row.
    if len(sim) != len(events):
        raise RuntimeError("Model A trades and trigger events are out of step")
    extra = events[["trigger", "prev_low", "adr_prev", "setup_age"]].set_axis(sim.index)
    return pd.concat([sim, extra], axis=1), rule


def pick_cards(sim: pd.DataFrame, s: StudyConfig, n_rows: int) -> pd.DataFrame:
    """Choose the cards: the chosen trades with a `tags` column (list of deck names), in a shuffled
    order. Everything is seeded, so the same data gives the same cards in the same order.

    Usable trades: not ⚠-flagged (possible unadjusted split), closed, and with at least
    `min_bars_after` sessions of data after the entry so the chart can show what happened.
      winner  best trade return first; a symbol's other touches within `winner_spacing` sessions of a
              card already chosen are the same move and are skipped
      failed  random sample of triggers that did not confirm by the close
      lost    random sample of confirmed breakouts (ok / too wide) with a negative return
      random  random sample of all usable trades, whatever happened
    """
    if sim.empty:
        return sim.assign(tags=pd.Series(dtype=object))
    usable = (~sim["flag_day_move"].astype(bool) & ~sim["open"].astype(bool) & sim["ret"].notna()
              & (sim["entry_row"] + s.min_bars_after <= n_rows - 1))
    pool = sim[usable]
    rng = np.random.default_rng(s.seed)
    tags: dict[int, list[str]] = {}

    seen: dict[str, list[int]] = {}
    n_winners = 0
    ranked = pool[pool["ret"] > 0].sort_values(["ret", "entry_row", "symbol"], ascending=[False, True, True])
    for i, sym, row in zip(ranked.index, ranked["symbol"], ranked["entry_row"]):
        if n_winners >= s.winners:
            break
        if any(abs(row - r) <= s.winner_spacing for r in seen.get(sym, [])):
            continue
        seen.setdefault(sym, []).append(row)
        tags.setdefault(i, []).append("winner")
        n_winners += 1

    def sample(mask: pd.Series, n: int, tag: str) -> None:
        idx = pool.index[mask.to_numpy()].to_numpy()
        if not len(idx) or not n:
            return
        for i in sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist()):
            tags.setdefault(i, []).append(tag)

    sample(pool["group"] == "failed", s.failures, "failed")
    sample((pool["group"] != "failed") & (pool["ret"] < 0), s.confirmed_losers, "lost")
    sample(pd.Series(True, index=pool.index), s.random, "random")

    chosen = pool.loc[sorted(tags)].copy()
    chosen["tags"] = [tags[i] for i in chosen.index]
    return chosen.iloc[rng.permutation(len(chosen))]


# --- one card -------------------------------------------------------------------------------------


def card(tr, card_id: int, panels: Panels, arr: dict[str, np.ndarray], smas: dict[int, np.ndarray],
         cfg: MarketConfig, meta: dict) -> dict:
    """Chart data and facts for one trade (`tr` is a row of `pick_cards`).

    The arrays cover sessions start..end of the price history:
      start = entry_row - bars_before (not before the first session)
      end   = the later of entry_row + bars_after and the last sale + EXIT_MARGIN (not past the data)
    `cut` = entry_row - start is the index of the trigger day, so arrays[:cut] is exactly what was
    known the evening before; the practice view draws only that part until the choice is made.
    """
    s, n = cfg.study, len(panels.dates)
    e, j = int(tr.entry_row), int(tr.col)
    start = max(0, e - s.bars_before)
    end = min(n - 1, max(e + s.bars_after, min(int(tr.exit_row) + EXIT_MARGIN, e + MAX_BARS_AFTER)))
    sl = slice(start, end + 1)
    close = arr["close"]
    vol = arr["volume"][sl, j]
    vmax = np.nanmax(vol) if np.isfinite(vol).any() else np.nan
    prev_close = close[e - 1, j] if e >= 1 else np.nan

    rets = {}
    for w, win in cfg.leaders.windows.items():
        k = e - 1 - win.lookback
        # Leader return as it stood the evening before: close_(t-1) / close_(t-1-lookback) - 1
        rets[w] = _pct(prev_close / close[k, j] - 1) if k >= 0 else None
    fwd = {}
    for d in cfg.backtest.forward_days:
        # Forward return as in the backtest report (no costs): close_(t+d) / entry - 1
        fwd[str(d)] = _pct(close[e + d, j] / tr.entry - 1) if e + d <= n - 1 else None

    exit_i = int(tr.exit_row) - start
    return {
        "id": card_id,
        "tags": list(tr.tags),
        "symbol": meta["symbol"],
        "name": meta["name"],
        "date": _yyyymmdd(panels.dates[e]),
        "cut": e - start,
        "exit": exit_i if exit_i <= end - start else None,
        "d": [_yyyymmdd(t) for t in panels.dates[sl]],
        "o": [_num(x) for x in arr["open"][sl, j]],
        "h": [_num(x) for x in arr["high"][sl, j]],
        "l": [_num(x) for x in arr["low"][sl, j]],
        "c": [_num(x) for x in close[sl, j]],
        # Volume only sets bar heights, so it is stored as 0..1000 of the window's largest bar.
        "v": [None if not np.isfinite(x) or not vmax > 0 else int(round(1000 * x / vmax)) for x in vol],
        **{f"s{k}": [_num(x) for x in smas[k][sl, j]] for k in CHART_SMAS},
        "trigger": _num(tr.trigger),
        "stop": _num(tr.prev_low),  # the watchlist stop: the previous day's low
        "entry": _num(tr.entry),
        "status": tr.group,
        "ret": _pct(tr.ret),
        "r": _num(tr.r_multiple, 3),
        "days": int(tr.days),
        "adr": _num(tr.adr_prev, 3),
        "risk": _num(tr.risk_pct, 3),
        # How far the trigger sat above the last close: trigger / close_(t-1) - 1
        "gap": _pct(tr.trigger / prev_close - 1),
        "vol_ratio": _num(tr.vol_ratio, 3),
        "age": int(tr.setup_age),
        "rets": rets,
        "fwd": fwd,
    }


def build_study(panels: Panels, cfg: MarketConfig, universe: pd.DataFrame, warmup: int | None = None) -> dict:
    """Everything the page needs, JSON-ready and deterministic (no build time)."""
    sim, rule = study_trades(panels, cfg, warmup)
    chosen = pick_cards(sim, cfg.study, len(panels.dates))
    arr = {f: getattr(panels, f).to_numpy() for f in ("open", "high", "low", "close", "volume")}
    smas = {k: ind.sma(panels.close, k).to_numpy() for k in CHART_SMAS}
    info = universe.set_index("symbol") if "symbol" in universe.columns and len(universe) else pd.DataFrame()

    def meta(sym: str) -> dict:
        row = info.loc[sym] if sym in info.index else None
        return {
            "symbol": str(row["source_symbol"]) if row is not None and row.get("source_symbol") else sym,
            "name": short_name(row["name"], 40) if row is not None and row.get("name") else "",
        }

    cards = [card(tr, i, panels, arr, smas, cfg, meta(tr.symbol))
             for i, tr in enumerate(chosen.itertuples(index=False))]
    st = bt.trade_stats(sim)
    return {
        "market": cfg.market,
        "title": MARKET_TITLE.get(cfg.market, cfg.market.upper()),
        "as_of": f"{panels.dates[-1]:%Y-%m-%d}" if len(panels.dates) else "",
        "first": f"{sim['entry_date'].min():%Y-%m-%d}" if len(sim) else "",
        "exit": rule.label,
        "windows": list(cfg.leaders.windows),
        "forward_days": list(cfg.backtest.forward_days),
        "bars_after": cfg.study.bars_after,
        "decks": [{"tag": t, "label": label, "n": sum(t in c["tags"] for c in cards)} for t, label in DECKS.items()],
        # The backtest's "A · all" line: what taking every trigger did.
        "all": ({"n": st["trades"], "win": _pct(st["win_rate"], 1), "ret": _pct(st["mean_ret"]),
                 "r": _num(st["mean_r"], 3)} if st["trades"] else None),
        "cards": cards,
    }


# --- page -----------------------------------------------------------------------------------------

STUDY_CSS = """
:root { --up:#1f7a44; --down:#c0392b; --sma10:#2b6cb0; --sma20:#b7791f; --sma50:#805ad5; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --up:#5fc095; --down:#e5746a; --sma10:#63b3ed; --sma20:#ecc94b; --sma50:#b794f4; } }
[hidden] { display: none !important; }
select, button { font: inherit; color: var(--text); }
select { padding: 5px 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); }
button { padding: 6px 14px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); cursor: pointer; }
button:disabled { opacity: .45; cursor: default; }
button.primary { background: var(--accent); border-color: var(--accent); color: var(--panel); font-weight: 600; }
nav.tabs button[aria-selected="true"] { color: var(--panel); }
.controls { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.controls .spacer { flex: 1; }
.controls label { color: var(--muted); font-size: 13px; }
canvas.chart { width: 100%; height: 300px; display: block; }
canvas.chart.big { height: 460px; }
@media (max-width: 640px) { canvas.chart.big { height: 340px; } canvas.chart { height: 240px; } }
#grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr)); gap: 12px; }
#grid .panel { margin: 0; }
.cardhead { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--muted); margin: 8px 0 2px; }
.legend i { display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 5px; border-radius: 2px; }
.muted { color: var(--muted); }
.neg { color: var(--down); }
.ask { margin: 14px 0 0; font-weight: 600; }
.actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
.outcome { border-top: 1px solid var(--line); margin-top: 12px; padding-top: 10px; }
.outcome p { margin: 0 0 6px; }
.pager { display: flex; gap: 12px; align-items: center; justify-content: center; margin: 14px 0; color: var(--muted); font-size: 13px; }
#score th { cursor: default; position: static; }
#score tr.ref td { color: var(--muted); }
details summary { cursor: pointer; font-weight: 600; }
details li { margin: 4px 0; }
"""

STUDY_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const CARDS = D.cards, BY_ID = new Map(CARDS.map(k => [k.id, k]));
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, ch => '&#' + ch.charCodeAt(0) + ';');
const sign = x => x > 0 ? '+' : x < 0 ? '−' : '';
const fmtPct = (x, d = 1) => x == null ? '–' : sign(x) + Math.abs(x).toFixed(d) + '%';
const fmtR = x => x == null ? '–' : sign(x) + Math.abs(x).toFixed(2) + 'R';
const fmtPrice = x => x == null ? '–' : x >= 1000 ? Math.round(x).toLocaleString('en-US') : x >= 100 ? x.toFixed(1) : x >= 1 ? x.toFixed(2) : x.toFixed(3);
const fmtDate = n => { const s = String(n); return s.slice(0, 4) + '-' + s.slice(4, 6) + '-' + s.slice(6, 8); };
const cls = x => x > 0 ? 'pos' : x < 0 ? 'neg' : '';
const plural = (n, w) => n + ' ' + w + (n === 1 ? '' : 's');
const OUTCOME = {ok: 'Confirmed breakout', 'too wide': 'Confirmed, stop too wide', failed: 'Failed to confirm'};
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// Answers live in this browser only; the key changes when the cards do.
const KEY = 'study:' + D.market + ':' + D.as_of + ':' + CARDS.length;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} };

let mode = 'practice', deck = (D.decks.find(d => d.n > 0) || {tag: 'all'}).tag, page = 0, current = null, revealed = false;
const inDeck = k => deck === 'all' || k.tags.includes(deck);
const deckCards = () => CARDS.filter(inDeck);
const nextCard = () => CARDS.find(k => inDeck(k) && !(k.id in answers)) || null;

function niceStep(range, target) {
  const raw = range / target, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}

// Scale marks: round steps for a narrow range; on a wide (log) range, 1-1.5-2-3-4-5-6-8 per decade,
// dropping any mark closer than 24px to the one below it.
function priceTicks(lo, hi, y) {
  const out = [];
  if (hi / lo < 1.8) {
    const step = niceStep(hi - lo, 5);
    for (let p = Math.ceil(lo / step) * step; p <= hi; p += step) out.push(p);
    return out;
  }
  let last = Infinity;
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 1.5, 2, 3, 4, 5, 6, 8]) {
      const p = m * Math.pow(10, e);
      if (p < lo || p > hi) continue;
      const yy = y(p);
      if (last - yy >= 24) { out.push(p); last = yy; }
    }
  return out;
}

// Draws bars 0..cut-1 when `reveal` is false: nothing from the trigger day on may influence the picture.
// That includes the scales (price range and volume heights use the visible bars only) and the width
// of the empty space on the right (a fixed number of sessions, not the length of this trade).
function draw(cv, k, reveal) {
  const W = cv.clientWidth, H = cv.clientHeight;
  if (!W || !H) return;
  const css = getComputedStyle(document.documentElement), v = name => css.getPropertyValue(name).trim();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const n = k.c.length, shown = reveal ? n : k.cut, total = reveal ? n : k.cut + D.bars_after + 1;
  const padR = 58, padT = 16, padB = 20, volH = Math.round(H * 0.16), gap = 8;
  const pw = W - padR, ph = H - padT - padB - volH - gap, bw = pw / total;
  const x = i => (i + 0.5) * bw;
  let lo = Math.min(k.stop == null ? Infinity : k.stop, k.trigger), hi = k.trigger, vmax = 0;
  for (let i = 0; i < shown; i++) {
    if (k.l[i] != null && k.h[i] != null) { lo = Math.min(lo, k.l[i]); hi = Math.max(hi, k.h[i]); }
    if (k.v[i] != null) vmax = Math.max(vmax, k.v[i]);
  }
  // Log price scale: equal percentage moves take equal height, so a tight base still reads as tight
  // next to a move that multiplied the price. y(p) = top + height x (1 - (ln p - ln lo) / (ln hi - ln lo))
  const pad = Math.max(Math.pow(hi / lo, 0.05), 1.02); lo /= pad; hi *= pad;
  const lnLo = Math.log(lo), span = Math.log(hi) - lnLo;
  const y = p => padT + ph * (1 - (Math.log(p) - lnLo) / span);
  g.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  g.lineWidth = 1;

  // price grid and scale
  g.textAlign = 'left'; g.textBaseline = 'middle';
  for (const p of priceTicks(lo, hi, y)) {
    const yy = Math.round(y(p)) + 0.5;
    g.strokeStyle = v('--line'); g.beginPath(); g.moveTo(0, yy); g.lineTo(pw, yy); g.stroke();
    g.fillStyle = v('--muted'); g.fillText(fmtPrice(p), pw + 6, yy);
  }
  // month marks (the year is shown only once the card is revealed)
  g.textBaseline = 'alphabetic';
  let labelEnd = -100, labels = 0;
  for (let i = 1; i < shown; i++) {
    const a = Math.floor(k.d[i - 1] / 100) % 100, b = Math.floor(k.d[i] / 100) % 100;
    if (a === b) continue;
    const xx = Math.round(i * bw) + 0.5;
    g.strokeStyle = v('--line'); g.beginPath(); g.moveTo(xx, padT); g.lineTo(xx, H - padB); g.stroke();
    const text = MONTHS[b - 1] + (reveal && (b === 1 || labels === 0) ? ' ' + Math.floor(k.d[i] / 10000) : '');
    const w = g.measureText(text).width;
    if (xx + 3 > labelEnd + 8 && xx + 3 + w < pw) {   // skip a label that would run into the previous one
      g.fillStyle = v('--muted'); g.fillText(text, xx + 3, H - 6);
      labelEnd = xx + 3 + w; labels++;
    }
  }
  // volume bars
  const up = v('--up'), down = v('--down'), body = Math.max(1, Math.min(bw * 0.7, 9)), vy0 = H - padB;
  g.globalAlpha = 0.45;
  for (let i = 0; i < shown; i++) {
    if (k.v[i] == null || k.c[i] == null || !vmax) continue;
    const hh = volH * k.v[i] / vmax;
    g.fillStyle = k.c[i] >= (k.o[i] == null ? k.c[i] : k.o[i]) ? up : down;
    g.fillRect(x(i) - body / 2, vy0 - hh, body, hh);
  }
  g.globalAlpha = 1;

  g.save(); g.beginPath(); g.rect(0, padT, pw, ph); g.clip();
  // candles
  for (let i = 0; i < shown; i++) {
    if (k.c[i] == null || k.h[i] == null || k.l[i] == null) continue;
    const op = k.o[i] == null ? k.c[i] : k.o[i], yo = y(op), yc = y(k.c[i]), xc = Math.round(x(i)) + 0.5;
    g.strokeStyle = g.fillStyle = k.c[i] >= op ? up : down;
    g.beginPath(); g.moveTo(xc, y(k.h[i])); g.lineTo(xc, y(k.l[i])); g.stroke();
    g.fillRect(x(i) - body / 2, Math.min(yo, yc), body, Math.max(1, Math.abs(yc - yo)));
  }
  // moving averages
  const line = (arr, color) => {
    g.strokeStyle = color; g.lineWidth = 1.3; g.beginPath();
    let pen = false;
    for (let i = 0; i < shown; i++) {
      if (arr[i] == null) { pen = false; continue; }
      if (pen) g.lineTo(x(i), y(arr[i])); else g.moveTo(x(i), y(arr[i]));
      pen = true;
    }
    g.stroke(); g.lineWidth = 1;
  };
  line(k.s50, v('--sma50')); line(k.s20, v('--sma20')); line(k.s10, v('--sma10'));
  // trigger and stop levels
  const from = x(Math.max(0, k.cut - 15));
  const level = (p, color) => {
    const yy = Math.round(y(p)) + 0.5;
    g.strokeStyle = color; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(from, yy); g.lineTo(pw, yy); g.stroke(); g.setLineDash([]);
  };
  level(k.trigger, v('--accent'));
  if (k.stop != null) level(k.stop, v('--down'));
  if (reveal) {
    const mark = (i, color) => {
      const xx = Math.round(x(i)) + 0.5;
      g.strokeStyle = color; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(xx, padT); g.lineTo(xx, padT + ph); g.stroke(); g.setLineDash([]);
    };
    mark(k.cut, v('--accent'));
    if (k.exit != null) mark(k.exit, v('--muted'));
  }
  g.restore();

  // level tags on the price scale, and day labels above the chart
  const tag = (p, color, yy) => {
    g.fillStyle = color; g.fillRect(pw + 2, yy - 8, padR - 4, 16);
    g.fillStyle = v('--panel'); g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText(fmtPrice(p), pw + 6, yy);
  };
  const yt = Math.round(y(k.trigger)) + 0.5;
  tag(k.trigger, v('--accent'), yt);
  if (k.stop != null) tag(k.stop, v('--down'), Math.max(Math.round(y(k.stop)) + 0.5, yt + 17));
  if (reveal) {
    const label = (i, text, color) => {
      g.fillStyle = color; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
      g.fillText(text, Math.min(Math.max(x(i), 14), pw - 14), padT - 4);
    };
    label(k.cut, 'buy', v('--accent'));
    if (k.exit != null && (k.exit - k.cut) * bw > 26) label(k.exit, 'out', v('--muted'));
  }
}

const LEGEND = '<div class="legend"><span><i style="background:var(--sma10)"></i>10-day average</span>' +
  '<span><i style="background:var(--sma20)"></i>20-day</span><span><i style="background:var(--sma50)"></i>50-day</span>' +
  '<span><i style="background:var(--accent)"></i>trigger</span>' +
  '<span><i style="background:var(--down)"></i>stop (previous day’s low)</span></div>';

// Facts known the evening before the trigger day.
function facts(k) {
  const items = [['ADR', k.adr == null ? '–' : k.adr.toFixed(1) + '%']]
    .concat(D.windows.map(w => [w.toUpperCase() + ' return', fmtPct(k.rets[w], 0)]))
    .concat([
      ['Trigger', fmtPrice(k.trigger) + ' (' + fmtPct(k.gap) + ' from the close)'],
      ['Stop', fmtPrice(k.stop) + (k.risk == null ? '' : ' (' + k.risk.toFixed(1) + '% below the entry)')],
      ['Setup qualified', plural(k.age, 'session') + ' earlier'],
    ]);
  return '<div class="chips">' + items.map(([a, b]) => '<span class="chip">' + a + ' <b>' + b + '</b></span>').join('') + '</div>';
}

function outcome(k) {
  const fw = D.forward_days.map(n => '<span class="chip">' + n + ' days <b class="' + cls(k.fwd[n]) + '">' + fmtPct(k.fwd[n]) + '</b></span>').join('');
  return '<div class="outcome"><p><b>' + esc(k.symbol) + '</b> <span class="muted">' + esc(k.name) + '</span> · trigger day ' + fmtDate(k.date) + '</p>' +
    '<p><span class="badge ' + (k.status === 'failed' ? 'wide' : 'ok') + '">' + OUTCOME[k.status] + '</span> ' +
    'Trade result <b class="' + cls(k.ret) + '">' + fmtPct(k.ret) + '</b> (<b class="' + cls(k.r) + '">' + fmtR(k.r) + '</b>), ' +
    'last sale after ' + plural(k.days, 'session') + '. Volume that day: ' + (k.vol_ratio == null ? '–' : k.vol_ratio.toFixed(1) + '× average') + '.</p>' +
    '<div class="chips"><span class="chip">Price against the entry after</span>' + fw + '</div></div>';
}

function stats(list) {
  if (!list.length) return null;
  const rs = list.filter(k => k.r != null);
  return {win: list.filter(k => k.ret > 0).length / list.length, ret: list.reduce((a, k) => a + k.ret, 0) / list.length,
          r: rs.length ? rs.reduce((a, k) => a + k.r, 0) / rs.length : null};
}
function scoreRow(label, list) {
  const s = stats(list);
  return '<tr><td class="l">' + label + '</td><td>' + list.length + '</td>' + (s
    ? '<td>' + Math.round(100 * s.win) + '%</td><td class="' + cls(s.ret) + '">' + fmtPct(s.ret, 2) + '</td><td class="' + cls(s.r) + '">' + fmtR(s.r) + '</td>'
    : '<td>–</td><td>–</td><td>–</td>') + '</tr>';
}
function renderScore() {
  const seen = deckCards().filter(k => k.id in answers), a = D.all;
  $('score').innerHTML = '<h2>Your picks so far</h2><div class="scroll"><table><thead><tr><th class="l"></th><th>Cards</th>' +
    '<th>Made money</th><th>Mean per trade</th><th>Mean R</th></tr></thead><tbody>' +
    scoreRow('Taken', seen.filter(k => answers[k.id] === 'take')) + scoreRow('Skipped', seen.filter(k => answers[k.id] === 'skip')) +
    scoreRow('All seen', seen) +
    (a ? '<tr class="ref"><td class="l">Every trigger in the backtest</td><td>' + a.n.toLocaleString('en-US') + '</td><td>' + Math.round(a.win) +
      '%</td><td>' + fmtPct(a.ret, 2) + '</td><td>' + fmtR(a.r) + '</td></tr>' : '') +
    '</tbody></table></div><p class="note">' + (deck === 'random'
      ? 'Your selection adds something only if “Taken” is clearly better than “All seen”. With fewer than about 100 answers the gap is mostly luck.'
      : 'This deck is a hand-picked group, so its score says little. Use the Random mix to measure your selection.') + '</p>';
}

function renderPractice() {
  renderScore();
  const el = $('pcard'), list = deckCards();
  if (!current) {
    el.innerHTML = '<p>' + (list.length ? 'You have been through every card in this deck. Choose another deck, or reset practice to start again.' : 'This deck has no cards.') + '</p>';
    return;
  }
  const k = current, done = list.filter(c => c.id in answers).length;
  el.innerHTML = '<div class="cardhead"><span class="muted">Card ' + (revealed ? done : done + 1) + ' of ' + list.length + '</span>' +
    '<span class="muted">' + (revealed ? 'Key: N for the next card' : 'The chart stops the evening before the trigger day. Keys: T take, S skip') + '</span></div>' +
    '<canvas class="chart big"></canvas>' + LEGEND + facts(k) +
    (revealed
      ? outcome(k) + '<div class="actions"><span>You chose <b>' + (answers[k.id] === 'take' ? 'Take' : 'Skip') + '</b>.</span><button id="next" class="primary">Next card</button></div>'
      : '<p class="ask">Would you place a buy order at the trigger?</p><div class="actions"><button id="take" class="primary">Take</button><button id="skip">Skip</button></div>');
  draw(el.querySelector('canvas'), k, revealed);
  if (revealed) $('next').onclick = advance;
  else { $('take').onclick = () => answer('take'); $('skip').onclick = () => answer('skip'); }
}
function answer(choice) {
  if (!current || revealed) return;
  answers[current.id] = choice; save(); revealed = true; renderPractice();
}
function advance() { current = nextCard(); revealed = false; renderPractice(); }

const PAGE = 12;
const SORTS = {
  'ret-desc': (a, b) => b.ret - a.ret, 'ret-asc': (a, b) => a.ret - b.ret,
  'date-desc': (a, b) => b.date - a.date, 'date-asc': (a, b) => a.date - b.date,
};
function renderBrowse() {
  const list = deckCards().sort(SORTS[$('sort').value]);
  const pages = Math.max(1, Math.ceil(list.length / PAGE));
  page = Math.max(0, Math.min(page, pages - 1));
  $('grid').innerHTML = list.slice(page * PAGE, (page + 1) * PAGE).map(k =>
    '<div class="panel card"><div class="cardhead"><span><b>' + esc(k.symbol) + '</b> · ' + fmtDate(k.date) +
    ' <span class="badge ' + (k.status === 'failed' ? 'wide' : 'ok') + '">' + OUTCOME[k.status] + '</span></span>' +
    '<span><b class="' + cls(k.ret) + '">' + fmtPct(k.ret) + '</b> · ' + fmtR(k.r) + ' · ' + plural(k.days, 'session') + '</span></div>' +
    '<canvas class="chart" data-id="' + k.id + '"></canvas>' + facts(k) + '</div>').join('') || '<p class="note">This deck has no cards.</p>';
  $('grid').querySelectorAll('canvas').forEach(cv => draw(cv, BY_ID.get(+cv.dataset.id), true));
  $('pageinfo').textContent = 'Page ' + (page + 1) + ' of ' + pages + ' · ' + plural(list.length, 'card');
  $('prev').disabled = page === 0; $('nextpage').disabled = page >= pages - 1;
}

function render() {
  document.querySelectorAll('#modes button').forEach(b => b.setAttribute('aria-selected', b.dataset.mode === mode));
  $('practice').hidden = mode !== 'practice'; $('browse').hidden = mode !== 'browse';
  $('sortwrap').hidden = mode !== 'browse'; $('reset').hidden = mode !== 'practice';
  if (mode === 'browse') { renderBrowse(); return; }
  if (!current || !inDeck(current) || (!revealed && current.id in answers)) { current = nextCard(); revealed = false; }
  renderPractice();
}

$('deck').innerHTML = D.decks.filter(d => d.n > 0).map(d => '<option value="' + d.tag + '">' + esc(d.label) + ' (' + d.n + ')</option>').join('') +
  '<option value="all">All cards (' + CARDS.length + ')</option>';
$('deck').value = deck;
$('deck').onchange = e => { deck = e.target.value; page = 0; current = null; revealed = false; render(); };
$('sort').onchange = () => { page = 0; renderBrowse(); };
document.querySelectorAll('#modes button').forEach(b => b.onclick = () => { mode = b.dataset.mode; render(); });
$('prev').onclick = () => { page--; renderBrowse(); window.scrollTo(0, 0); };
$('nextpage').onclick = () => { page++; renderBrowse(); window.scrollTo(0, 0); };
let armed = 0;
$('reset').onclick = e => {
  const now = Date.now(), b = e.currentTarget;
  if (now - armed < 4000) { answers = {}; save(); armed = 0; b.textContent = 'Reset practice'; current = null; revealed = false; render(); return; }
  armed = now; b.textContent = 'Click again to clear all answers';
  setTimeout(() => { if (armed === now) { armed = 0; b.textContent = 'Reset practice'; } }, 4000);
};
document.addEventListener('keydown', e => {
  if (mode !== 'practice' || e.metaKey || e.ctrlKey || e.altKey || /^(SELECT|INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
  const key = e.key.toLowerCase();
  if (!revealed && key === 't') answer('take');
  else if (!revealed && key === 's') answer('skip');
  else if (revealed && (key === 'n' || key === 'arrowright')) advance();
});
let timer;
const redraw = () => { clearTimeout(timer); timer = setTimeout(() => {
  if (mode === 'browse') { renderBrowse(); return; }
  const cv = $('pcard').querySelector('canvas');
  if (cv && current) draw(cv, current, revealed);
}, 120); };
window.addEventListener('resize', redraw);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
render();
"""


def render_page(study: dict) -> str:
    """The whole page as one HTML string: styles, data and script inline, nothing loaded from elsewhere."""
    title = f"Study set · {study['title']}"
    # "</" inside a <script> block could end it early (a company name is enough); "<\/" is the same JSON.
    data = json.dumps(study, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    a = study.get("all")
    intro = (
        f"Past days when a stock from the previous evening's watchlist reached its trigger price, "
        f"{study['first']} to {study['as_of']}. {len(study['cards'])} cards"
        + (f" chosen from {a['n']:,} triggers." if a else ".")
    )
    notes = f"""<details class="panel"><summary>How to read this</summary><ul>
<li><b>Practice</b> hides everything from the trigger day on. Decide whether you would place the buy order, then see what happened.
<b>Browse</b> shows whole charts.</li>
<li>The facts under each chart are what the watchlist alert showed the evening before.</li>
<li>The price scale is logarithmic: equal percentage moves take equal height.</li>
<li><b>Trade result</b> is the backtest's trade: bought at the trigger (or at the open if it opened above), stop at that day's low
from the next day on, then “{html.escape(study['exit'])}”, after costs. The dotted lines mark the buy day and the last sale.</li>
<li><b>R</b> is the result divided by the distance from the entry to the stop shown (the previous day's low).</li>
<li>Limits: entries come from daily prices. Qullamaggie buys an opening-range high with the stop at the low of that day, which
needs intraday prices and would give different results. Only stocks still listed today are included, and names with a
possible unadjusted split are left out.</li>
<li>Only the <b>Random mix</b> is a fair sample. The other decks are hand-picked groups for looking at, not for scoring.</li>
</ul></details>"""
    body = f"""<header><h1>{html.escape(title)}</h1><p>{html.escape(intro)}</p></header>
<nav class="tabs" id="modes"><button data-mode="practice" aria-selected="true">Practice</button><button data-mode="browse" aria-selected="false">Browse</button></nav>
<div class="panel controls"><label>Deck <select id="deck"></select></label>
<label id="sortwrap" hidden>Sort <select id="sort"><option value="ret-desc">Biggest gain first</option><option value="ret-asc">Biggest loss first</option>
<option value="date-desc">Newest first</option><option value="date-asc">Oldest first</option></select></label>
<span class="spacer"></span><button id="reset">Reset practice</button></div>
<section id="practice"><div class="panel card" id="pcard"></div><div class="panel" id="score"></div></section>
<section id="browse" hidden><div id="grid"></div>
<div class="pager"><button id="prev">Previous</button><span id="pageinfo"></span><button id="nextpage">Next</button></div></section>
{notes}
<footer><p>Information only, not investment advice. Paper trade before using real money. Prices: Yahoo Finance, NSE end-of-day
files. Data to {html.escape(study['as_of'])}. Your answers are stored in this browser only.</p></footer>"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}{STUDY_CSS}</style></head>
<body><main>{body}</main>
<script type="application/json" id="data">{data}</script><script>{STUDY_JS}</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    from scanner.data.cache import PriceCache

    ap = argparse.ArgumentParser(prog="python -m scanner.study", description="Build the study set of past trigger touches")
    ap.add_argument("--market", default="us", choices=["us", "india"])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "study"), help="folder for <market>.html (default: study)")
    args = ap.parse_args(argv)
    cfg = load_config(args.market)
    cache = PriceCache(cfg.data.cache_path())
    if not cache.prices_path.exists():
        print("No cached prices. Run `python -m scanner.run --market", args.market, "` first.", file=sys.stderr)
        return 1
    universe = pd.read_csv(cache.universe_path, dtype=str, keep_default_na=False)
    prices = cache.load()
    panels = Panels.from_long(prices[prices["symbol"].isin(universe["symbol"])])
    study = build_study(panels, cfg, universe)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{args.market}.html"
    target.write_text(render_page(study))
    decks = ", ".join(f"{d['label']}: {d['n']}" for d in study["decks"])
    print(f"Wrote {target} ({target.stat().st_size / 1e6:.1f} MB): {len(study['cards'])} cards. {decks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
