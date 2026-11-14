"""Point-in-time backtest of the scans (SPEC "Backtest"): signals, forward returns, trade simulation.

    uv run python -m scanner.backtest --market us
    uv run python -m scanner.backtest --market india

No lookahead: the live scan's rule frames (liquidity -> leaders -> setup) are computed once over the
full history. Every rule only uses rolling windows, shifts and same-day cross-sectional ranks, so the
value on day t depends only on data up to t. `verify_against_live` re-runs the live scanner on
truncated data (panels.as_of(day)) and checks the signals match.

Two entry models (user decision: compare both):
  A  buy-stop at the trigger, placed from the evening watchlist. On day t, a name on day t-1's
     watchlist whose high reaches its trigger is bought at max(open_t, trigger) (a gap above the
     trigger fills at the open). This includes triggers that then FAIL: a close back below the
     trigger, or not enough volume. Stop = day t's low, active from day t+1 (daily bars can't show
     whether the low came before or after the fill).
  B  next-day open after the end-of-day breakout alert: each confirmed breakout on day t (close >
     trigger and volume >= 1.5x average; status "ok" or "too wide") is bought at open_{t+1}. Stop =
     day t's low, active from t+1. Skipped if open_{t+1} is already at or below the stop.

Exits (SPEC): stop at the breakout-day low (filled at the stop, or at the open if a day gaps below
it); sell `fraction` at the close N sessions after entry; trail the rest out at the first close below
the k-day SMA after that. Still open at the end of the data = valued at the last close, marked open.
Costs: entry x (1 + c), exit x (1 - c), c = (cost + slippage) % per side.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from scanner import indicators as ind
from scanner.config import MarketConfig, load_config
from scanner.data.base import Panels
from scanner.scans import (
    LeaderFrames,
    LiquidityFrames,
    SetupFrames,
    _rows_needed,
    _watch_state,
    leader_frames,
    liquidity_frames,
    run_scans,
    setup_frames,
)

EVENT_COLUMNS = [
    "date", "symbol", "row", "trigger", "open", "high", "low", "close", "entry_a", "stop", "status",
    "vol_ratio", "adr_prev", "setup_age", "flag_day_move",
]


# --- signals ----------------------------------------------------------------------------------


@dataclass
class HistoryFrames:
    panels: Panels
    liq: LiquidityFrames
    lf: LeaderFrames
    sf: SetupFrames
    first_row: int  # first session with a full warm-up for every rule


def history_frames(
    panels: Panels, cfg: MarketConfig, mode: Literal["fixed", "rank"] | None = None, warmup: int | None = None
) -> HistoryFrames:
    """Rule frames over the whole history. Signals start after `warmup` sessions (default: the history
    the live scan reads, so every rule, including the 6M window, has full data from day one)."""
    mode = mode or cfg.leaders.mode
    liq = liquidity_frames(panels, cfg.liquidity)
    lf = leader_frames(panels, liq, cfg.leaders, mode)
    sf = setup_frames(panels, liq, lf, cfg.setup)
    return HistoryFrames(panels, liq, lf, sf, first_row=_rows_needed(cfg) if warmup is None else warmup)


def trigger_events(hf: HistoryFrames, cfg: MarketConfig) -> pd.DataFrame:
    """Every day t on which a name from day t-1's watchlist traded up to its trigger (Model A), with
    the breakout classification the live scanner gives that day (Model B uses status != "failed"):

    status = "ok"        close > trigger, volume >= min_volume_ratio x average, risk% <= ADR%_(t-1)
             "too wide"  confirmed, but risk% = (close - low) / close x 100 is wider than ADR%_(t-1)
             "failed"    touched the trigger without a confirmed close (closed below / low volume)
    """
    p, sf, liq = hf.panels, hf.sf, hf.liq
    o, h, lo, c, v = (getattr(p, f).to_numpy() for f in ("open", "high", "low", "close", "volume"))
    avg_vol = sf.avg_volume_prior.to_numpy()
    adr = liq.adr_pct.to_numpy()
    max_lb = max(w.lookback for w in cfg.leaders.windows.values())
    ratio = p.close / p.close.shift(1)
    day_move = np.fmax(ratio.rolling(max_lb, min_periods=1).max(), (1 / ratio).rolling(max_lb, min_periods=1).max())
    flagged = (day_move >= cfg.leaders.flag_day_move_multiple).to_numpy()
    bcfg = cfg.breakout
    symbols = p.symbols
    rows = []
    for t in range(max(hf.first_row, 1), len(p.dates)):
        st = _watch_state(sf, p.close, t - 1, bcfg.watchlist_memory_sessions)
        idx = np.flatnonzero(st["on_watchlist"].to_numpy())
        if not len(idx):
            continue
        trig = st["trigger"].to_numpy()[idx]
        touched = h[t, idx] >= trig
        if not touched.any():
            continue
        idx, trig = idx[touched], trig[touched]
        age = st["age"].to_numpy()[idx] + 1
        close_t, low_t, open_t = c[t, idx], lo[t, idx], o[t, idx]
        confirmed = (close_t > trig) & (v[t, idx] >= bcfg.min_volume_ratio * avg_vol[t, idx])
        risk = 100 * (close_t - low_t) / close_t
        wide = risk > bcfg.max_risk_vs_adr * adr[t - 1, idx]
        status = np.where(confirmed, np.where(wide, "too wide", "ok"), "failed")
        for k, j in enumerate(idx):
            rows.append(
                (p.dates[t], symbols[j], t, trig[k], open_t[k], h[t, j], low_t[k], close_t[k],
                 max(open_t[k], trig[k]), low_t[k], status[k], v[t, j] / avg_vol[t, j], adr[t - 1, j],
                 int(age[k]), bool(flagged[t, j]))
            )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def verify_against_live(panels: Panels, cfg: MarketConfig, events: pd.DataFrame, days: list) -> list[str]:
    """Re-run the live scanner on panels truncated at each day and compare its breakouts (symbol,
    status, trigger, stop) with the backtest's confirmed events. Returns a list of mismatches."""
    problems = []
    for day in days:
        live = run_scans(panels.as_of(day), cfg).breakouts.table
        mine = events[(events["date"] == pd.Timestamp(day)) & (events["status"] != "failed")].set_index("symbol")
        if set(live.index) != set(mine.index):
            problems.append(f"{pd.Timestamp(day):%Y-%m-%d}: live {sorted(live.index)} vs backtest {sorted(mine.index)}")
            continue
        for s in live.index:
            a, b = live.loc[s], mine.loc[s]
            if a["status"] != b["status"] or abs(a["trigger"] - b["trigger"]) > 1e-6 * a["trigger"] or a["stop"] != b["stop"]:
                problems.append(f"{pd.Timestamp(day):%Y-%m-%d} {s}: live {a['status']}/{a['trigger']:.4f} "
                                f"vs backtest {b['status']}/{b['trigger']:.4f}")
    return problems


# --- trades -------------------------------------------------------------------------------------

TRADE_COLUMNS = [
    "model", "symbol", "col", "signal_date", "entry_row", "entry_date", "entry", "stop", "stop_from",
    "group", "risk_pct", "vol_ratio", "flag_day_move",
]


def build_trades(events: pd.DataFrame, panels: Panels, model: Literal["A", "B"]) -> pd.DataFrame:
    """Entries for one model (see module docstring). risk_pct = (entry - stop) / entry x 100."""
    col = {s: i for i, s in enumerate(panels.symbols)}
    o = panels.open.to_numpy()
    rows = []
    last = len(panels.dates) - 1
    for e in events.itertuples(index=False):
        j = col[e.symbol]
        if model == "A":
            entry_row, entry, stop_from = e.row, e.entry_a, e.row + 1
        else:
            if e.status == "failed" or e.row >= last:
                continue
            entry_row, entry, stop_from = e.row + 1, o[e.row + 1, j], e.row + 1
            if not np.isfinite(entry) or entry <= e.stop:
                continue  # gapped to or below the stop before we could buy: skipped
        risk = 100 * (entry - e.stop) / entry
        rows.append((model, e.symbol, j, e.date, entry_row, panels.dates[entry_row], entry, e.stop, stop_from,
                     e.status, risk if risk > 0 else np.nan, e.vol_ratio, e.flag_day_move))
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


@dataclass(frozen=True)
class ExitRule:
    fraction: float  # sold at the close N sessions after entry
    after_days: int  # N
    trail_sma: int  # the rest leaves at the first close below this SMA (after the partial)

    @property
    def label(self) -> str:
        frac = "1/3" if abs(self.fraction - 1 / 3) < 1e-6 else "1/2" if abs(self.fraction - 0.5) < 1e-6 else f"{self.fraction:.2f}"
        return f"sell {frac} after {self.after_days}d, trail {self.trail_sma}-SMA"


def simulate(trades: pd.DataFrame, panels: Panels, rule: ExitRule, cost_pct: float, min_risk_pct: float = 0.0
             ) -> pd.DataFrame:
    """One result row per trade: ret (after costs, fraction of entry), r_multiple, exit_row, days, open.

    Each session from stop_from on, in this order:
      open <= stop            -> everything left exits at the open (gap through the stop)
      low <= stop             -> everything left exits at the stop
      partial not done and session >= entry + N -> sell `fraction` at the close
      partial done and close < SMA_k             -> the rest exits at the close
    Missing bars are skipped. Left over at the end of the data: valued at the last close, open=True.
    Return = sum(weight_i x exit_i x (1 - c)) / (entry x (1 + c)) - 1,  c = cost_pct / 100.
    R multiple = return / max(stop distance, min_risk_pct) (both as fractions of the entry price): a
    near-zero stop distance would otherwise turn ordinary returns into huge R values.
    """
    o, lo, c = (getattr(panels, f).to_numpy() for f in ("open", "low", "close"))
    sma = ind.sma(panels.close, rule.trail_sma).to_numpy()
    cost = cost_pct / 100
    n = len(panels.dates)
    out = []
    for tr in trades.itertuples(index=False):
        j, e, entry, stop = tr.col, tr.entry_row, tr.entry, tr.stop
        remaining, proceeds, partial_done, exit_row, still_open = 1.0, 0.0, False, None, False
        last_close = c[e, j]
        for d in range(tr.stop_from, n):
            if not np.isfinite(c[d, j]):
                continue
            last_close = c[d, j]
            if o[d, j] <= stop:
                proceeds += remaining * o[d, j]
                remaining, exit_row = 0.0, d
                break
            if lo[d, j] <= stop:
                proceeds += remaining * stop
                remaining, exit_row = 0.0, d
                break
            if not partial_done and d >= e + rule.after_days:
                proceeds += rule.fraction * c[d, j]
                remaining -= rule.fraction
                partial_done = True
            if partial_done and np.isfinite(sma[d, j]) and c[d, j] < sma[d, j]:
                proceeds += remaining * c[d, j]
                remaining, exit_row = 0.0, d
                break
        if remaining > 0:
            proceeds += remaining * last_close
            exit_row, still_open = n - 1, True
        ret = proceeds * (1 - cost) / (entry * (1 + cost)) - 1
        r_mult = ret / (max(tr.risk_pct, min_risk_pct) / 100) if np.isfinite(tr.risk_pct) else np.nan
        out.append((ret, r_mult, exit_row, exit_row - e, still_open))
    res = pd.DataFrame(out, columns=["ret", "r_multiple", "exit_row", "days", "open"], index=trades.index)
    return pd.concat([trades, res], axis=1)


def forward_returns(trades: pd.DataFrame, panels: Panels, bench: pd.DataFrame, days: list[int]) -> pd.DataFrame:
    """Close k sessions after the entry session vs the entry price (no costs), and the benchmark from
    the entry session's open to the same close:  fwd_k = close_{e+k} / entry - 1,
    bench_k = bench_close_{e+k} / bench_open_e - 1,  excess_k = fwd_k - bench_k."""
    c = panels.close.to_numpy()
    b_open = bench["open"].reindex(panels.dates).to_numpy()
    b_close = bench["close"].reindex(panels.dates).to_numpy()
    n = len(panels.dates)
    out = trades.copy()
    for k in days:
        rows = out["entry_row"].to_numpy() + k
        ok = rows < n
        r = np.where(ok, rows, 0)
        fwd = np.where(ok, c[r, out["col"].to_numpy()] / out["entry"].to_numpy() - 1, np.nan)
        bmk = np.where(ok, b_close[r] / b_open[out["entry_row"].to_numpy()] - 1, np.nan)
        out[f"fwd_{k}"], out[f"bench_{k}"], out[f"excess_{k}"] = fwd, bmk, fwd - bmk
    return out


# --- statistics -----------------------------------------------------------------------------------


def trade_stats(t: pd.DataFrame) -> dict:
    """Per-trade summary. expectancy = mean return per trade; profit factor = sum of gains / sum of losses."""
    if t.empty:
        return {"trades": 0}
    r = t["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "trades": len(t),
        "win_rate": len(wins) / len(t),
        "mean_ret": r.mean(),
        "median_ret": r.median(),
        "avg_win": wins.mean() if len(wins) else np.nan,
        "avg_loss": losses.mean() if len(losses) else np.nan,
        "mean_r": t["r_multiple"].mean(),
        "median_r": t["r_multiple"].median(),
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "median_days": t["days"].median(),
        "open": int(t["open"].sum()),
    }


def equity_curve(sim: pd.DataFrame, panels: Panels, cfg_bt, start_equity: float = 100.0) -> tuple[pd.Series, dict]:
    """Realized equity when risking risk_per_trade_pct of equity per trade:
        position = equity x risk_per_trade_pct / max(stop distance %, min_risk_pct),
    capped at max_position_pct of equity and at the cash left (open positions never exceed 100% of
    equity: no leverage), at most max_positions open. P&L is booked on the exit day (open trades at the
    end of the data at their last close), so drawdowns while a trade is open are not shown."""
    trades = sim.dropna(subset=["risk_pct"])
    if len(trades) and (trades["model"] == "A").all():
        # Model A: which buy-stops fill first within a day is unknown and the day's volume comes after
        # the fill, so same-day fills are taken in a fixed random order (seeded, repeatable).
        order = np.random.default_rng(11).permutation(len(trades))
        trades = trades.assign(_prio=order).sort_values(["entry_row", "_prio"])
    else:
        # Model B: the breakout day's volume is known before the next open.
        trades = trades.sort_values(["entry_row", "vol_ratio"], ascending=[True, False])
    return _realized_curve(trades, panels, cfg_bt, start_equity)


def _realized_curve(trades: pd.DataFrame, panels: Panels, cfg_bt, start_equity: float) -> tuple[pd.Series, dict]:
    n = len(panels.dates)
    equity = start_equity
    curve = np.full(n, np.nan)
    by_entry: dict[int, list] = {}
    for tr in trades.itertuples(index=False):
        by_entry.setdefault(tr.entry_row, []).append(tr)
    open_pos: list[tuple[int, float, float]] = []  # (exit_row, pnl, position value)
    taken = skipped = 0
    for d in range(n):
        # book exits up to and including today
        still = []
        for exit_row, pnl, value in open_pos:
            if exit_row <= d:
                equity += pnl
            else:
                still.append((exit_row, pnl, value))
        open_pos = still
        for tr in by_entry.get(d, []):  # already in priority order
            cash = equity - sum(v for _, _, v in open_pos)
            if len(open_pos) >= cfg_bt.max_positions or cash <= 0:
                skipped += 1
                continue
            risk = max(tr.risk_pct, cfg_bt.min_risk_pct)
            size = min(equity * cfg_bt.risk_per_trade_pct / risk, equity * cfg_bt.max_position_pct / 100, cash)
            open_pos.append((tr.exit_row, size * tr.ret, size))
            taken += 1
        curve[d] = equity
    s = pd.Series(curve, index=panels.dates).dropna()
    first = trades["entry_row"].min() if len(trades) else 0
    s = s.iloc[first:] if len(s) > first else s
    return s, {"taken": taken, "skipped_full": skipped, **curve_stats(s)}


def curve_stats(s: pd.Series) -> dict:
    """CAGR = (end / start)^(1 / years) - 1;  max drawdown = largest fall from a running peak."""
    if len(s) < 2 or s.iloc[0] <= 0:
        return {"cagr": np.nan, "max_drawdown": np.nan}
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = (s / s.cummax() - 1).min()
    return {"cagr": cagr, "max_drawdown": dd}


# --- report -------------------------------------------------------------------------------------


def _pct(x, digits=1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:+.{digits}f}%"


def _plain_pct(x, digits=0) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _num(x, digits=2) -> str:
    return "–" if x is None or not np.isfinite(x) else ("∞" if np.isinf(x) else f"{x:.{digits}f}")


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _stats_row(label: str, s: dict) -> list[str]:
    if not s.get("trades"):
        return [label, "0"] + ["–"] * 8
    return [label, f"{s['trades']:,}", _plain_pct(s["win_rate"]), _pct(s["avg_win"]), _pct(s["avg_loss"]),
            _pct(s["mean_ret"], 2), _num(s["mean_r"]), _num(s["median_r"]), _num(s["profit_factor"]),
            _num(s["median_days"], 0)]


STATS_HEADER = ["Group", "Trades", "Win rate", "Avg win", "Avg loss", "Mean / trade", "Mean R", "Median R",
                "Profit factor", "Median days"]
GROUPS = {"A": ["all", "ok", "too wide", "failed"], "B": ["all", "ok", "too wide"]}
# Model A's groups depend on the breakout day's close and volume, which aren't known when a buy-stop
# fills: they explain results but can't be traded. Model B's groups are known before the next open.
HINDSIGHT = "†"


def _label(m: str, g: str) -> str:
    return f"{m} · {g}" + (f" {HINDSIGHT}" if m == "A" and g != "all" else "")
MODEL_NAME = {"A": "A · buy-stop at the trigger (breakout-day entry)", "B": "B · next-day open after the alert"}


def _group(df: pd.DataFrame, g: str) -> pd.DataFrame:
    return df if g == "all" else df[df["group"] == g]


@dataclass
class BacktestResult:
    market: str
    report: str
    trades: pd.DataFrame  # headline simulation + forward returns, both models
    verify_problems: list[str]


def run_backtest(panels: Panels, cfg: MarketConfig, bench: pd.DataFrame, universe_size: int,
                 verify_days: int = 12, seed: int = 7) -> BacktestResult:
    b = cfg.backtest
    hf = history_frames(panels, cfg)
    events = trigger_events(hf, cfg)
    rng = np.random.default_rng(seed)
    candidates = list(panels.dates[hf.first_row + 1:])
    days = sorted(rng.choice(candidates, size=min(verify_days, len(candidates)), replace=False).tolist()) + [panels.dates[-1]]
    problems = verify_against_live(panels, cfg, events, days)

    cost = b.cost_pct_per_side + b.slippage_pct_per_side
    rules = [ExitRule(f, n, k) for f, n, k in itertools.product(b.partial_fractions, b.partial_after_days, b.trail_sma)]
    headline = ExitRule(b.headline_partial_fraction, b.headline_partial_after_days, b.headline_trail_sma)
    trades = {m: build_trades(events, panels, m) for m in ("A", "B")}
    sims = {(m, r): simulate(trades[m], panels, r, cost, b.min_risk_pct) for m in trades for r in rules}
    head = {m: forward_returns(sims[(m, next(r for r in rules if r == headline))], panels, bench, b.forward_days)
            for m in trades}

    first, last = panels.dates[hf.first_row], panels.dates[-1]
    sessions = len(panels.dates) - hf.first_row
    n_b_candidates = int((events["status"] != "failed").sum())
    lines = [
        f"# Backtest · {cfg.market.upper()}",
        "",
        f"Generated {date.today():%Y-%m-%d} by `python -m scanner.backtest --market {cfg.market}`. "
        "**Information only, not investment advice. Paper trade before using real money.**",
        "",
        "## Setup",
        "",
        f"- Signals from **{first:%Y-%m-%d} to {last:%Y-%m-%d}** ({sessions:,} sessions), on today's universe of "
        f"{universe_size:,} stocks; leaders in **{cfg.leaders.mode}** mode; all thresholds from `config/{cfg.market}.yaml`.",
        f"- Costs: {b.cost_pct_per_side:g}% + {b.slippage_pct_per_side:g}% slippage per buy or sell. "
        f"Benchmark: {b.benchmark}.",
        f"- Point-in-time check: the live scanner re-run on data cut off at {len(days)} random days "
        + ("gave **exactly the same breakouts** as the backtest." if not problems
           else f"**differed on {len(problems)} day(s)**: {problems[:3]}"),
        "",
        "Entry models:",
        f"- **{MODEL_NAME['A']}**: every day a name from the previous evening's watchlist traded up to its trigger; "
        "bought at the trigger (or the open if it gapped above). Includes triggers that **failed** (closed back below "
        "the trigger or without 1.5× volume), because a buy-stop fills on those too. Stop = that day's low, from the next day.",
        f"- **{MODEL_NAME['B']}**: every confirmed breakout (ok or too wide), bought at the next open; skipped if "
        "that open was already at or below the stop.",
        f"- Exits (SPEC): stop at the breakout-day low; sell part after N days; trail the rest below an SMA. "
        f"Headline exit, fixed before running: **{headline.label}**.",
        "",
        "## Read this first: biases",
        "",
        "- **Survivorship.** Only stocks listed *today* are tested. Stocks delisted since 2022 (many failed momentum "
        "names among them) are missing from free data, so results are **better than reality**.",
        "- **Universe look-ahead.** Today's lists also decide which tickers count as common stock in the past.",
        "- **Data quality.** Unadjusted splits/bonuses create fake moves; results are also shown without ⚠-flagged names.",
        f"- **{HINDSIGHT} Hindsight groups.** Model A's split into ok / too wide / failed uses the breakout day's close "
        "and volume, which aren't known when a buy-stop fills. Those rows explain where A's results come from but "
        "can't be traded; **only “A · all” is a tradeable strategy**. Model B's groups are known before its entry.",
        f"- **R multiples** use a minimum stop distance of {b.min_risk_pct:g}%: a few near-zero stops otherwise "
        "dominate the average. Median R is shown next to it.",
        "- **Daily bars.** They can't show whether the day's low came before or after a buy-stop filled; Model A's stop "
        "only starts the next day, which flatters A slightly. Gaps through stops are filled at the open.",
        "- **Costs and slippage are estimates**; real fills on breakouts are often worse.",
        f"- **One regime.** {first:%Y}–{last:%Y} only. **8 exit combinations** were tested, so the best one looks "
        "good partly by luck; judge by the pre-chosen headline. Results don't transfer automatically to other years.",
        "",
        "## Signals",
        "",
        _md_table(
            ["", "Count"],
            [
                ["Trigger touches (Model A fills)", f"{len(trades['A']):,}"],
                ["…of which ok breakouts", f"{int((events['status'] == 'ok').sum()):,}"],
                ["…of which too wide (stop > 1 ADR)", f"{int((events['status'] == 'too wide').sum()):,}"],
                ["…of which failed (no confirmed close)", f"{int((events['status'] == 'failed').sum()):,}"],
                ["Confirmed breakouts (Model B candidates)", f"{n_b_candidates:,}"],
                ["…entered at the next open", f"{len(trades['B']):,}"],
                ["…skipped: opened at or below the stop", f"{n_b_candidates - len(trades['B']):,}"],
            ],
        ),
    ]

    lines += ["", "## Forward returns (no costs)", "",
              "Mean return from the entry price to the close k sessions later, and the benchmark over the same "
              "sessions. *Beat* = share of trades ahead of the benchmark.", ""]
    rows = []
    for m in ("A", "B"):
        for g in GROUPS[m]:
            t = _group(head[m], g)
            cells = [_label(m, g), f"{len(t):,}"]
            for k in b.forward_days:
                cells += [_pct(t[f"fwd_{k}"].mean()), _pct(t[f"bench_{k}"].mean()),
                          _plain_pct((t[f"excess_{k}"] > 0).mean())]
            rows.append(cells)
    hdr = ["Model · group", "Trades"]
    for k in b.forward_days:
        hdr += [f"{k}d", f"{k}d bench", f"{k}d beat"]
    lines.append(_md_table(hdr, rows))

    lines += ["", f"## Trade simulation: headline exit ({headline.label})", ""]
    for m in ("A", "B"):
        sim = sims[(m, headline)]
        lines += [f"**{MODEL_NAME[m]}**", "",
                  _md_table(STATS_HEADER, [_stats_row(_label(m, g), trade_stats(_group(sim, g))) for g in GROUPS[m]]), ""]

    lines += ["## All 8 exit combinations", "",
              "Mean return per trade (after costs) and profit factor. The headline is marked ★.", ""]
    rows = []
    for r in rules:
        cells = [("★ " if r == headline else "") + r.label]
        for m, g in (("A", "all"), ("B", "all"), ("B", "ok")):
            s = trade_stats(_group(sims[(m, r)], g))
            cells += [_pct(s.get("mean_ret"), 2) if s.get("trades") else "–", _num(s.get("profit_factor"))]
        rows.append(cells)
    lines.append(_md_table(["Exit", "A all: mean", "PF", "B all: mean", "PF", "B ok: mean", "PF"], rows))

    lines += ["", "## By year (headline exit)", ""]
    rows = []
    for m, g in (("A", "all"), ("B", "all"), ("B", "ok")):
        sim = _group(sims[(m, headline)], g)
        for year, t in sim.groupby(sim["entry_date"].dt.year):
            s = trade_stats(t)
            rows.append([_label(m, g), str(year), f"{s['trades']:,}", _plain_pct(s["win_rate"]), _pct(s["mean_ret"], 2),
                         _num(s["profit_factor"])])
    lines.append(_md_table(["Model · group", "Year", "Trades", "Win rate", "Mean / trade", "Profit factor"], rows))

    lines += ["", "## Without ⚠-flagged names (headline exit)", "",
              "Names whose 6-month window had a one-day move beyond the flag threshold (possible unadjusted split/bonus).", ""]
    rows = []
    for m in ("A", "B"):
        for g in GROUPS[m]:
            sim = _group(sims[(m, headline)], g)
            clean = sim[~sim["flag_day_move"]]
            rows.append(_stats_row(f"{_label(m, g)} (n flagged: {int(sim['flag_day_move'].sum())})", trade_stats(clean)))
    lines.append(_md_table(STATS_HEADER, rows))

    lines += ["", "## Equity curve (headline exit, realized)", "",
              f"Risking {b.risk_per_trade_pct:g}% of equity per trade (stop distance at least {b.min_risk_pct:g}%; no "
              f"position above {b.max_position_pct:g}% of equity; never more than 100% invested), at most "
              f"{b.max_positions} open. Model A takes same-day fills in a fixed random order (the day's volume is only "
              "known after the fill); Model B by volume. P&L is counted when a trade closes, so drawdowns while "
              "trades are open are not shown. Only tradeable groups are listed.", ""]
    rows = []
    for m, g in (("A", "all"), ("B", "all"), ("B", "ok")):
        curve, cs = equity_curve(_group(sims[(m, headline)], g), panels, b)
        rows.append([f"{m} · {g}", f"{cs['taken']:,}", f"{cs['skipped_full']:,}", _pct(cs["cagr"]), _pct(cs["max_drawdown"])])
    bc = bench["close"].reindex(panels.dates[hf.first_row:]).dropna()
    bs = curve_stats(bc)
    rows.append([f"{b.benchmark} buy and hold", "–", "–", _pct(bs["cagr"]), _pct(bs["max_drawdown"])])
    lines.append(_md_table(["Model · group", "Trades taken", "Skipped (full or no cash)", "CAGR", "Max drawdown"], rows))
    lines.append("")

    all_trades = pd.concat([head["A"], head["B"]], ignore_index=True)
    return BacktestResult(cfg.market, "\n".join(lines), all_trades, problems)


def load_benchmark(symbol: str, start: date) -> pd.DataFrame:
    from scanner.data.yfinance_provider import YFinanceProvider

    batches = list(YFinanceProvider(pause_seconds=0).fetch_batches([symbol], start))
    df = pd.concat([x.prices for x in batches])
    if df.empty:
        raise RuntimeError(f"no benchmark data for {symbol}")
    return df.set_index("date")[["open", "close"]]


def main(argv: list[str] | None = None) -> int:
    from scanner.data.cache import PriceCache

    ap = argparse.ArgumentParser(prog="python -m scanner.backtest", description="Point-in-time backtest of the scans")
    ap.add_argument("--market", default="us", choices=["us", "india"])
    ap.add_argument("--out", default="reports", help="folder for backtest-<market>.md (default: reports)")
    args = ap.parse_args(argv)
    cfg = load_config(args.market)
    cache = PriceCache(cfg.data.cache_path())
    if not cache.prices_path.exists():
        print("No cached prices. Run `python -m scanner.run --market", args.market, "` first.", file=sys.stderr)
        return 1
    universe = pd.read_csv(cache.universe_path, dtype=str, keep_default_na=False)
    prices = cache.load()
    panels = Panels.from_long(prices[prices["symbol"].isin(universe["symbol"])])
    bench = load_benchmark(cfg.backtest.benchmark, panels.dates[0].date())
    res = run_backtest(panels, cfg, bench, len(universe))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"backtest-{args.market}.md").write_text(res.report)
    trades_dir = Path(cfg.data.cache_path()).parent.parent / "backtest"
    trades_dir.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(trades_dir / f"{args.market}_trades.csv", index=False)
    print(res.report)
    print(f"\nWrote {out / f'backtest-{args.market}.md'} and {trades_dir / f'{args.market}_trades.csv'}")
    return 0 if not res.verify_problems else 2


if __name__ == "__main__":
    sys.exit(main())
