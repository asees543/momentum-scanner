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


def simulate(trades: pd.DataFrame, panels: Panels, rule: ExitRule, cost_pct: float) -> pd.DataFrame:
    """One result row per trade: ret (after costs, fraction of entry), r_multiple, exit_row, days, open.

    Each session from stop_from on, in this order:
      open <= stop            -> everything left exits at the open (gap through the stop)
      low <= stop             -> everything left exits at the stop
      partial not done and session >= entry + N -> sell `fraction` at the close
      partial done and close < SMA_k             -> the rest exits at the close
    Missing bars are skipped. Left over at the end of the data: valued at the last close, open=True.
    Return = sum(weight_i x exit_i x (1 - c)) / (entry x (1 + c)) - 1,  c = cost_pct / 100.
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
        r_mult = ret / (tr.risk_pct / 100) if np.isfinite(tr.risk_pct) else np.nan
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
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "median_days": t["days"].median(),
        "open": int(t["open"].sum()),
    }


def equity_curve(sim: pd.DataFrame, panels: Panels, cfg_bt, start_equity: float = 100.0) -> tuple[pd.Series, dict]:
    """Realized equity when risking risk_per_trade_pct of equity per trade (position = risk budget /
    stop distance, capped at max_position_pct of equity), at most max_positions open; same-day signals
    are taken in order of volume ratio. P&L is booked on the exit day (open trades at the end of the
    data at their last close), so drawdowns while a trade is open are not shown."""
    trades = sim.dropna(subset=["risk_pct"]).sort_values(["entry_row", "vol_ratio"], ascending=[True, False])
    return _realized_curve(trades, panels, cfg_bt, start_equity)


def _realized_curve(trades: pd.DataFrame, panels: Panels, cfg_bt, start_equity: float) -> tuple[pd.Series, dict]:
    n = len(panels.dates)
    equity = start_equity
    curve = np.full(n, np.nan)
    by_entry: dict[int, list] = {}
    for tr in trades.itertuples(index=False):
        by_entry.setdefault(tr.entry_row, []).append(tr)
    open_pos: list[tuple[int, float]] = []  # (exit_row, pnl)
    taken = skipped = 0
    for d in range(n):
        # book exits up to and including today
        still = []
        for exit_row, pnl in open_pos:
            if exit_row <= d:
                equity += pnl
            else:
                still.append((exit_row, pnl))
        open_pos = still
        for tr in by_entry.get(d, []):  # already sorted by volume ratio
            if len(open_pos) >= cfg_bt.max_positions:
                skipped += 1
                continue
            size = min(equity * cfg_bt.risk_per_trade_pct / tr.risk_pct, equity * cfg_bt.max_position_pct / 100)
            open_pos.append((tr.exit_row, size * tr.ret))
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
