"""Market-agnostic scans. They take price panels + config and never know which market they run on.

Phase 1: Stage 1 liquidity filter and Stage 2 momentum leaders (SPEC.md).
Phase 2 will add the consolidation setup (watchlist) and the breakout trigger.

All scans are evaluated on a single "as-of" session: the last row of the panels passed in.
Pass `panels.as_of(date)` to evaluate a past date with no lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from scanner import indicators as ind
from scanner.config import LeadersConfig, LiquidityConfig
from scanner.data.base import Panels


@dataclass
class LeadersResult:
    as_of: pd.Timestamp
    mode: Literal["fixed", "rank"]
    # One row per liquid symbol that passed at least one window, sorted for display.
    # Columns: close, ret_<w>, pass_<w> (and rs_<w> in rank mode) per window, adr_pct,
    # avg_traded_value, n_passed.
    table: pd.DataFrame
    counts: dict[str, int]  # window -> number of symbols passing that window
    active_count: int  # symbols with a bar on the as-of date
    stale_count: int  # symbols in the panels with NO bar on the as-of date (halted/delisted/failed)
    liquid_count: int  # symbols passing the Stage 1 liquidity filter
    windows: list[str]


def liquidity_snapshot(panels: Panels, lcfg: LiquidityConfig) -> pd.DataFrame:
    """Stage 1 on the as-of (last) session, one row per symbol that has a bar that day.

    liquid = close >= min_price
             AND mean(close x volume, last `window` bars) >= min_avg_traded_value
             AND ADR%(last `window` bars) >= min_adr_pct
    A symbol without a full `window` of bars has NaN averages and is not liquid.
    """
    n = lcfg.window
    tail = panels.close.index[-(n + 1):]  # a few extra rows are harmless; keeps the maths cheap
    close = panels.close.loc[tail]
    high = panels.high.loc[tail]
    low = panels.low.loc[tail]
    vol = panels.volume.loc[tail]

    last_close = close.iloc[-1]
    snap = pd.DataFrame(
        {
            "close": last_close,
            "avg_traded_value": ind.avg_traded_value(close, vol, n).iloc[-1],
            "adr_pct": ind.adr_pct(high, low, n).iloc[-1],
        }
    )
    snap = snap[last_close.notna()]
    snap["liquid"] = (
        (snap["close"] >= lcfg.min_price)
        & (snap["avg_traded_value"] >= lcfg.min_avg_traded_value)
        & (snap["adr_pct"] >= lcfg.min_adr_pct)
    ).fillna(False).astype(bool)
    return snap


def momentum_leaders(
    panels: Panels,
    lcfg: LiquidityConfig,
    leaders: LeadersConfig,
    mode: Literal["fixed", "rank"] | None = None,
) -> LeadersResult:
    """Stage 2 momentum leaders among Stage 1 liquid symbols, for every configured window.

    fixed mode: pass_w = close_today >= min_multiple_w x close[lookback_w bars ago]
                (e.g. 1m: close >= 1.25 x close 22 bars ago, i.e. a return of at least +25%)
    rank mode:  pass_w = RS percentile of return_w > 100 - rank_top_pct, ranked only among liquid
                symbols that have a return for that window (the top 7% by default)
    A symbol without enough history for a window (e.g. a recent IPO for 6m) just fails that window.
    """
    if panels.close.empty:
        raise ValueError("no price data")
    mode = mode or leaders.mode
    as_of = panels.dates[-1]
    max_lb = max(w.lookback for w in leaders.windows.values())
    rows = panels.close.index[-(max_lb + 1):]
    close = panels.close.loc[rows]

    snap = liquidity_snapshot(panels, lcfg)
    liquid = snap.index[snap["liquid"]]
    out = snap.loc[liquid, ["close"]].copy()

    counts: dict[str, int] = {}
    for name, w in leaders.windows.items():
        now = close.iloc[-1].reindex(liquid)
        then = (
            close.iloc[-(w.lookback + 1)].reindex(liquid)
            if len(close) > w.lookback
            else pd.Series(float("nan"), index=liquid)
        )
        ret = now / then - 1  # = pct_return(close, lookback) on the as-of row
        out[f"ret_{name}"] = ret
        if mode == "fixed":
            passed = (now >= w.min_multiple * then).fillna(False)
        else:
            rs = ind.rs_rank_pct(ret)
            out[f"rs_{name}"] = rs
            passed = (rs > 100 - leaders.rank_top_pct).fillna(False)
        out[f"pass_{name}"] = passed.astype(bool)
        counts[name] = int(passed.sum())

    out["adr_pct"] = snap.loc[liquid, "adr_pct"]
    out["avg_traded_value"] = snap.loc[liquid, "avg_traded_value"]
    pass_cols = [f"pass_{n}" for n in leaders.windows]
    out["n_passed"] = out[pass_cols].sum(axis=1).astype(int)
    out = out[out["n_passed"] > 0]
    out = out.sort_values(["n_passed", f"ret_{leaders.sort_by}"], ascending=[False, False], kind="stable")
    out.index.name = "symbol"

    total_symbols = panels.close.shape[1]
    return LeadersResult(
        as_of=as_of,
        mode=mode,
        table=out,
        counts=counts,
        active_count=len(snap),
        stale_count=total_symbols - len(snap),
        liquid_count=len(liquid),
        windows=list(leaders.windows),
    )
