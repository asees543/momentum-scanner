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
    # Columns: close, ret_<w>, pass_<w> (and rs_<w> in rank mode) per window, max_day_move,
    # flag_day_move, adr_pct, avg_traded_value, n_passed.
    table: pd.DataFrame
    counts: dict[str, int]  # window -> number of symbols passing that window
    active_count: int  # symbols with a bar on the as-of date
    stale_count: int  # symbols in the panels with NO bar on the as-of date (halted/delisted/failed)
    liquid_count: int  # symbols passing the Stage 1 liquidity filter
    windows: list[str]


@dataclass
class LiquidityFrames:
    """Stage 1 on every session of the panels (date x symbol)."""

    liquid: pd.DataFrame  # bool
    avg_traded_value: pd.DataFrame
    adr_pct: pd.DataFrame


def liquidity_frames(panels: Panels, lcfg: LiquidityConfig) -> LiquidityFrames:
    """Stage 1 liquidity filter, evaluated on every session:

    liquid_t = close_t >= min_price
               AND mean(close x volume over the last `window` bars) >= min_avg_traded_value
               AND ADR%(last `window` bars) >= min_adr_pct
    A missing value (no bar, or less than a full window) compares False, so it is not liquid.
    """
    n = lcfg.window
    avg_tv = ind.avg_traded_value(panels.close, panels.volume, n)
    adr = ind.adr_pct(panels.high, panels.low, n)
    liquid = (
        (panels.close >= lcfg.min_price)
        & (avg_tv >= lcfg.min_avg_traded_value)
        & (adr >= lcfg.min_adr_pct)
    )
    return LiquidityFrames(liquid=liquid, avg_traded_value=avg_tv, adr_pct=adr)


def liquidity_snapshot(panels: Panels, lcfg: LiquidityConfig) -> pd.DataFrame:
    """Stage 1 on the as-of (last) session, one row per symbol that has a bar that day.
    Columns: close, avg_traded_value, adr_pct, liquid."""
    p = panels.tail(lcfg.window + 1)
    liq = liquidity_frames(p, lcfg)
    last_close = p.close.iloc[-1]
    snap = pd.DataFrame(
        {
            "close": last_close,
            "avg_traded_value": liq.avg_traded_value.iloc[-1],
            "adr_pct": liq.adr_pct.iloc[-1],
            "liquid": liq.liquid.iloc[-1].astype(bool),
        }
    )
    return snap[last_close.notna()]


@dataclass
class LeaderFrames:
    """Stage 2 on every session of the panels, per window name."""

    returns: dict[str, pd.DataFrame]
    passed: dict[str, pd.DataFrame]  # bool; liquid that day AND passing the window
    rs: dict[str, pd.DataFrame]  # rank mode only

    def any_window(self) -> pd.DataFrame:
        """True where the symbol passed at least one window that session."""
        frames = list(self.passed.values())
        out = frames[0].copy()
        for f in frames[1:]:
            out |= f
        return out


def leader_frames(
    panels: Panels,
    liq: LiquidityFrames,
    leaders: LeadersConfig,
    mode: Literal["fixed", "rank"],
) -> LeaderFrames:
    """Stage 2 momentum leaders among liquid symbols, for every window and every session.

    fixed mode: pass_w,t = close_t >= min_multiple_w x close_(t - lookback_w)
                (e.g. 1m: close >= 1.25 x close 22 bars ago, i.e. a return of at least +25%)
    rank mode:  pass_w,t = RS percentile of return_w,t > 100 - rank_top_pct, ranked each session
                only among symbols liquid that session (the top 7% by default)
    A symbol without enough history for a window (e.g. a recent IPO for 6m) just fails that window.
    """
    returns, passed, rs_all = {}, {}, {}
    close = panels.close
    for name, w in leaders.windows.items():
        then = close.shift(w.lookback)
        ret = close / then - 1  # = indicators.pct_return(close, lookback)
        returns[name] = ret
        if mode == "fixed":
            ok = close >= w.min_multiple * then
        else:
            rs = ind.rs_rank_pct(ret.where(liq.liquid))
            rs_all[name] = rs
            ok = rs > 100 - leaders.rank_top_pct
        passed[name] = ok & liq.liquid
    return LeaderFrames(returns=returns, passed=passed, rs=rs_all)


def momentum_leaders(
    panels: Panels,
    lcfg: LiquidityConfig,
    leaders: LeadersConfig,
    mode: Literal["fixed", "rank"] | None = None,
) -> LeadersResult:
    """Stage 2 leaders on the as-of (last) session. See leader_frames for the rules."""
    if panels.close.empty:
        raise ValueError("no price data")
    mode = mode or leaders.mode
    max_lb = max(w.lookback for w in leaders.windows.values())
    p = panels.tail(max_lb + lcfg.window + 1)
    as_of = p.dates[-1]
    liq = liquidity_frames(p, lcfg)
    lf = leader_frames(p, liq, leaders, mode)

    last_close = p.close.iloc[-1]
    active = last_close.notna()
    liquid_now = liq.liquid.iloc[-1] & active
    liquid = liquid_now.index[liquid_now]
    out = pd.DataFrame({"close": last_close[liquid]})

    counts: dict[str, int] = {}
    for name in leaders.windows:
        out[f"ret_{name}"] = lf.returns[name].iloc[-1][liquid]
        if mode == "rank":
            out[f"rs_{name}"] = lf.rs[name].iloc[-1][liquid]
        out[f"pass_{name}"] = lf.passed[name].iloc[-1][liquid].astype(bool)
        counts[name] = int(out[f"pass_{name}"].sum())

    # Data-quality flag. Largest one-day move inside the longest window, as a multiple >= 1:
    #   max_day_move = max over days of max(close_t / close_{t-1}, close_{t-1} / close_t)
    # A 10x overnight jump is either a genuine squeeze or a reverse split the data source has not
    # adjusted for yet (price x k, share volume / k). Either way the return is worth a chart check.
    out["max_day_move"] = max_day_move(p.close.iloc[-(max_lb + 1):])[liquid]
    out["flag_day_move"] = (out["max_day_move"] >= leaders.flag_day_move_multiple).fillna(False).astype(bool)

    out["adr_pct"] = liq.adr_pct.iloc[-1][liquid]
    out["avg_traded_value"] = liq.avg_traded_value.iloc[-1][liquid]
    pass_cols = [f"pass_{n}" for n in leaders.windows]
    out["n_passed"] = out[pass_cols].sum(axis=1).astype(int)
    out = out[out["n_passed"] > 0]
    out = out.sort_values(["n_passed", f"ret_{leaders.sort_by}"], ascending=[False, False], kind="stable")
    out.index.name = "symbol"

    return LeadersResult(
        as_of=as_of,
        mode=mode,
        table=out,
        counts=counts,
        active_count=int(active.sum()),
        stale_count=int((~active).sum()),
        liquid_count=len(liquid),
        windows=list(leaders.windows),
    )


def max_day_move(close: pd.DataFrame) -> pd.Series:
    """Per symbol, the largest one-day move in `close` as a multiple >= 1 (up or down)."""
    ratio = close / close.shift(1)
    return pd.concat([ratio.max(), 1 / ratio.min()], axis=1).max(axis=1)
