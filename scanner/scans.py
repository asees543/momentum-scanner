"""Market-agnostic scans. They take price panels + config and never know which market they run on.

Stage 1 liquidity filter, Stage 2 momentum leaders, Stage 3 consolidation setup (watchlist) and
Stage 4 breakout trigger (SPEC.md).

Rules are computed as date x symbol frames, then read on a single "as-of" session: the last row
of the panels passed in. Pass `panels.as_of(date)` to evaluate a past date with no lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from scanner import indicators as ind
from scanner.config import BreakoutConfig, LeadersConfig, LiquidityConfig, MarketConfig, SetupConfig
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
    liq = liquidity_frames(p, lcfg)
    return leaders_from_frames(p, liq, leader_frames(p, liq, leaders, mode), leaders, mode)


def leaders_from_frames(
    p: Panels, liq: LiquidityFrames, lf: LeaderFrames, leaders: LeadersConfig, mode: Literal["fixed", "rank"]
) -> LeadersResult:
    """Read the leaders table off the last row of precomputed frames."""
    max_lb = max(w.lookback for w in leaders.windows.values())
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
        as_of=p.dates[-1],
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


# --- Stage 3: consolidation setup -------------------------------------------------------------


@dataclass
class SetupFrames:
    """Stage 3 on every session (date x symbol)."""

    qualified: pd.DataFrame  # bool: every rule below holds
    trigger: pd.DataFrame  # highest high of the last `trigger_bars` bars
    conditions: dict[str, pd.DataFrame]  # each rule on its own (bool), for inspection and tests
    leader_recent: dict[str, pd.DataFrame]  # per window: leader on any of the last N sessions
    avg_volume_prior: pd.DataFrame  # mean volume of the `volume_avg_window` sessions before t


def setup_frames(panels: Panels, liq: LiquidityFrames, lf: LeaderFrames, scfg: SetupConfig) -> SetupFrames:
    """Stage 3 "orderly consolidation" rules (SPEC), evaluated on every session t:

    leader     : passed a leaders window (and was liquid) on any of the last N sessions (N = 20)
    trend      : close_t > SMA50_t
    sma_order  : SMA10_t > SMA20_t
    near_high  : close_t >= (1 - 15/100) x highest high of the last 20 bars
    higher_lows: lowest low of bars t-4..t  >  lowest low of bars t-9..t-5
    tight      : mean daily range% over the last k bars < 1.0 x ADR%_t, for at least one k in 3..5
                 (daily range% = 100 x (high/low - 1), the same quantity ADR% averages)
    dryup      : volume_t <= 0.70 x mean volume of the 20 sessions before t

    trigger_t = highest high of the last 10 bars (bars t-9..t).
    """
    c, h, lo, v = panels.close, panels.high, panels.low, panels.volume
    n_lead = scfg.leader_lookback_sessions

    def recent(mask: pd.DataFrame) -> pd.DataFrame:
        # True if the mask was True on any of the last n_lead sessions (t included).
        return mask.astype(float).rolling(n_lead, min_periods=1).max() > 0

    leader_recent = {name: recent(m) for name, m in lf.passed.items()}
    leader_any = recent(lf.any_window())

    hl = scfg.higher_lows_bars
    rng = ind.daily_range_pct(h, lo)
    tight = pd.DataFrame(False, index=c.index, columns=c.columns)
    for k in range(scfg.tight_min_days, scfg.tight_max_days + 1):
        tight |= rng.rolling(k, min_periods=k).mean() < scfg.tight_vs_adr * liq.adr_pct
    avg_vol = ind.prior_mean(v, scfg.volume_avg_window)

    conditions = {
        "leader": leader_any,
        "trend": c > ind.sma(c, scfg.sma_trend),
        "sma_order": ind.sma(c, scfg.sma_fast) > ind.sma(c, scfg.sma_slow),
        "near_high": c >= (1 - scfg.max_below_high_pct / 100) * ind.highest(h, scfg.high_window),
        "higher_lows": ind.lowest(lo, hl) > ind.lowest(lo, hl).shift(hl),
        "tight": tight,
        "dryup": v <= scfg.dryup_max_volume_ratio * avg_vol,
    }
    qualified = conditions["leader"].copy()
    for name, cond in conditions.items():
        if name != "leader":
            qualified &= cond
    return SetupFrames(
        qualified=qualified,
        trigger=ind.highest(h, scfg.trigger_bars),
        conditions=conditions,
        leader_recent=leader_recent,
        avg_volume_prior=avg_vol,
    )


def _latest_setup(sf: SetupFrames, close: pd.DataFrame, end: int, memory: int) -> pd.DataFrame:
    """For each symbol, the most recent qualifying setup day s among the `memory` sessions ending at
    row position `end` (inclusive), and whether any close after s (up to `end`) was above its trigger.

    Returns columns: has_setup (bool), age (sessions from s to `end`), trigger (trigger on day s),
    closed_above (bool). Rows are symbols.
    """
    lo = max(0, end - memory + 1)
    q = sf.qualified.iloc[lo : end + 1].to_numpy(dtype=bool)
    trig = sf.trigger.iloc[lo : end + 1].to_numpy(dtype=float)
    cl = close.iloc[lo : end + 1].to_numpy(dtype=float)
    m = q.shape[0]
    has = q.any(axis=0)
    last = m - 1 - np.argmax(q[::-1], axis=0)  # row of the most recent True (garbage if not has)
    cols = np.arange(q.shape[1])
    trigger = np.where(has, trig[last, cols], np.nan)
    after = np.arange(m)[:, None] > last[None, :]
    with np.errstate(invalid="ignore"):
        closed_above = ((cl > trigger[None, :]) & after).any(axis=0) & has
    return pd.DataFrame(
        {"has_setup": has, "age": np.where(has, m - 1 - last, -1), "trigger": trigger, "closed_above": closed_above},
        index=sf.qualified.columns,
    )


@dataclass
class WatchlistResult:
    as_of: pd.Timestamp
    # One row per watchlist symbol, closest to its trigger first. Columns: close, trigger,
    # to_trigger_pct, stop, risk_pct, risk_too_wide, adr_pct, vol_ratio, setup_age, led_<w> per
    # window, avg_traded_value, max_day_move, flag_day_move.
    table: pd.DataFrame
    qualified_today: int  # names whose setup qualifies on the as-of session itself
    windows: list[str] = field(default_factory=list)


@dataclass
class BreakoutResult:
    as_of: pd.Timestamp
    # One row per breakout (accepted and rejected). Columns: close, trigger, stop, risk_pct, adr_pct,
    # vol_ratio, status ("ok" / "too wide"), setup_age, led_<w>, avg_traded_value, max_day_move,
    # flag_day_move.
    table: pd.DataFrame
    ok_count: int
    rejected_count: int
    windows: list[str] = field(default_factory=list)


@dataclass
class ScanResults:
    leaders: LeadersResult
    watchlist: WatchlistResult
    breakouts: BreakoutResult


def _rows_needed(cfg: MarketConfig) -> int:
    """Sessions of history the full scan reads (so it can work on a short tail of the panels)."""
    s = cfg.setup
    max_lb = max(w.lookback for w in cfg.leaders.windows.values())
    per_day = max(
        max_lb + cfg.liquidity.window,
        s.sma_trend,
        s.high_window,
        2 * s.higher_lows_bars,
        s.volume_avg_window + 1,
        s.trigger_bars,
        s.tight_max_days + cfg.liquidity.window,
    )
    return per_day + s.leader_lookback_sessions + cfg.breakout.watchlist_memory_sessions + 2


def watchlist_from_frames(
    p: Panels, liq: LiquidityFrames, sf: SetupFrames, cfg: MarketConfig, max_lb: int
) -> WatchlistResult:
    """Watchlist on the as-of session t = names that could break out on the NEXT session:
    their setup qualified on one of the last M sessions (t included; M = watchlist_memory_sessions)
    and no close since then has been above that day's trigger.

    stop      = low_t (stand-in for the breakout-day low, which SPEC uses as the real stop)
    risk%     = 100 x (trigger - stop) / trigger; risk_too_wide if risk% > max_risk_vs_adr x ADR%_t
    to_trigger% = 100 x (trigger / close_t - 1)
    vol_ratio = volume_t / mean volume of the 20 sessions before t
    """
    t = len(p.dates) - 1
    latest = _latest_setup(sf, p.close, t, cfg.breakout.watchlist_memory_sessions)
    keep = latest["has_setup"] & ~latest["closed_above"]
    syms = latest.index[keep]
    close = p.close.iloc[-1][syms]
    trig = latest.loc[syms, "trigger"]
    stop = p.low.iloc[-1][syms]
    adr = liq.adr_pct.iloc[-1][syms]
    out = pd.DataFrame(
        {
            "close": close,
            "trigger": trig,
            "to_trigger_pct": 100 * (trig / close - 1),
            "stop": stop,
            "risk_pct": 100 * (trig - stop) / trig,
            "adr_pct": adr,
            "vol_ratio": p.volume.iloc[-1][syms] / sf.avg_volume_prior.iloc[-1][syms],
            "setup_age": latest.loc[syms, "age"].astype(int),
        }
    )
    out["risk_too_wide"] = (out["risk_pct"] > cfg.breakout.max_risk_vs_adr * out["adr_pct"]).astype(bool)
    windows = list(cfg.leaders.windows)
    for w in windows:
        out[f"led_{w}"] = sf.leader_recent[w].iloc[-1][syms].astype(bool)
    out["avg_traded_value"] = liq.avg_traded_value.iloc[-1][syms]
    out["max_day_move"] = max_day_move(p.close.iloc[-(max_lb + 1):])[syms]
    out["flag_day_move"] = (out["max_day_move"] >= cfg.leaders.flag_day_move_multiple).fillna(False).astype(bool)
    out = out.sort_values(["to_trigger_pct", "risk_pct"], kind="stable")
    out.index.name = "symbol"
    return WatchlistResult(
        as_of=p.dates[-1],
        table=out,
        qualified_today=int(sf.qualified.iloc[-1].sum()),
        windows=windows,
    )


# --- Stage 4: breakout trigger ----------------------------------------------------------------


def breakouts_from_frames(
    p: Panels, liq: LiquidityFrames, sf: SetupFrames, cfg: MarketConfig, max_lb: int
) -> BreakoutResult:
    """Breakouts on the as-of session t (SPEC Stage 4):

    eligible  : setup qualified on some day s in t-M..t-1 (the most recent such s is used,
                M = watchlist_memory_sessions) and no close on s+1..t-1 was above trigger_s,
                so only the FIRST close above the trigger counts
    breakout  : close_t > trigger_s  AND  volume_t >= 1.5 x mean volume of the 20 sessions before t
    stop      = low_t (the breakout day's low)
    risk%     = 100 x (close_t - stop) / close_t
    status    = "too wide" if risk% > max_risk_vs_adr x ADR%_(t-1), else "ok". ADR is taken from the
                day before so the breakout bar's own wide range cannot loosen the rule.
    """
    windows = list(cfg.leaders.windows)
    t = len(p.dates) - 1
    bcfg = cfg.breakout
    if t < 1:
        empty = pd.DataFrame(columns=["close", "trigger", "stop", "risk_pct", "adr_pct", "vol_ratio", "status"])
        return BreakoutResult(as_of=p.dates[-1], table=empty, ok_count=0, rejected_count=0, windows=windows)

    latest = _latest_setup(sf, p.close, t - 1, bcfg.watchlist_memory_sessions)
    eligible = latest["has_setup"] & ~latest["closed_above"]
    close_t = p.close.iloc[-1]
    vol_t = p.volume.iloc[-1]
    avg_vol = sf.avg_volume_prior.iloc[-1]
    trig = latest["trigger"]
    hit = eligible & (close_t > trig) & (vol_t >= bcfg.min_volume_ratio * avg_vol)
    syms = hit.index[hit.fillna(False).astype(bool)]

    stop = p.low.iloc[-1][syms]
    out = pd.DataFrame(
        {
            "close": close_t[syms],
            "trigger": trig[syms],
            "stop": stop,
            "risk_pct": 100 * (close_t[syms] - stop) / close_t[syms],
            "adr_pct": liq.adr_pct.iloc[-2][syms],
            "vol_ratio": vol_t[syms] / avg_vol[syms],
            "setup_age": (latest.loc[syms, "age"] + 1).astype(int),  # sessions from setup day to breakout
        }
    )
    too_wide = out["risk_pct"] > bcfg.max_risk_vs_adr * out["adr_pct"]
    out["status"] = np.where(too_wide, "too wide", "ok")
    for w in windows:
        out[f"led_{w}"] = sf.leader_recent[w].iloc[-1][syms].astype(bool)
    out["avg_traded_value"] = liq.avg_traded_value.iloc[-1][syms]
    out["max_day_move"] = max_day_move(p.close.iloc[-(max_lb + 1):])[syms]
    out["flag_day_move"] = (out["max_day_move"] >= cfg.leaders.flag_day_move_multiple).fillna(False).astype(bool)
    out = out.sort_values(["status", "vol_ratio"], ascending=[True, False], kind="stable")  # "ok" first
    out.index.name = "symbol"
    return BreakoutResult(
        as_of=p.dates[-1],
        table=out,
        ok_count=int((out["status"] == "ok").sum()),
        rejected_count=int((out["status"] != "ok").sum()),
        windows=windows,
    )


def run_scans(panels: Panels, cfg: MarketConfig, mode: Literal["fixed", "rank"] | None = None) -> ScanResults:
    """All stages on the as-of (last) session of `panels`, sharing one set of frames."""
    if panels.close.empty:
        raise ValueError("no price data")
    mode = mode or cfg.leaders.mode
    max_lb = max(w.lookback for w in cfg.leaders.windows.values())
    p = panels.tail(_rows_needed(cfg))
    liq = liquidity_frames(p, cfg.liquidity)
    lf = leader_frames(p, liq, cfg.leaders, mode)
    sf = setup_frames(p, liq, lf, cfg.setup)
    return ScanResults(
        leaders=leaders_from_frames(p, liq, lf, cfg.leaders, mode),
        watchlist=watchlist_from_frames(p, liq, sf, cfg, max_lb),
        breakouts=breakouts_from_frames(p, liq, sf, cfg, max_lb),
    )
