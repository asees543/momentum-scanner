"""Data-quality report (SPEC Phase 4: flag splits, bonuses, missing days, circuit-limit stocks).

Looks at the last `window` sessions (default: the longest leaders window, 126 = 6M) of the panels and
reports, per symbol:

* jumps       one-day moves of at least flag_day_move_multiple (up or down). For each: the price
              ratio close/prev_close, the nearest "clean" corporate-action ratio (1:1 bonus = 1/2,
              1:10 split = 1/10, 1:10 reverse split = 10, ...) when within 6%, and the volume change
              (mean volume of the 5 sessions after / the 5 before). An unadjusted split or bonus
              shows price x k with volume / k, so price_ratio x vol_change ~ 1.
              Optionally, Yahoo's own split list is checked for that date.
* missing     sessions inside the symbol's own history in the window with no bar (gaps).
* frozen      the longest streak of circuit-locked sessions (indicators.circuit_locked; NSE bands).
* zero_volume sessions with a bar but zero volume.
* stale       symbols with no bar on the last session.
Rows are flagged `liquid` when the symbol passed Stage 1 on any session of the window, since
problems on illiquid names never reach the scans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from scanner import indicators as ind
from scanner.config import MarketConfig
from scanner.data.base import Panels
from scanner.scans import liquidity_frames

# Corporate-action price ratios (new price / old price) and how they read.
CLEAN_RATIOS: dict[float, str] = {
    1 / 2: "1/2 (1:1 bonus or 1:2 split)",
    2 / 3: "2/3 (1:2 bonus)",
    1 / 3: "1/3 (2:1 bonus or 1:3 split)",
    1 / 4: "1/4 (3:1 bonus or 1:4 split)",
    1 / 5: "1/5 (4:1 bonus or 1:5 split)",
    1 / 10: "1/10 (1:10 split)",
    2.0: "2 (1:2 reverse split)",
    5.0: "5 (1:5 reverse split)",
    10.0: "10 (1:10 reverse split)",
    20.0: "20 (1:20 reverse split)",
}
RATIO_TOLERANCE = 0.06

SplitLookup = Callable[[str], "list[pd.Timestamp]"]  # symbol -> dates Yahoo recorded a split/bonus


def nearest_clean_ratio(ratio: float) -> str:
    """Label of the corporate-action ratio within 6% of `ratio`, else ''."""
    if not ratio or not math.isfinite(ratio):
        return ""
    for k, label in CLEAN_RATIOS.items():
        if abs(ratio / k - 1) <= RATIO_TOLERANCE:
            return label
    return ""


@dataclass
class QualityReport:
    market: str
    as_of: pd.Timestamp
    window: int
    symbols: int
    jumps: pd.DataFrame  # symbol, date, prev_close, close, price_ratio, clean_ratio, vol_change, liquid, yahoo_split
    missing: pd.DataFrame  # symbol, missing_sessions, first_gap, liquid
    frozen: pd.DataFrame  # symbol, longest_streak, frozen_days, last_frozen, liquid
    zero_volume: pd.DataFrame  # symbol, zero_volume_days, liquid
    stale: pd.DataFrame  # symbol, last_bar, liquid


def quality_report(
    panels: Panels,
    cfg: MarketConfig,
    window: int | None = None,
    split_lookup: SplitLookup | None = None,
    max_split_lookups: int = 30,
) -> QualityReport:
    max_lb = max(w.lookback for w in cfg.leaders.windows.values())
    window = window or max_lb
    lcfg = cfg.liquidity
    p = panels.tail(window + lcfg.window + 1)
    liq = liquidity_frames(p, lcfg)
    rows = p.dates[-window:]
    close, high, low, vol = (getattr(p, f).loc[rows] for f in ("close", "high", "low", "volume"))
    ever_liquid = liq.liquid.loc[rows].any()

    # --- jumps: one-day moves of at least the flag multiple ---------------------------------
    full_close = p.close
    ratio = (full_close / full_close.shift(1)).loc[rows]
    m = cfg.leaders.flag_day_move_multiple
    hits = ratio.stack()
    hits = hits[(hits >= m) | (hits <= 1 / m)]
    jump_rows = []
    v_all = p.volume
    for (date, sym), r in hits.items():
        pos = v_all.index.get_loc(date)
        before = v_all[sym].iloc[max(0, pos - 5) : pos].mean()
        after = v_all[sym].iloc[pos + 1 : pos + 6].mean()
        jump_rows.append(
            {
                "symbol": sym,
                "date": date,
                "prev_close": float(full_close[sym].iloc[pos - 1]),
                "close": float(full_close[sym].iloc[pos]),
                "price_ratio": float(r),
                "clean_ratio": nearest_clean_ratio(float(r)),
                "vol_change": float(after / before) if before and before > 0 and not np.isnan(after) else np.nan,
                "liquid": bool(ever_liquid.get(sym, False)),
            }
        )
    jumps = pd.DataFrame(jump_rows, columns=["symbol", "date", "prev_close", "close", "price_ratio", "clean_ratio",
                                             "vol_change", "liquid"])
    jumps["yahoo_split"] = ""
    if split_lookup is not None and len(jumps):
        # Check Yahoo's split list for liquid names first (one request per symbol).
        order = jumps.sort_values(["liquid", "symbol"], ascending=[False, True])["symbol"].drop_duplicates()
        for sym in order.head(max_split_lookups):
            try:
                dates = {pd.Timestamp(d).normalize().tz_localize(None) for d in split_lookup(sym)}
            except Exception:
                continue
            idx = jumps.index[jumps["symbol"] == sym]
            jumps.loc[idx, "yahoo_split"] = [
                "yes" if pd.Timestamp(d).normalize() in dates else "no" for d in jumps.loc[idx, "date"]
            ]
    jumps = jumps.sort_values(["liquid", "date"], ascending=[False, False]).reset_index(drop=True)

    # --- missing sessions inside each symbol's own history ----------------------------------
    has = close.notna()
    started = has.cummax()  # True from the symbol's first bar in the window onwards
    ended = has[::-1].cummax()[::-1]  # True up to its last bar
    gap = started & ended & ~has
    n_missing = gap.sum()
    first_gap = gap.idxmax().where(n_missing > 0)
    missing = pd.DataFrame({"missing_sessions": n_missing, "first_gap": first_gap, "liquid": ever_liquid})
    missing = missing[missing["missing_sessions"] > 0].rename_axis("symbol").reset_index()
    missing = missing.sort_values(["liquid", "missing_sessions"], ascending=[False, False]).reset_index(drop=True)

    # --- frozen-price (circuit-locked) streaks ------------------------------------------------
    locked = ind.circuit_locked(
        p.high, p.low, p.close, lcfg.circuit_lock_min_move_pct, lcfg.circuit_lock_max_range_pct
    ).loc[rows]
    streak = ind.run_length(locked)
    frozen = pd.DataFrame(
        {
            "longest_streak": streak.max(),
            "frozen_days": locked.sum(),
            "last_frozen": locked[::-1].idxmax().where(locked.any()),
            "liquid": ever_liquid,
        }
    )
    frozen = frozen[frozen["frozen_days"] > 0].rename_axis("symbol").reset_index()
    frozen = frozen.sort_values(["longest_streak", "frozen_days"], ascending=False).reset_index(drop=True)

    # --- zero volume and stale ------------------------------------------------------------------
    zero = ((vol == 0) & has).sum()
    zero_volume = pd.DataFrame({"zero_volume_days": zero, "liquid": ever_liquid})
    zero_volume = zero_volume[zero_volume["zero_volume_days"] > 0].rename_axis("symbol").reset_index()
    zero_volume = zero_volume.sort_values("zero_volume_days", ascending=False).reset_index(drop=True)

    last_bar = panels.close.apply(lambda s: s.last_valid_index())
    stale_mask = panels.close.iloc[-1].isna()
    stale = pd.DataFrame(
        {"last_bar": last_bar[stale_mask], "liquid": ever_liquid.reindex(last_bar[stale_mask].index).fillna(False)}
    ).rename_axis("symbol").reset_index()

    return QualityReport(
        market=cfg.market,
        as_of=panels.dates[-1],
        window=window,
        symbols=panels.close.shape[1],
        jumps=jumps,
        missing=missing,
        frozen=frozen,
        zero_volume=zero_volume,
        stale=stale,
    )


def yahoo_split_dates(symbol: str) -> list[pd.Timestamp]:
    """Dates on which Yahoo recorded a split or bonus for `symbol` (one network request)."""
    import logging

    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    return list(yf.Ticker(symbol).splits.index)
