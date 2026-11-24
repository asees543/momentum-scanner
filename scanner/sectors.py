"""Sector and industry strength: where in the market are the leaders?

Two parts.

1. The map. Every liquid name is looked up once on Yahoo (`yf.Ticker(symbol).info`: "sector" and "industry", the same
   classification for the US and for NSE) and stored in reference/sectors/<market>.csv, which is committed. The daily run
   only looks up names the map does not have yet (at most `sectors.max_new_per_run`, `--update-sectors`), so a Yahoo
   outage or rate limit just leaves some names unclassified for a day; it never stops a scan. A lookup that returns no
   sector is stored as status "none" and retried after `retry_days`.

       uv run python -m scanner.sectors --market us        # fill or refresh the map from the cached prices

2. The strength. For each sector (and industry) among the liquid names:
     leader_pct   = 100 x (names that are leaders in any window today) / (liquid names in the group)
     above_pct    = 100 x (names with close > SMA(sectors.sma)) / (liquid names in the group)
     median_ret   = median of the sort window's return (3 months) over the group's names
     d_above      = above_pct now minus above_pct `change_sessions` sessions ago  (money flowing in, or out)
   A group with at least `min_names_*` names is scored as the average of the percentile ranks (0-100, among the ranked
   groups) of leader_pct, above_pct and median_ret, so every measure counts the same. It is a ranking, not a forecast.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from scanner import indicators as ind
from scanner.config import PROJECT_ROOT, MarketConfig, SectorConfig, load_config

REFERENCE_DIR = PROJECT_ROOT / "reference" / "sectors"
MAP_COLUMNS = ["symbol", "sector", "industry", "as_of", "status"]
Fetch = Callable[[str], "tuple[str, str] | None"]


# --- the map --------------------------------------------------------------------------------------------


def map_path(market: str, directory: Path = REFERENCE_DIR) -> Path:
    return Path(directory) / f"{market}.csv"


def load_map(market: str, directory: Path = REFERENCE_DIR) -> pd.DataFrame:
    """The stored map, indexed by the symbol used in the price panels. Empty (right columns) when there is no file."""
    path = map_path(market, directory)
    if not path.exists():
        return pd.DataFrame(columns=MAP_COLUMNS[1:], index=pd.Index([], name="symbol"))
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.set_index("symbol")[MAP_COLUMNS[1:]]


def save_map(df: pd.DataFrame, market: str, directory: Path = REFERENCE_DIR) -> Path:
    """Write the map sorted by symbol (so the committed file only changes where the map did)."""
    path = map_path(market, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index().rename(columns={"index": "symbol"}).sort_values("symbol")[MAP_COLUMNS]
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def yahoo_fetch(symbol: str) -> tuple[str, str] | None:
    """(sector, industry) from Yahoo, or None when Yahoo has no sector for it (funds, odd listings). Network errors raise."""
    import yfinance as yf

    info = yf.Ticker(symbol).info
    sector = (info.get("sector") or "").strip()
    return (sector, (info.get("industry") or "").strip()) if sector else None


@dataclass
class UpdateStats:
    looked_up: int = 0
    classified: int = 0
    none: int = 0
    errors: int = 0
    stopped_early: bool = False
    skipped_for_cap: int = 0


def needs_lookup(row: pd.Series | None, today: date, cfg: SectorConfig) -> bool:
    """Not in the map yet; or classified more than stale_days ago; or without a sector for more than retry_days."""
    if row is None:
        return True
    try:
        age = (today - date.fromisoformat(row["as_of"])).days
    except (ValueError, TypeError):
        return True
    return age >= (cfg.stale_days if row["status"] == "ok" else cfg.retry_days)


def update_map(symbols: list[str], current: pd.DataFrame, cfg: SectorConfig, fetch: Fetch = yahoo_fetch,
               today: date | None = None, pause: float = 0.0, max_consecutive_errors: int = 8,
               ) -> tuple[pd.DataFrame, UpdateStats]:
    """Look up the symbols the map lacks (in the order given, so put the most important first), at most max_new_per_run.

    A network error leaves the symbol alone (it is tried again next run); `max_consecutive_errors` in a row means
    Yahoo is refusing us (rate limit), so the update stops for this run. Returns the new map and what happened."""
    today = today or date.today()
    out = current.copy()
    stats = UpdateStats()
    todo = [s for s in dict.fromkeys(symbols) if needs_lookup(out.loc[s] if s in out.index else None, today, cfg)]
    stats.skipped_for_cap = max(0, len(todo) - cfg.max_new_per_run)
    streak = 0
    for s in todo[: cfg.max_new_per_run]:
        try:
            got = fetch(s)
        except Exception:  # noqa: BLE001 - any failure of the source is the same to us: try again next run
            stats.errors += 1
            streak += 1
            if streak >= max_consecutive_errors:
                stats.stopped_early = True
                break
            continue
        streak = 0
        stats.looked_up += 1
        sector, industry = got if got else ("", "")
        out.loc[s, ["sector", "industry", "as_of", "status"]] = [sector, industry, today.isoformat(), "ok" if got else "none"]
        stats.classified += bool(got)
        stats.none += not got
        if pause:
            time.sleep(pause)
    out.index.name = "symbol"
    return out, stats


# --- the strength ---------------------------------------------------------------------------------------


@dataclass
class SectorResult:
    as_of: pd.Timestamp
    liquid: int  # liquid (Stage 1) names on the as-of session
    classified: int  # of those, how many have a sector in the map
    # One row per ranked group, strongest first. Columns: names, leaders, leader_pct, above_pct, d_above, median_ret,
    # watchlist, trigger_bars, score. Index = the group's name.
    sectors: pd.DataFrame
    industries: pd.DataFrame  # the same, for industries; the index is the industry, `sector` column says which sector
    sector_of: dict[str, str] = field(default_factory=dict)  # symbol -> sector, for the classified liquid names

    @property
    def coverage(self) -> float:
        return self.classified / self.liquid if self.liquid else float("nan")


def _score(df: pd.DataFrame) -> pd.Series:
    """Average of the percentile ranks (0-100) of the three measures; rank 100 = best in the table."""
    parts = [df[c].rank(pct=True) * 100 for c in ("leader_pct", "above_pct", "median_ret")]
    return sum(parts) / 3


def sector_strength(panels, cfg: MarketConfig, sector_map: pd.DataFrame, liquid: pd.Index, leaders: pd.Index,
                    watchlist: pd.Index, triggers: pd.Index) -> SectorResult:
    """Rank the sectors and industries of the liquid names (see the module docstring for the measures).

    `liquid` = the Stage 1 names on the last session, `leaders` = names that are leaders in any window today,
    `watchlist` / `triggers` = names on the watchlist / with a trigger bar today (counted, not scored)."""
    sc = cfg.sectors
    window = cfg.leaders.windows[cfg.leaders.sort_by].lookback
    p = panels.tail(max(sc.sma, window) + sc.change_sessions + 2)
    c = p.close
    above_now = (c.iloc[-1] > ind.sma(c, sc.sma).iloc[-1])
    above_then = (c.iloc[-1 - sc.change_sessions] > ind.sma(c, sc.sma).iloc[-1 - sc.change_sessions])
    ret = ind.pct_return(c, window).iloc[-1]

    names = pd.Index([s for s in liquid if s in sector_map.index and sector_map.loc[s, "sector"] != ""])
    info = sector_map.loc[names, ["sector", "industry"]]
    df = pd.DataFrame({
        "sector": info["sector"], "industry": info["industry"],
        "leader": names.isin(leaders), "above": above_now.reindex(names).fillna(False).astype(bool),
        "above_then": above_then.reindex(names).fillna(False).astype(bool), "ret": ret.reindex(names),
        "watch": names.isin(watchlist), "trigger": names.isin(triggers),
    }, index=names)

    def table(by: str, minimum: int) -> pd.DataFrame:
        g = df[df[by] != ""].groupby(by)
        out = pd.DataFrame({
            "names": g.size(),
            "leaders": g["leader"].sum().astype(int),
            "above_pct": 100 * g["above"].mean(),
            "d_above": 100 * (g["above"].mean() - g["above_then"].mean()),
            "median_ret": g["ret"].median(),
            "watchlist": g["watch"].sum().astype(int),
            "trigger_bars": g["trigger"].sum().astype(int),
        })
        out["leader_pct"] = 100 * out["leaders"] / out["names"]
        out = out[out["names"] >= minimum].copy()
        if out.empty:
            out["score"] = pd.Series(dtype=float)
            return out
        out["score"] = _score(out)
        cols = ["names", "leaders", "leader_pct", "above_pct", "d_above", "median_ret", "watchlist", "trigger_bars", "score"]
        return out[cols].sort_values(["score", "names"], ascending=[False, False], kind="stable")

    sectors = table("sector", sc.min_names_sector)
    industries = table("industry", sc.min_names_industry)
    industries.insert(0, "sector", df.drop_duplicates("industry").set_index("industry")["sector"].reindex(industries.index))
    return SectorResult(as_of=p.dates[-1], liquid=len(liquid), classified=len(names), sectors=sectors,
                        industries=industries, sector_of=info["sector"].to_dict())


# --- command line ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from scanner import scans
    from scanner.data.base import Panels
    from scanner.data.cache import PriceCache

    ap = argparse.ArgumentParser(prog="python -m scanner.sectors", description="Fill the sector map from Yahoo")
    ap.add_argument("--market", default="us", choices=["us", "india"])
    ap.add_argument("--max", type=int, help="look up at most this many names (default: sectors.max_new_per_run; 0 = no limit)")
    ap.add_argument("--all-liquid", action="store_true", help="every name ever liquid in the last 60 sessions, not only today's")
    args = ap.parse_args(argv)
    cfg = load_config(args.market)
    cache = PriceCache(cfg.data.cache_path())
    if not cache.prices_path.exists():
        print("No cached prices. Run `python -m scanner.run --market", args.market, "` first.", file=sys.stderr)
        return 1
    universe = pd.read_csv(cache.universe_path, dtype=str, keep_default_na=False)
    prices = cache.load()
    panels = Panels.from_long(prices[prices["symbol"].isin(universe["symbol"])])
    # Most traded first, so a cap or a rate limit cuts the least important names.
    snap = scans.liquidity_snapshot(panels, cfg.liquidity)
    liquid = snap[snap["liquid"]].sort_values("avg_traded_value", ascending=False).index.tolist()
    current = load_map(args.market)
    sc = cfg.sectors if args.max is None else cfg.sectors.model_copy(update={"max_new_per_run": args.max or 10**9})
    new, st = update_map(liquid, current, sc, pause=0.1)
    path = save_map(new, args.market)
    print(f"{args.market}: {len(liquid)} liquid names, {len(current)} in the map before. Looked up {st.looked_up} "
          f"({st.classified} with a sector, {st.none} without), {st.errors} errors"
          + (", stopped early (Yahoo refusing requests)" if st.stopped_early else "")
          + (f", {st.skipped_for_cap} left for the next run (cap)" if st.skipped_for_cap else "")
          + f". Wrote {path} ({len(new)} names).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
