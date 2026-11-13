"""Parquet price cache with incremental daily updates.

Layout (per market, under data/cache/<market>/, gitignored):
  prices.parquet   long format: symbol, date, open, high, low, close, volume
  meta.json        bookkeeping (last complete session fetched, timestamps)
  universe.csv     the filtered symbol list (written by the market loader)

Update rules
------------
* New symbol            -> download full history (from `history_start`).
* Known symbol          -> re-download only the last `overlap_bars` sessions plus anything newer.
                           Fresh rows replace cached rows for the same dates (Yahoo sometimes
                           revises the latest bar's volume after the fact).
* Split / revision trap -> after a split Yahoo re-adjusts ALL past prices, so the cached history no
                           longer lines up with new bars (a fake -50% or -90% day). On every update we
                           compare overlapping closes; if any differs by more than
                           `tolerance_pct` %, that symbol's full history is downloaded again.
                           A symbol whose cached data ends before the overlap window (so nothing can
                           be compared) is also re-downloaded in full.
* Failed download       -> cached rows are kept as they are; the symbol is reported.

Writes go to a temp file then os.replace(), so an interrupted run never leaves a corrupt cache.
During the first big download the cache is checkpointed every few batches, so a rerun resumes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from datetime import timedelta
from typing import Callable, Sequence

import pandas as pd

from scanner.data.base import PriceProvider, drop_incomplete_bars, empty_long, normalize_long

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]  # (phase, symbols_done, symbols_total)


class PriceCache:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.prices_path = self.dir / "prices.parquet"
        self.meta_path = self.dir / "meta.json"
        self.universe_path = self.dir / "universe.csv"

    def load(self) -> pd.DataFrame:
        if not self.prices_path.exists():
            return empty_long()
        return normalize_long(pd.read_parquet(self.prices_path))

    def save(self, df: pd.DataFrame) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.prices_path.with_suffix(".parquet.tmp")
        normalize_long(df).to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, self.prices_path)

    def read_meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text())

    def write_meta(self, meta: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
        os.replace(tmp, self.meta_path)


@dataclass
class UpdateReport:
    requested: int = 0
    new_symbols: int = 0
    incremental: int = 0
    full_refetch: list[str] = field(default_factory=list)  # split/revision detected or no overlap
    failed: list[str] = field(default_factory=list)  # nothing returned; old rows (if any) kept
    already_current: int = 0  # known symbols skipped: already checked against this cutoff
    cutoff: date | None = None


def trading_calendar(df: pd.DataFrame, min_coverage: float = 0.5) -> pd.DatetimeIndex:
    """Dates on which at least min_coverage x (median symbols per date) symbols have a bar.
    Same rule as Panels.from_long, so stray holiday bars from one ticker don't count as sessions."""
    if df.empty:
        return pd.DatetimeIndex([])
    counts = df.groupby("date")["symbol"].size()
    return pd.DatetimeIndex(counts.index[counts >= min_coverage * counts.median()]).sort_values()


def _collect(
    provider: PriceProvider,
    symbols: Sequence[str],
    start: date,
    end: date,
    phase: str,
    progress: ProgressFn | None,
    checkpoint: Callable[[pd.DataFrame], None] | None = None,
    checkpoint_every: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    done = 0
    total = len(symbols)
    for n, batch in enumerate(provider.fetch_batches(symbols, start, end), start=1):
        got_n = 0
        if not batch.prices.empty:
            frames.append(batch.prices)
            got_n = batch.prices["symbol"].nunique()
        failed.extend(batch.failed)
        done = min(total, done + got_n + len(batch.failed))
        if progress:
            progress(phase, done, total)
        if checkpoint and n % checkpoint_every == 0 and frames:
            checkpoint(pd.concat(frames, ignore_index=True))
    got = normalize_long(pd.concat(frames, ignore_index=True)) if frames else empty_long()
    got = drop_incomplete_bars(got, end)
    # A symbol counts as failed only if it never came back (the provider may retry it later).
    failed = sorted(set(failed) - set(got["symbol"].unique()))
    return got, failed


def _merge(base: pd.DataFrame, fresh: pd.DataFrame, replace_symbols: Sequence[str] = ()) -> pd.DataFrame:
    """Combine cached and freshly fetched rows. Fresh rows win on (symbol, date) clashes; symbols in
    `replace_symbols` have ALL their cached rows dropped first (full re-download)."""
    if len(replace_symbols):
        base = base[~base["symbol"].isin(list(replace_symbols))]
    if fresh.empty:
        return base
    if base.empty:
        return normalize_long(fresh)
    return normalize_long(pd.concat([base, fresh], ignore_index=True))


def find_mismatched(cached: pd.DataFrame, fresh: pd.DataFrame, tolerance_pct: float) -> list[str]:
    """Symbols whose freshly downloaded closes disagree with cached closes on overlapping dates by
    more than tolerance_pct percent:  |fresh / cached - 1| x 100 > tolerance_pct.
    Symbols with fresh rows but no overlapping date at all are also returned (cannot verify)."""
    if fresh.empty:
        return []
    m = fresh[["symbol", "date", "close"]].merge(
        cached[["symbol", "date", "close"]], on=["symbol", "date"], suffixes=("", "_cached")
    )
    diff_pct = (m["close"] / m["close_cached"] - 1).abs() * 100
    bad = set(m.loc[diff_pct > tolerance_pct, "symbol"])
    no_overlap = set(fresh["symbol"]) - set(m["symbol"])
    return sorted(bad | no_overlap)


def update_prices(
    cache: PriceCache,
    provider: PriceProvider,
    symbols: Sequence[str],
    *,
    history_start: date,
    cutoff: date,
    overlap_bars: int,
    tolerance_pct: float,
    refresh: bool = False,
    progress: ProgressFn | None = None,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, UpdateReport]:
    """Bring the cache up to `cutoff` (last complete session) for `symbols`; return the full cache.

    meta.json keeps, per symbol, the cutoff it was last successfully checked against, so running
    twice after the same close skips the download (unless refresh=True), and a symbol that failed
    is retried on the next run. Tracking this per symbol (not one flag for the whole cache) means a
    quick run on a few symbols never makes a later full run skip the rest.
    """
    symbols = list(dict.fromkeys(symbols))
    cutoff_iso = cutoff.isoformat()
    report = UpdateReport(requested=len(symbols), cutoff=cutoff)
    working = cache.load()
    meta = cache.read_meta()
    checked: dict[str, str] = dict(meta.get("checked", {}))

    have = set(working["symbol"].unique())
    new = [s for s in symbols if s not in have]
    existing = [s for s in symbols if s in have]
    report.new_symbols = len(new)

    # 1) Brand-new symbols: full history, checkpointed so an interrupted first run can resume.
    if new:
        def checkpoint(partial: pd.DataFrame) -> None:
            cache.save(_merge(working, partial))

        fresh, failed = _collect(provider, new, history_start, cutoff, "new symbols", progress, checkpoint)
        working = _merge(working, fresh)
        report.failed.extend(failed)
        checked.update({s: cutoff_iso for s in fresh["symbol"].unique()})
        cache.save(working)

    # 2) Known symbols: incremental with overlap check, skipping those already checked for this cutoff.
    todo = existing if refresh else [s for s in existing if checked.get(s) != cutoff_iso]
    report.already_current = len(existing) - len(todo)
    if todo:
        cal = trading_calendar(working)
        start_ts = cal[-overlap_bars] if len(cal) >= overlap_bars else cal[0]
        last_seen = working[working["symbol"].isin(todo)].groupby("symbol")["date"].max()
        stale = set(last_seen.index[last_seen < start_ts])
        incremental = [s for s in todo if s not in stale]
        report.incremental = len(incremental)

        fresh, failed = _collect(provider, incremental, start_ts.date(), cutoff, "update", progress)
        report.failed.extend(failed)
        mismatched = find_mismatched(working, fresh, tolerance_pct)
        fresh_ok = fresh[~fresh["symbol"].isin(mismatched)]
        working = _merge(working, fresh_ok)
        checked.update({s: cutoff_iso for s in fresh_ok["symbol"].unique()})

        full = sorted(stale | set(mismatched))
        if full:
            log.info("full re-download for %d symbols (split/revision/stale): %s", len(full), full[:20])
            refetched, failed_full = _collect(provider, full, history_start, cutoff, "re-download", progress)
            got = sorted(refetched["symbol"].unique())
            working = _merge(working, refetched, replace_symbols=got)
            report.full_refetch = got
            report.failed.extend(failed_full)
            checked.update({s: cutoff_iso for s in got})

    report.failed = sorted(set(report.failed))
    cache.save(working)
    meta.update(
        updated_at=(now or datetime.now(timezone.utc)).isoformat(),
        provider=getattr(provider, "name", "unknown"),
        history_start=history_start.isoformat(),
        last_cutoff=cutoff_iso,
        checked=checked,
    )
    cache.write_meta(meta)
    return working, report


UNIVERSE_COLUMNS = ["symbol", "source_symbol", "name", "exchange"]


def load_cached_universe(
    cache: PriceCache,
    max_age_hours: float,
    build: Callable[[], tuple[pd.DataFrame, int]],
    now: datetime,
    force: bool = False,
    label: str = "",
) -> pd.DataFrame:
    """Market universe (columns symbol, source_symbol, name, exchange), cached as universe.csv.

    `build()` downloads and filters the list, returning (universe, raw_row_count). It is called only
    when the cached copy is older than max_age_hours (or force=True). If it fails, the cached copy is
    used with a warning; with no cached copy the error propagates.
    """
    meta = cache.read_meta()
    fetched_at = meta.get("universe_fetched_at")
    fresh_enough = (
        fetched_at is not None
        and cache.universe_path.exists()
        and now - datetime.fromisoformat(fetched_at) < timedelta(hours=max_age_hours)
    )
    if fresh_enough and not force:
        return pd.read_csv(cache.universe_path, dtype=str, keep_default_na=False)
    try:
        uni, raw_count = build()
    except Exception as e:
        if cache.universe_path.exists():
            log.warning("universe download failed (%s); using cached list from %s", e, fetched_at)
            return pd.read_csv(cache.universe_path, dtype=str, keep_default_na=False)
        raise
    uni = uni[UNIVERSE_COLUMNS]
    cache.dir.mkdir(parents=True, exist_ok=True)
    uni.to_csv(cache.universe_path, index=False)
    meta = cache.read_meta()
    meta.update(universe_fetched_at=now.isoformat(), universe_raw_count=int(raw_count), universe_count=int(len(uni)))
    cache.write_meta(meta)
    log.info("%s universe: %d listed rows -> %d kept", label, raw_count, len(uni))
    return uni
