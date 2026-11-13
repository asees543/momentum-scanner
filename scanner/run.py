"""CLI: update market data and print the momentum-leaders scan.

    uv run python -m scanner.run --market us
    uv run python -m scanner.run --market us --window 1m --mode rank
    uv run python -m scanner.run --market us --no-update --as-of 2026-10-30
    uv run python -m scanner.run --market us --symbols NVDA,PLTR,HOOD     # quick dev run

Output is information, not investment advice.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date, datetime, timezone
from types import ModuleType

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from scanner.config import ConfigError, MarketConfig, load_config
from scanner.data import us_loader
from scanner.data.base import Panels, last_complete_session_date
from scanner.data.cache import PriceCache, UpdateReport, update_prices
from scanner.scans import LeadersResult, momentum_leaders

MARKETS: dict[str, ModuleType] = {"us": us_loader}
CURRENCY_SYMBOL = {"USD": "$", "INR": "₹"}

console = Console()
err_console = Console(stderr=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m scanner.run", description=__doc__.split("\n\n")[0])
    p.add_argument("--market", default="us", choices=sorted(MARKETS))
    p.add_argument("--window", help="show only names passing this window (e.g. 1m, 3m, 6m)")
    p.add_argument("--mode", choices=["fixed", "rank"], help="override leaders.mode from the config")
    p.add_argument("--no-update", action="store_true", help="use cached data only (no downloads)")
    p.add_argument("--refresh", action="store_true", help="re-run the incremental update even if already done today")
    p.add_argument("--as-of", type=date.fromisoformat, help="evaluate the scan on a past session (YYYY-MM-DD)")
    p.add_argument("--symbols", help="comma-separated symbols instead of the full universe (dev)")
    p.add_argument("--limit", type=int, help="only the first N universe symbols, alphabetically (dev)")
    p.add_argument("--csv", help="also write the result table to this CSV file")
    p.add_argument("-v", "--verbose", action="store_true", help="show info-level logs")
    return p.parse_args(argv)


def short_name(name: str) -> str:
    """'NVIDIA Corporation - Common Stock' -> 'NVIDIA Corporation'."""
    cut = re.split(
        r"\s+-\s+|\s+(?:Class [A-Z] )?(?:New )?(?:Common|Ordinary|American Depositary|Depositary|Units?)\b",
        str(name),
        maxsplit=1,
    )[0]
    return cut.strip(" ,") or str(name)


def fmt_money(value: float, cur: str) -> str:
    sym = CURRENCY_SYMBOL.get(cur, "")
    if pd.isna(value):
        return "-"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= div:
            return f"{sym}{value / div:.1f}{suffix}"
    return f"{sym}{value:.0f}"


def fmt_pct(value: float) -> str:
    return "-" if pd.isna(value) else f"{value * 100:+.0f}%"


class ProgressReporter:
    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.tasks: dict[str, int] = {}

    def __call__(self, phase: str, done: int, total: int) -> None:
        if phase not in self.tasks:
            self.tasks[phase] = self.progress.add_task(phase, total=total)
        self.progress.update(self.tasks[phase], completed=done, total=total)


def select_symbols(universe: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    syms = universe["symbol"].tolist()
    return syms[: args.limit] if args.limit else syms


def load_data(cfg: MarketConfig, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, UpdateReport | None]:
    market = MARKETS[args.market]
    cache = PriceCache(cfg.data.cache_path())

    if args.no_update:
        if not cache.prices_path.exists():
            raise SystemExit("No cached prices yet. Run once without --no-update.")
        universe = (
            pd.read_csv(cache.universe_path, dtype=str, keep_default_na=False)
            if cache.universe_path.exists()
            else pd.DataFrame(columns=["symbol", "name"])
        )
        symbols = select_symbols(universe, args) if (args.symbols or len(universe)) else []
        prices = cache.load()
        if symbols:
            prices = prices[prices["symbol"].isin(symbols)]
        return universe, prices, None

    universe = market.load_universe(cfg)
    symbols = select_symbols(universe, args)
    now = datetime.now(timezone.utc)
    cutoff = last_complete_session_date(now, cfg.data)
    history_start = (pd.Timestamp(cutoff) - pd.DateOffset(years=cfg.data.history_years)).date()
    with Progress(
        TextColumn("[bold]{task.description:<12}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err_console,
        transient=True,
    ) as progress:
        prices, report = update_prices(
            cache,
            market.make_provider(cfg),
            symbols,
            history_start=history_start,
            cutoff=cutoff,
            overlap_bars=cfg.data.incremental_overlap_bars,
            tolerance_pct=cfg.data.revision_tolerance_pct,
            refresh=args.refresh,
            progress=ProgressReporter(progress),
            now=now,
        )
    prices = prices[prices["symbol"].isin(symbols)]
    return universe, prices, report


def render(res: LeadersResult, cfg: MarketConfig, universe: pd.DataFrame, args: argparse.Namespace,
           report: UpdateReport | None) -> pd.DataFrame:
    names = dict(zip(universe.get("symbol", []), universe.get("name", [])))
    table = res.table
    if args.window:
        table = table[table[f"pass_{args.window}"]]
    cur = cfg.currency
    lo, hi = cfg.leaders.target_count

    t = Table(
        title=f"{cfg.market.upper()} momentum leaders · {res.as_of:%Y-%m-%d} · {res.mode} mode",
        header_style="bold",
        title_justify="left",
        pad_edge=False,
    )
    t.add_column("#", justify="right", style="dim")
    t.add_column("Symbol", style="bold")
    t.add_column("Name", max_width=28, no_wrap=True, overflow="ellipsis")
    t.add_column("Close", justify="right")
    for w in res.windows:
        t.add_column(w.upper(), justify="right")
        if res.mode == "rank":
            t.add_column(f"RS {w.upper()}", justify="right", style="dim")
    t.add_column("Scans", justify="center")
    t.add_column("ADR%", justify="right")
    t.add_column(f"Avg {CURRENCY_SYMBOL.get(cur, cur)}Vol", justify="right")

    for i, (sym, r) in enumerate(table.iterrows(), start=1):
        shown = f"{sym} [yellow]⚠[/]" if r["flag_day_move"] else sym
        cells = [str(i), shown, short_name(names.get(sym, "")), f"{r['close']:.2f}"]
        for w in res.windows:
            txt = fmt_pct(r[f"ret_{w}"])
            cells.append(f"[green]{txt}[/]" if r[f"pass_{w}"] else f"[dim]{txt}[/]")
            if res.mode == "rank":
                cells.append("-" if pd.isna(r[f"rs_{w}"]) else f"{r[f'rs_{w}']:.0f}")
        cells.append(" ".join("[green]✓[/]" if r[f"pass_{w}"] else "[dim]·[/]" for w in res.windows))
        cells += [f"{r['adr_pct']:.1f}", fmt_money(r["avg_traded_value"], cur)]
        t.add_row(*cells)

    if len(table):
        console.print(t)
    else:
        console.print(f"[yellow]No leaders on {res.as_of:%Y-%m-%d}.[/]")

    liq = cfg.liquidity
    sym = CURRENCY_SYMBOL.get(cur, cur + " ")
    console.print(
        f"[dim]As of {res.as_of:%Y-%m-%d} (last complete session) · {res.active_count:,} symbols with data"
        + (f" ({res.stale_count:,} without a bar that day)" if res.stale_count else "")
        + f" · {res.liquid_count:,} pass liquidity (close ≥ {sym}{liq.min_price:g}, "
        f"{liq.window}d avg {sym}vol ≥ {fmt_money(liq.min_avg_traded_value, cur)}, ADR ≥ {liq.min_adr_pct:g}%)[/]"
    )
    parts = []
    for w in res.windows:
        n = res.counts[w]
        flag = "" if lo <= n <= hi else (" [yellow](below target)[/]" if n < lo else " [yellow](above target)[/]")
        parts.append(f"{w.upper()} {n}{flag}")
    rule = (
        "close ≥ " + ", ".join(f"{cfg.leaders.windows[w].min_multiple:g}× {cfg.leaders.windows[w].lookback} bars ago"
                               for w in res.windows)
        if res.mode == "fixed"
        else f"top {cfg.leaders.rank_top_pct:g}% by return"
    )
    console.print(f"Per scan: {' · '.join(parts)}  [dim](target {lo}–{hi}; {rule})[/]")
    n_flag = int(table["flag_day_move"].sum()) if len(table) else 0
    if n_flag:
        m = cfg.leaders.flag_day_move_multiple
        console.print(
            f"[yellow]⚠[/] {n_flag} names had a one-day move ≥ {m:g}× (or ≤ 1/{m:g}×) inside the window: "
            "a real squeeze or a reverse split the data source has not adjusted yet. Check the chart."
        )
    if report is not None:
        msg = (
            f"[dim]Data: {report.new_symbols:,} new, {report.incremental:,} updated, "
            f"{len(report.full_refetch)} re-downloaded (split/revision)"
            + (f", {report.already_current:,} already current" if report.already_current else "")
            + "[/]"
        )
        console.print(msg)
        if report.full_refetch:
            console.print(f"[dim]  re-downloaded: {', '.join(report.full_refetch[:15])}"
                          + (" …" if len(report.full_refetch) > 15 else "") + "[/]")
        if report.failed:
            console.print(f"[yellow]  {len(report.failed)} symbols returned no data: "
                          f"{', '.join(report.failed[:15])}{' …' if len(report.failed) > 15 else ''}[/]")
    console.print("[dim]Information only, not investment advice.[/]")
    return table


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=err_console, show_path=False)],
    )
    try:
        cfg = load_config(args.market)
    except ConfigError as e:
        err_console.print(f"[red]{e}[/]")
        return 2
    if args.window and args.window not in cfg.leaders.windows:
        err_console.print(f"[red]--window must be one of {list(cfg.leaders.windows)}[/]")
        return 2

    try:
        universe, prices, report = load_data(cfg, args)
    except KeyboardInterrupt:
        err_console.print("[yellow]Interrupted. Downloaded batches were checkpointed; rerun to resume.[/]")
        return 130
    if prices.empty:
        err_console.print("[red]No price data for the selected symbols.[/]")
        return 1

    panels = Panels.from_long(prices)
    if args.as_of:
        panels = panels.as_of(args.as_of)
        if panels.close.empty:
            err_console.print(f"[red]No sessions on or before {args.as_of}.[/]")
            return 1
    res = momentum_leaders(panels, cfg.liquidity, cfg.leaders, mode=args.mode)
    table = render(res, cfg, universe, args, report)

    if args.csv:
        out = table.copy()
        names = dict(zip(universe.get("symbol", []), universe.get("name", [])))
        out.insert(0, "name", [names.get(s, "") for s in out.index])
        out.insert(0, "as_of", res.as_of.date())
        out.to_csv(args.csv)
        console.print(f"[dim]Wrote {len(out)} rows to {args.csv}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
