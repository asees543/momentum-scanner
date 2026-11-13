"""CLI: update market data and print breakouts, the watchlist and momentum leaders.

    uv run python -m scanner.run --market us                       # breakouts + watchlist + leader counts
    uv run python -m scanner.run --market us --scan leaders        # full leaders table
    uv run python -m scanner.run --market us --scan watchlist --window 1m
    uv run python -m scanner.run --market us --no-update --as-of 2026-10-30
    uv run python -m scanner.run --market us --symbols NVDA,PLTR,HOOD     # quick dev run
    uv run python -m scanner.run --market us --alert --dry-run    # print the Telegram messages
    uv run --env-file .env python -m scanner.run --market us --alert   # send + save state/us.json
    uv run --env-file .env python -m scanner.run --market us --no-update --send-watchlist   # full list now

Output is information, not investment advice.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from scanner.config import ConfigError, MarketConfig, load_config
from scanner.data import india_loader, us_loader
from scanner.data.base import Panels, last_complete_session_date
from scanner.data.cache import PriceCache, UpdateReport, update_prices
from scanner.notify import (
    NotifyError,
    PrintNotifier,
    TelegramNotifier,
    format_alerts,
    format_failure,
    format_warning,
    CURRENCY_SYMBOL,
    format_watchlist,
    money,
    move_label,
    short_name,
)
from scanner.scans import BreakoutResult, LeadersResult, ScanResults, WatchlistResult, run_scans
from scanner.state import STATE_DIR, MarketState, diff_results, state_path

MARKETS: dict[str, ModuleType] = {"us": us_loader, "india": india_loader}

console = Console()
err_console = Console(stderr=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m scanner.run", description=__doc__.split("\n\n")[0])
    p.add_argument("--market", default="us", choices=sorted(MARKETS))
    p.add_argument("--scan", default="all", choices=["all", "leaders", "watchlist", "breakouts"],
                   help="what to print (default: breakouts, watchlist and leader counts)")
    p.add_argument("--window", help="only names that are leaders in this window (e.g. 1m, 3m, 6m); for the "
                   "watchlist and breakouts: a leader in that window during the setup lookback")
    p.add_argument("--mode", choices=["fixed", "rank"], help="override leaders.mode from the config")
    p.add_argument("--no-update", action="store_true", help="use cached data only (no downloads)")
    p.add_argument("--refresh", action="store_true", help="re-run the incremental update even if already done today")
    p.add_argument("--as-of", type=date.fromisoformat, help="evaluate the scan on a past session (YYYY-MM-DD)")
    p.add_argument("--symbols", help="comma-separated symbols instead of the full universe (dev)")
    p.add_argument("--limit", type=int, help="only the first N universe symbols, alphabetically (dev)")
    p.add_argument("--csv", help="also write the table(s) to CSV; with --scan all, out.csv becomes "
                   "out_breakouts.csv, out_watchlist.csv and out_leaders.csv")
    p.add_argument("--alert", action="store_true",
                   help="compare with the last sent state (state/<market>.json) and send what is new via Telegram")
    p.add_argument("--send-watchlist", action="store_true",
                   help="send the whole current watchlist to Telegram now (state is not touched)")
    p.add_argument("--dry-run", action="store_true",
                   help="with --alert or --send-watchlist: print the messages instead of sending; state is not saved")
    p.add_argument("--state-dir", default=str(STATE_DIR), help=argparse.SUPPRESS)  # tests / experiments
    p.add_argument("-v", "--verbose", action="store_true", help="show info-level logs")
    return p.parse_args(argv)


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


def _names(universe: pd.DataFrame) -> dict[str, str]:
    return dict(zip(universe.get("symbol", []), universe.get("name", [])))


def _displays(universe: pd.DataFrame) -> dict[str, str]:
    """Yahoo symbol -> the exchange's own symbol (RELIANCE.NS -> RELIANCE, BRK-B -> BRK.B)."""
    if "source_symbol" not in universe.columns:
        return {}
    return {s: d for s, d in zip(universe["symbol"], universe["source_symbol"]) if d}


def _symbol_cell(sym: str, r: pd.Series, displays: dict[str, str]) -> str:
    shown = displays.get(sym, sym)
    return f"{shown} [yellow]⚠[/]" if r.get("flag_day_move", False) else shown


def _leader_marks(r: pd.Series, windows: list[str], prefix: str) -> str:
    return " ".join("[green]✓[/]" if r[f"{prefix}{w}"] else "[dim]·[/]" for w in windows)


def _new_table(title: str) -> Table:
    return Table(title=title, header_style="bold", title_justify="left", pad_edge=False)


def render_leaders_table(res: LeadersResult, cfg: MarketConfig, names: dict, window: str | None,
        displays: dict[str, str] | None = None) -> pd.DataFrame:
    table = res.table
    if window:
        table = table[table[f"pass_{window}"]]
    cur = cfg.currency
    t = _new_table(f"{cfg.market.upper()} momentum leaders · {res.as_of:%Y-%m-%d} · {res.mode} mode")
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
        cells = [str(i), _symbol_cell(sym, r, displays or {}), short_name(names.get(sym, "")), f"{r['close']:.2f}"]
        for w in res.windows:
            txt = fmt_pct(r[f"ret_{w}"])
            cells.append(f"[green]{txt}[/]" if r[f"pass_{w}"] else f"[dim]{txt}[/]")
            if res.mode == "rank":
                cells.append("-" if pd.isna(r[f"rs_{w}"]) else f"{r[f'rs_{w}']:.0f}")
        cells.append(_leader_marks(r, res.windows, "pass_"))
        cells += [f"{r['adr_pct']:.1f}", money(r["avg_traded_value"], cur)]
        t.add_row(*cells)
    if len(table):
        console.print(t)
    else:
        console.print(f"[yellow]No leaders on {res.as_of:%Y-%m-%d}.[/]")
    return table


def render_leaders_summary(res: LeadersResult, cfg: MarketConfig, flagged: int | None = None) -> None:
    cur = cfg.currency
    lo, hi = cfg.leaders.target_count
    liq = cfg.liquidity
    sym = CURRENCY_SYMBOL.get(cur, cur + " ")
    console.print(
        f"[dim]As of {res.as_of:%Y-%m-%d} (last complete session) · {res.active_count:,} symbols with data"
        + (f" ({res.stale_count:,} without a bar that day)" if res.stale_count else "")
        + f" · {res.liquid_count:,} pass liquidity (close ≥ {sym}{liq.min_price:g}, "
        f"{liq.window}d avg {sym}vol ≥ {money(liq.min_avg_traded_value, cur)}, ADR ≥ {liq.min_adr_pct:g}%)[/]"
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
    console.print(f"Leaders per scan: {' · '.join(parts)}  [dim](target {lo}–{hi}; {rule})[/]")
    if flagged:
        m = cfg.leaders.flag_day_move_multiple
        console.print(
            f"[yellow]⚠[/] {flagged} names had a one-day move of {move_label(m)} or more inside the 6M window: "
            "a real squeeze, or a split/bonus the data source has not adjusted yet. Check the chart."
        )


def render_watchlist(wl: WatchlistResult, cfg: MarketConfig, names: dict, window: str | None,
        displays: dict[str, str] | None = None) -> pd.DataFrame:
    table = wl.table
    if window:
        table = table[table[f"led_{window}"]]
    cur = cfg.currency
    s, b = cfg.setup, cfg.breakout
    t = _new_table(f"{cfg.market.upper()} watchlist · {wl.as_of:%Y-%m-%d} · can break out next session")
    t.add_column("#", justify="right", style="dim")
    t.add_column("Symbol", style="bold")
    t.add_column("Name", max_width=24, no_wrap=True, overflow="ellipsis")
    t.add_column("Close", justify="right")
    t.add_column("Trigger", justify="right", style="bold")
    t.add_column("To trig", justify="right")
    t.add_column("Stop", justify="right")
    t.add_column("Risk%", justify="right")
    t.add_column("ADR%", justify="right")
    t.add_column("Vol/avg", justify="right")
    t.add_column("Leader " + " ".join(w.upper() for w in wl.windows), justify="center")
    t.add_column("Age", justify="right")
    t.add_column(f"Avg {CURRENCY_SYMBOL.get(cur, cur)}Vol", justify="right")
    for i, (sym, r) in enumerate(table.iterrows(), start=1):
        risk = f"{r['risk_pct']:.1f}"
        t.add_row(
            str(i),
            _symbol_cell(sym, r, displays or {}),
            short_name(names.get(sym, "")),
            f"{r['close']:.2f}",
            f"{r['trigger']:.2f}",
            f"{r['to_trigger_pct']:+.1f}%",
            f"{r['stop']:.2f}",
            f"[yellow]{risk}[/]" if r["risk_too_wide"] else risk,
            f"{r['adr_pct']:.1f}",
            f"{r['vol_ratio']:.2f}",
            _leader_marks(r, wl.windows, "led_"),
            "today" if r["setup_age"] == 0 else f"{int(r['setup_age'])}d",
            money(r["avg_traded_value"], cur),
        )
    if len(table):
        console.print(t)
    else:
        console.print(f"[yellow]Watchlist is empty on {wl.as_of:%Y-%m-%d}.[/]")
    wide = int(table["risk_too_wide"].sum()) if len(table) else 0
    console.print(
        f"Watchlist: {len(table)} names ({wl.qualified_today} qualified today, the rest within the last "
        f"{b.watchlist_memory_sessions} sessions and still below their trigger)"
        + (f" · [yellow]{wide} with risk wider than ADR[/]" if wide else "")
    )
    console.print(
        f"[dim]Trigger = highest high of the last {s.trigger_bars} bars on the setup day · stop = latest bar's low "
        f"(the real stop is the breakout day's low) · risk% = (trigger − stop) / trigger · Age = sessions since "
        f"the setup last qualified · Vol/avg = today's volume ÷ previous {s.volume_avg_window}-session mean[/]"
    )
    return table


def render_breakouts(bo: BreakoutResult, cfg: MarketConfig, names: dict, window: str | None,
        displays: dict[str, str] | None = None) -> pd.DataFrame:
    table = bo.table
    if window and len(table):
        table = table[table[f"led_{window}"]]
    cur = cfg.currency
    b = cfg.breakout
    t = _new_table(f"{cfg.market.upper()} breakouts · {bo.as_of:%Y-%m-%d}")
    t.add_column("#", justify="right", style="dim")
    t.add_column("Symbol", style="bold")
    t.add_column("Name", max_width=24, no_wrap=True, overflow="ellipsis")
    t.add_column("Close", justify="right")
    t.add_column("Trigger", justify="right")
    t.add_column("Stop", justify="right", style="bold")
    t.add_column("Risk%", justify="right")
    t.add_column("ADR%", justify="right")
    t.add_column("Vol×", justify="right")
    t.add_column("Status")
    t.add_column("Leader " + " ".join(w.upper() for w in bo.windows), justify="center")
    t.add_column("Setup", justify="right")
    t.add_column(f"Avg {CURRENCY_SYMBOL.get(cur, cur)}Vol", justify="right")
    for i, (sym, r) in enumerate(table.iterrows(), start=1):
        ok = r["status"] == "ok"
        t.add_row(
            str(i),
            _symbol_cell(sym, r, displays or {}),
            short_name(names.get(sym, "")),
            f"{r['close']:.2f}",
            f"{r['trigger']:.2f}",
            f"{r['stop']:.2f}",
            f"{r['risk_pct']:.1f}",
            f"{r['adr_pct']:.1f}",
            f"{r['vol_ratio']:.1f}",
            "[green]ok[/]" if ok else "[yellow]too wide[/]",
            _leader_marks(r, bo.windows, "led_"),
            f"{int(r['setup_age'])}d ago",
            money(r["avg_traded_value"], cur),
        )
    if len(table):
        console.print(t)
    else:
        console.print(f"No breakouts on {bo.as_of:%Y-%m-%d}.")
    if len(table):
        n_ok = int((table["status"] == "ok").sum())
        console.print(
            f"Breakouts: {n_ok} ok, {len(table) - n_ok} too wide  [dim](close > trigger, volume ≥ "
            f"{b.min_volume_ratio:g}× average; stop = day low; too wide = risk% > {b.max_risk_vs_adr:g}× ADR%)[/]"
        )
    return table


def render_data_report(report: UpdateReport | None) -> None:
    if report is None:
        return
    console.print(
        f"[dim]Data: {report.new_symbols:,} new, {report.incremental:,} updated, "
        f"{len(report.full_refetch)} re-downloaded (split/revision)"
        + (f", {report.already_current:,} already current" if report.already_current else "")
        + "[/]"
    )
    if report.full_refetch:
        console.print(f"[dim]  re-downloaded: {', '.join(report.full_refetch[:15])}"
                      + (" …" if len(report.full_refetch) > 15 else "") + "[/]")
    if report.failed:
        console.print(f"[yellow]  {len(report.failed)} symbols returned no data: "
                      f"{', '.join(report.failed[:15])}{' …' if len(report.failed) > 15 else ''}[/]")


def render(res: ScanResults, cfg: MarketConfig, universe: pd.DataFrame, args: argparse.Namespace,
           report: UpdateReport | None) -> dict[str, pd.DataFrame]:
    """Print the requested scan(s); return the tables shown, keyed by scan name."""
    names = _names(universe)
    displays = _displays(universe)
    shown: dict[str, pd.DataFrame] = {}
    if args.scan in ("all", "breakouts"):
        shown["breakouts"] = render_breakouts(res.breakouts, cfg, names, args.window, displays)
        console.print()
    if args.scan in ("all", "watchlist"):
        shown["watchlist"] = render_watchlist(res.watchlist, cfg, names, args.window, displays)
        console.print()
    if args.scan == "leaders":
        shown["leaders"] = render_leaders_table(res.leaders, cfg, names, args.window, displays)
        render_leaders_summary(res.leaders, cfg, int(shown["leaders"]["flag_day_move"].sum()))
    elif args.scan == "all":
        shown["leaders"] = res.leaders.table
        render_leaders_summary(res.leaders, cfg)
    render_data_report(report)
    console.print("[dim]Information only, not investment advice.[/]")
    return shown


def write_csv(path: str, shown: dict[str, pd.DataFrame], as_of: pd.Timestamp, names: dict) -> None:
    target = Path(path)
    for scan, table in shown.items():
        out_path = target if len(shown) == 1 else target.with_name(f"{target.stem}_{scan}{target.suffix or '.csv'}")
        out = table.copy()
        out.insert(0, "name", [names.get(s, "") for s in out.index])
        out.insert(0, "as_of", as_of.date())
        out.to_csv(out_path)
        console.print(f"[dim]Wrote {len(out)} rows to {out_path}[/]")


def alert_flow(res: ScanResults, cfg: MarketConfig, universe: pd.DataFrame, chart_url, dry_run: bool,
               state_dir: Path = STATE_DIR, notifier=None) -> int:
    """Diff today's results against the last SENT state, deliver, then save the new state.
    State is saved only after delivery succeeded, and never on a dry run."""
    path = state_path(cfg.market, state_dir)
    prev = MarketState.load(path)
    diff = diff_results(prev, res, cfg.market, cfg.alerts)
    if diff.status == "skip_same_day":
        console.print(f"[dim]Alerts: {diff.reason}; nothing sent.[/]")
        return 0
    if notifier is None:
        notifier = PrintNotifier() if dry_run else TelegramNotifier.from_env()
    if diff.status in ("refuse_older", "unhealthy"):
        notifier.send([format_warning(cfg.market, diff)])
        err_console.print(f"[red]Alerts not sent: {diff.reason}[/]")
        return 1
    messages = format_alerts(diff, res, cfg, universe, chart_url)
    notifier.send(messages)
    if not dry_run:
        diff.next_state.save(path)
    console.print(
        f"Alerts: {len(messages)} message(s) {'printed' if dry_run else 'sent'} · "
        f"{len(diff.breakouts)} breakouts ok, {len(diff.wide_breakouts)} too wide, "
        f"{len(diff.new_watchlist)} new watchlist names"
        + (" · first run: watchlist recorded as baseline" if diff.first_run else "")
        + (" · dry run, state not saved" if dry_run else f" · state saved to {path}")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _main(args)
    except Exception as e:
        if args.alert and not args.dry_run:
            try:  # best effort: tell the chat the scheduled run broke
                TelegramNotifier.from_env().send([format_failure(args.market, f"{type(e).__name__}: {e}")])
            except NotifyError:
                pass
        raise


def _main(args: argparse.Namespace) -> int:
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
    if args.dry_run and not (args.alert or args.send_watchlist):
        err_console.print("[red]--dry-run only applies to --alert or --send-watchlist[/]")
        return 2
    if args.alert and not args.dry_run and (args.symbols or args.limit or args.as_of):
        # A partial universe or a past date would overwrite state/<market>.json with the wrong watchlist.
        err_console.print("[red]--alert with --symbols/--limit/--as-of is only allowed with --dry-run[/]")
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
    res = run_scans(panels, cfg, mode=args.mode)
    shown = render(res, cfg, universe, args, report)
    if args.csv:
        write_csv(args.csv, shown, res.leaders.as_of, _names(universe))
    if args.send_watchlist:
        messages = format_watchlist(res, cfg, universe, MARKETS[args.market].chart_url)
        (PrintNotifier() if args.dry_run else TelegramNotifier.from_env()).send(messages)
        console.print(
            f"Full watchlist: {len(res.watchlist.table)} names in {len(messages)} message(s) "
            + ("printed (dry run)" if args.dry_run else "sent")
        )
    if args.alert:
        console.print()
        return alert_flow(res, cfg, universe, MARKETS[args.market].chart_url, args.dry_run, Path(args.state_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
