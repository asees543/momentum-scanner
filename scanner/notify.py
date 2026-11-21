"""Alert messages and delivery (Telegram primary; email is optional in SPEC and not built yet).

Message layout (Telegram HTML); each entry is three lines so it reads well on a phone:

    US scan · Thu 12 Nov 2026
    Watchlist 64 (15 new) · Breakouts 1 ok, 0 too wide · Leaders 1M 141 / 3M 46 / 6M 22

    🚀 Breakouts
    SGMT · Sagimet Biosciences
    11.41 › trig 10.92 · stop 10.80 · risk 5.3% · ADR 7.3 · vol 1.5×
    1M … · 3M … · 6M … · $11.4M · chart

    ⚠️ Breakouts, stop wider than ADR      (listed when alerts.list_wide_breakouts is on)
    … same three-line entries …

    👀 New on watchlist (closest to trigger first)
    XERS · Xeris Biopharma Holdings
    16.65 · trig 16.88 (+1.4%) · stop 16.35 · risk 3.1% · ADR 5.4
    1M +63% · 3M +108% · 6M +161% · $43.8M · chart

    Information only, not investment advice.

`format_watchlist` sends the whole current watchlist in the same entry format (on demand;
it never touches state).

Secrets come from the environment (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID): `.env` locally via
`uv run --env-file .env ...`, repository secrets in GitHub Actions. The token is part of every Bot
API URL, so all error text is scrubbed before it can reach a log or a message.

Helpers:
    uv run --env-file .env python -m scanner.notify --chat-id   # list chats that messaged the bot
    uv run --env-file .env python -m scanner.notify --test      # send a test message
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time
from typing import Callable, Protocol

import pandas as pd

from scanner.config import MarketConfig
from scanner.scans import ScanResults
from scanner.state import Diff

TELEGRAM_LIMIT = 4096
MESSAGE_BUDGET = 4000  # stay under the limit with a margin
CURRENCY_SYMBOL = {"USD": "$", "INR": "₹"}
DISCLAIMER = "<i>Information only, not investment advice.</i>"

ChartUrlFn = Callable[[str, "str | None", "str | None"], str]


class NotifyError(Exception):
    """Delivery failed. The message never contains the bot token."""


# --- formatting -------------------------------------------------------------------------------


def _e(text) -> str:
    return html.escape(str(text), quote=False)


# Units per currency, largest first: US dollars in B/M/K; rupees in crore (1 Cr = 10,000,000) and
# lakh (1 L = 100,000), the units Indian market data is quoted in.
MONEY_UNITS = {
    "USD": ((1e9, "B"), (1e6, "M"), (1e3, "K")),
    "INR": ((1e7, " Cr"), (1e5, " L")),
}


def money(value: float, cur: str) -> str:
    """$43.8M, ₹12.3 Cr, ₹45.0 L."""
    sym = CURRENCY_SYMBOL.get(cur, "")
    if pd.isna(value):
        return "-"
    for div, suffix in MONEY_UNITS.get(cur, MONEY_UNITS["USD"]):
        if abs(value) >= div:
            return f"{sym}{value / div:.1f}{suffix}"
    return f"{sym}{value:.0f}"


def move_label(multiple: float) -> str:
    """The one-day-move flag threshold in percent: 3.0 -> '+200% or -67%', 1.3 -> '+30% or -23%'.
    A move of `multiple` up is +(m-1); the same multiple down is a fall of 1 - 1/m."""
    return f"+{(multiple - 1) * 100:.0f}% or −{(1 - 1 / multiple) * 100:.0f}%"


def flag_note(cfg: MarketConfig) -> str:
    return (
        f"⚠ = a one-day move of {move_label(cfg.leaders.flag_day_move_multiple)} or more in the last 6 months: "
        "a real squeeze, or a split/bonus the data source has not adjusted yet. Check the chart."
    )


def _pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x * 100:+.0f}%"


def short_name(name: str, max_len: int = 32) -> str:
    """'NVIDIA Corporation - Common Stock' -> 'NVIDIA Corporation' (cut to max_len characters)."""
    cut = re.split(
        r"\s+-\s+|\s+(?:Class [A-Z] )?(?:New )?(?:Common|Ordinary|American Depositary|Depositary|Units?)\b",
        str(name),
        maxsplit=1,
    )[0]
    cut = cut.strip(" ,") or str(name)
    return cut if len(cut) <= max_len else cut[: max_len - 1].rstrip() + "…"


def _title_line(sym: str, r: pd.Series, name: str) -> str:
    flag = " ⚠" if r.get("flag_day_move", False) else ""
    return f"<b>{_e(sym)}</b>{flag}" + (f" · {_e(name)}" if name else "")


def _returns_line(r: pd.Series, windows: list[str], cur: str, url: str) -> str:
    rets = " · ".join(f"{w.upper()} {_pct(r[f'ret_{w}'])}" for w in windows)
    return f"{rets} · {money(r['avg_traded_value'], cur)} · <a href=\"{_e(url)}\">chart</a>"


def breakout_entry(sym: str, r: pd.Series, windows: list[str], cur: str, url: str, name: str = "") -> str:
    numbers = (
        f"{r['close']:.2f} › trig {r['trigger']:.2f} · stop {r['stop']:.2f} · "
        f"risk {r['risk_pct']:.1f}% · ADR {r['adr_pct']:.1f} · vol {r['vol_ratio']:.1f}×"
    )
    return "\n".join([_title_line(sym, r, name), numbers, _returns_line(r, windows, cur, url)])


def watchlist_entry(sym: str, r: pd.Series, windows: list[str], cur: str, url: str, name: str = "") -> str:
    risk = f"risk {r['risk_pct']:.1f}%" + (" (wide)" if r["risk_too_wide"] else "")
    numbers = (
        f"{r['close']:.2f} · trig {r['trigger']:.2f} ({r['to_trigger_pct']:+.1f}%) · "
        f"stop {r['stop']:.2f} · {risk} · ADR {r['adr_pct']:.1f}"
    )
    return "\n".join([_title_line(sym, r, name), numbers, _returns_line(r, windows, cur, url)])


def jack_rule_text(cfg: MarketConfig) -> str:
    """The other screener's filters in words, e.g. 'above 21 EMA, 50 & 200 SMA, ADR above 4%, 50-day value above $250.0M'. Plain text: escape it for HTML."""
    j = cfg.jack
    return (
        f"above {j.ema_fast} EMA, {j.sma_mid} & {j.sma_long} SMA, ADR above {j.min_adr_pct:g}%, "
        f"{j.adv_window}-day value above {money(j.min_avg_traded_value, cfg.currency)}"
    )


def jack_entry(sym: str, r: pd.Series, windows: list[str], cur: str, url: str, name: str = "") -> str:
    """One name on both screens: where it is on our lists, then the usual numbers."""
    if r["ours"] == "leader":
        numbers = f"leader · {r['close']:.2f} · ADR {r['adr_pct']:.1f}"
    else:
        numbers = (
            f"{r['ours']} · {r['close']:.2f} · trig {r['trigger']:.2f} · stop {r['stop']:.2f} · ADR {r['adr_pct']:.1f}"
        )
    return "\n".join([_title_line(sym, r, name), numbers, _returns_line(r, windows, cur, url)])


BONUS_WORDS = [  # (column, label) in the order the course lists them
    ("inside", "inside bar"), ("tightest", "tightest of recent bars"), ("low_volume", "low volume"),
    ("strong_close", "strong close"), ("half_adr", "range under half an ADR"), ("confluence", "2+ key averages"),
]


TB_DIVIDER = "──────── not on our lists below ────────"


def trigger_bar_flags(r: pd.Series) -> list[str]:
    """Which of the six course bonuses this bar has, in words ('low volume' carries the ratio)."""
    out = []
    for col, label in BONUS_WORDS:
        if r[col]:
            out.append(f"low volume {r['vol_ratio']:.1f}×" if col == "low_volume" else label)
    return out


def trigger_bar_entry(sym: str, r: pd.Series, windows: list[str], cur: str, url: str, name: str = "") -> str:
    """One trigger bar: the plan (buy above the bar's high, stop under its low) and why it ranks."""
    wide = " · wide" if r["risk_pct"] > 5 else ""
    plan = (
        f"{int(r['score'])}/6 · on {_e(r['ma'])} · buy above {r['buy_above']:.2f} · stop {r['stop']:.2f} "
        f"({r['risk_pct']:.1f}%, {r['risk_vs_adr']:.1f}× ADR{wide})"
    )
    why = trigger_bar_flags(r) + ([f"{r['below_high_pct']:.0f}% below 52w high"] if pd.notna(r["below_high_pct"]) else [])
    if r["ours"]:
        why.append({"watchlist": "on our watchlist", "leader": "a leader today", "breakout": "a breakout today"}[r["ours"]])
    return "\n".join([_title_line(sym, r, name), plan, ", ".join(why), _returns_line(r, windows, cur, url)])


WINDOW_WORDS = {"1m": "1-month", "3m": "3-month", "6m": "6-month"}
LIST_CHUNK = 20  # names per block, so a long list splits between names, never inside one


def window_list_blocks(w: str, table: pd.DataFrame, cfg: MarketConfig, name, display) -> list[str]:
    """One window's leaders as numbered one-liners, biggest gain first. The header travels with the first
    names (a header alone at the end of a message would be an orphan)."""
    word = WINDOW_WORDS.get(w, w)
    head = (
        f"📈 <b>{_e(w.upper())} leaders</b> · {len(table)} names, biggest {word} gain first "
        f"(last-session value above {money(cfg.leaders.min_day_value, cfg.currency)})"
    )
    if table.empty:
        return [f"{head}\nNone today."]
    lines = []
    for i, (sym, r) in enumerate(table.iterrows(), start=1):
        flag = " ⚠" if r["flag_day_move"] else ""
        nm = name(sym)
        lines.append(
            f"{i}. <b>{_e(display(sym))}</b>{flag}" + (f" · {_e(nm)}" if nm else "")
            + f" · {_pct(r[f'ret_{w}'])} · {money(r['day_value'], cfg.currency)}"
        )
    chunks = ["\n".join(lines[i:i + LIST_CHUNK]) for i in range(0, len(lines), LIST_CHUNK)]
    chunks[0] = f"{head}\n{chunks[0]}"
    return chunks


def _lookups(universe: pd.DataFrame, chart_url: ChartUrlFn):
    """Per-symbol company name, chart link and display symbol from the universe table.
    The display symbol is the exchange's own (RELIANCE, BRK.B) rather than Yahoo's (RELIANCE.NS, BRK-B)."""
    info = universe.set_index("symbol") if "symbol" in universe.columns and len(universe) else pd.DataFrame()

    def display(sym: str) -> str:
        if sym in info.index and "source_symbol" in info.columns and info.loc[sym, "source_symbol"]:
            return str(info.loc[sym, "source_symbol"])
        return sym

    def name(sym: str) -> str:
        if sym in info.index and "name" in info.columns:
            return short_name(info.loc[sym, "name"])
        return ""

    def url(sym: str) -> str:
        if sym in info.index:
            row = info.loc[sym]
            return chart_url(sym, row.get("exchange"), row.get("source_symbol"))
        return chart_url(sym, None, None)

    return name, url, display


def pack(blocks: list[str], budget: int = MESSAGE_BUDGET) -> list[str]:
    """Join blocks with blank lines into as few messages as possible, each <= budget characters.
    A block is never split (one block = one alert entry), unless it alone exceeds the budget."""
    messages: list[str] = []
    current = ""
    for block in blocks:
        while len(block) > budget:  # pathological single block: hard split
            if current:
                messages.append(current)
                current = ""
            messages.append(block[:budget])
            block = block[budget:]
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= budget:
            current = candidate
        else:
            messages.append(current)
            current = block
    if current:
        messages.append(current)
    return messages


def format_alerts(
    diff: Diff,
    results: ScanResults,
    cfg: MarketConfig,
    universe: pd.DataFrame,
    chart_url: ChartUrlFn,
) -> list[str]:
    """Messages for a run with status "send"."""
    cur = cfg.currency
    windows = list(cfg.leaders.windows)
    name, url, display = _lookups(universe, chart_url)

    lead = results.leaders
    counts = " / ".join(f"{w.upper()} {lead.counts.get(w, 0)}" for w in windows)
    header = (
        f"<b>{_e(cfg.market.upper())} scan · {diff.as_of:%a %d %b %Y}</b>\n"
        f"Watchlist {diff.watchlist_size} ({len(diff.new_watchlist)} new) · "
        f"Breakouts {len(diff.breakouts)} ok, {diff.rejected_breakouts} too wide · Leaders {counts}"
    )
    blocks = [header]
    bo, wl = results.breakouts.table, results.watchlist.table
    shown: list[pd.Series] = []

    def bo_entries(symbols):
        shown.extend(bo.loc[s] for s in symbols)
        return [breakout_entry(display(s), bo.loc[s], windows, cur, url(s), name(s)) for s in symbols]

    def wl_entries(symbols):
        shown.extend(wl.loc[s] for s in symbols)
        return [watchlist_entry(display(s), wl.loc[s], windows, cur, url(s), name(s)) for s in symbols]

    if diff.breakouts:
        blocks.append("🚀 <b>Breakouts</b>")
        blocks += bo_entries(diff.breakouts)
    wide = diff.wide_breakouts if cfg.alerts.list_wide_breakouts else []
    if wide:
        blocks.append(
            "⚠️ <b>Breakouts, stop wider than ADR</b>\n"
            "SPEC rejects these: the distance from close to the day's low is more than one average daily range."
        )
        blocks += bo_entries(wide)
    for session, missed in diff.catch_up:
        t = missed.table
        ok = [s for s in t.index if t.loc[s, "status"] == "ok"]
        wide_m = [s for s in t.index if t.loc[s, "status"] != "ok"] if cfg.alerts.list_wide_breakouts else []
        if not ok and not wide_m:
            continue
        blocks.append(
            f"⏪ <b>Breakouts on {session:%a %d %b}</b> (a session the earlier runs missed)"
            + (f"\n{len(ok)} ok, {len(wide_m)} with stop wider than ADR" if wide_m else "")
        )
        shown.extend(t.loc[s] for s in ok + wide_m)
        blocks += [breakout_entry(display(s), t.loc[s], windows, cur, url(s), name(s)) for s in ok]
        blocks += [
            breakout_entry(display(s), t.loc[s], windows, cur, url(s), name(s)).replace("</b>", "</b> (wide)", 1)
            for s in wide_m
        ]
    if diff.catch_up_skipped:
        blocks.append(f"⏪ {diff.catch_up_skipped} earlier missed session(s) were not checked.")
    tbr = results.trigger_bars
    if tbr is not None:
        tb = tbr.table
        shown_tb = tb.head(cfg.trigger_bar.message_max)
        head = (
            f"🔔 <b>Trigger bars today</b> · {len(tb)} names, {len(shown_tb)} shown (names on our lists first, then by "
            "how many of the course's 6 signs of a good one they have). A tight bar on a key moving average in an "
            "uptrend: be ready for a break above its high on the next session."
        )
        if tb.empty:
            blocks.append(f"{head}\nNone today.")
        else:
            blocks.append(head)
            shown.extend(shown_tb.loc[s] for s in shown_tb.index)
            on_ours = None  # the table is ordered names on our lists first: draw the line where they end
            for s in shown_tb.index:
                ours = shown_tb.loc[s, "ours"] != ""
                if on_ours and not ours:
                    blocks.append(TB_DIVIDER)
                on_ours = ours
                blocks.append(trigger_bar_entry(display(s), shown_tb.loc[s], windows, cur, url(s), name(s)))
            if len(tb) > len(shown_tb):
                blocks.append(f"…and {len(tb) - len(shown_tb)} more on the dashboard.")
    if diff.new_watchlist:
        blocks.append("👀 <b>New on watchlist</b> (closest to trigger first)")
        blocks += wl_entries(diff.new_watchlist)
    if diff.first_run:
        blocks.append(
            f"📋 <b>First run: full watchlist</b> ({diff.watchlist_size} names, closest to trigger first)\n"
            "These are recorded as the baseline. From the next session, only new names are listed."
        )
        blocks += wl_entries(list(wl.index))
    if not diff.breakouts and not wide and not diff.new_watchlist and not diff.first_run and not diff.catch_up:
        blocks.append("Nothing new today.")
    jk = results.jack
    if jk is not None:
        # All names on both screens, every day (not only new ones): a short list, and the point is to see it.
        label = f"🎯 <b>On both screens</b> (ours and Corsellis's: {_e(jack_rule_text(cfg))})"
        if jk.table.empty:
            blocks.append(f"{label}\nNone today. {jk.passing_jack} names pass their screen alone.")
        else:
            blocks.append(f"{label}\n{len(jk.table)} of {jk.passing_jack} names that pass their screen are on our lists.")
            shown.extend(jk.table.loc[s] for s in jk.table.index)
            blocks += [jack_entry(display(s), jk.table.loc[s], windows, cur, url(s), name(s)) for s in jk.table.index]
    if results.window_lists is not None:
        # The three leader lists in full, every day: 1M, 3M and 6M each ranked by their own gain.
        for w in windows:
            lst = results.window_lists.get(w)
            if lst is None:
                continue
            shown.extend(lst.loc[s] for s in lst.index[lst["flag_day_move"]])
            blocks += window_list_blocks(w, lst, cfg, name, display)
    if any(bool(r.get("flag_day_move", False)) for r in shown):
        blocks.append(flag_note(cfg))
    blocks.append(DISCLAIMER)
    return pack(blocks)


def format_watchlist(results: ScanResults, cfg: MarketConfig, universe: pd.DataFrame, chart_url: ChartUrlFn) -> list[str]:
    """The whole current watchlist (closest to trigger first), for on-demand sending."""
    wl = results.watchlist
    windows = list(cfg.leaders.windows)
    name, url, display = _lookups(universe, chart_url)
    header = (
        f"<b>{_e(cfg.market.upper())} watchlist · {wl.as_of:%a %d %b %Y}</b>\n"
        f"{len(wl.table)} names, closest to trigger first ({wl.qualified_today} qualified that day, the rest "
        f"within the last {cfg.breakout.watchlist_memory_sessions} sessions and still below their trigger)"
    )
    blocks = [header]
    blocks += [
        watchlist_entry(display(s), wl.table.loc[s], windows, cfg.currency, url(s), name(s)) for s in wl.table.index
    ]
    if wl.table.empty:
        blocks.append("The watchlist is empty.")
    if len(wl.table) and wl.table["flag_day_move"].any():
        blocks.append(flag_note(cfg))
    blocks.append(DISCLAIMER)
    return pack(blocks)


def format_warning(market: str, diff: Diff) -> str:
    return (
        f"⚠️ <b>{_e(market.upper())} scan</b> · {diff.as_of:%a %d %b %Y}\n"
        f"{_e(diff.reason)}.\nNo alerts sent; state unchanged."
    )


def format_failure(market: str, error_text: str) -> str:
    return f"❌ <b>{_e(market.upper())} scan failed</b>\n<code>{_e(error_text[:1500])}</code>"


# --- delivery ---------------------------------------------------------------------------------


class Notifier(Protocol):
    def send(self, messages: list[str]) -> None: ...


def html_to_text(message: str) -> str:
    """Terminal rendition of a Telegram HTML message (for --dry-run)."""
    text = re.sub(r'<a href="([^"]*)">([^<]*)</a>', lambda m: f"{m.group(2)} {html.unescape(m.group(1))}", message)
    text = re.sub(r"</?(b|i|code)>", "", text)
    return html.unescape(text)


class PrintNotifier:
    """Dry run: print exactly what would be sent."""

    def __init__(self, out=None) -> None:
        self.out = out or sys.stdout

    def send(self, messages: list[str]) -> None:
        for i, m in enumerate(messages, start=1):
            print(f"──── message {i}/{len(messages)} ({len(m)} chars) ────", file=self.out)
            print(html_to_text(m), file=self.out)
        print("──── dry run: nothing sent ────", file=self.out)


class TelegramNotifier:
    API = "https://api.telegram.org"

    def __init__(self, token: str, chat_id: str, session=None, sleep: Callable[[float], None] = time.sleep,
                 max_retries: int = 3) -> None:
        if not token or not chat_id:
            raise NotifyError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set")
        self._token = token
        self.chat_id = chat_id
        self._sleep = sleep
        self.max_retries = max_retries
        if session is None:
            import requests

            session = requests.Session()
        self._session = session

    @classmethod
    def from_env(cls, **kw) -> TelegramNotifier:
        return cls(os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", ""), **kw)

    def scrub(self, text: str) -> str:
        return str(text).replace(self._token, "***") if self._token else str(text)

    def _call(self, method: str, http: str = "post", **kwargs) -> dict:
        url = f"{self.API}/bot{self._token}/{method}"
        for attempt in range(self.max_retries + 1):
            try:
                resp = getattr(self._session, http)(url, timeout=30, **kwargs)
            except Exception as e:  # network error: its text contains the URL, hence the token
                raise NotifyError(f"Telegram {method} failed: {self.scrub(e)}") from None
            if resp.status_code == 429 and attempt < self.max_retries:
                try:
                    wait = float(resp.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    wait = 5.0
                self._sleep(wait)
                continue
            try:
                body = resp.json()
            except Exception:
                body = {"ok": False, "description": resp.text[:300]}
            if resp.status_code != 200 or not body.get("ok"):
                raise NotifyError(
                    f"Telegram {method} failed ({resp.status_code}): {self.scrub(body.get('description', ''))}"
                )
            return body
        raise NotifyError(f"Telegram {method} still rate-limited after {self.max_retries} retries")

    def send(self, messages: list[str]) -> None:
        for m in messages:
            self._call(
                "sendMessage",
                json={"chat_id": self.chat_id, "text": m, "parse_mode": "HTML", "disable_web_page_preview": True},
            )

    def chat_ids(self) -> list[dict]:
        """Chats that recently messaged the bot (for finding TELEGRAM_CHAT_ID)."""
        body = self._call("getUpdates", http="get")
        seen: dict[int, dict] = {}
        for upd in body.get("result", []):
            msg = upd.get("message") or upd.get("channel_post") or upd.get("my_chat_member") or {}
            chat = msg.get("chat") or {}
            if "id" in chat:
                seen[chat["id"]] = {
                    "id": chat["id"],
                    "type": chat.get("type", ""),
                    "name": chat.get("title") or chat.get("username") or chat.get("first_name", ""),
                }
        return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scanner.notify")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--chat-id", action="store_true", help="list chats that messaged the bot")
    g.add_argument("--test", action="store_true", help="send a test message to TELEGRAM_CHAT_ID")
    args = p.parse_args(argv)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set (put it in .env and use: uv run --env-file .env ...)", file=sys.stderr)
        return 2
    try:
        if args.chat_id:
            chats = TelegramNotifier(token, "unused").chat_ids()
            if not chats:
                print("No chats found. Send any message to your bot in Telegram, then run this again.")
            for c in chats:
                print(f"TELEGRAM_CHAT_ID={c['id']}   ({c['type']}: {c['name']})")
        else:
            TelegramNotifier.from_env().send(["✅ momentum-scanner test message. Alerts will arrive in this chat."])
            print("Test message sent.")
    except NotifyError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
