import io
from datetime import date

import pandas as pd
import pytest

from scanner import scans
from scanner.data.us_loader import chart_url
from scanner.notify import (
    MESSAGE_BUDGET,
    NotifyError,
    PrintNotifier,
    TelegramNotifier,
    format_alerts,
    format_failure,
    format_warning,
    format_watchlist,
    html_to_text,
    pack,
    short_name,
)
from scanner.state import Diff, MarketState, diff_results
from tests.synth import panels_from, textbook, us_cfg

CFG = us_cfg()
UNIVERSE = pd.DataFrame(
    {"symbol": ["X", "BRK-B"], "source_symbol": ["X", "BRK.B"], "name": ["X Corp", "Berkshire"],
     "exchange": ["NASDAQ", "NYSE"]}
)
TOKEN = "123456:SECRET-token-value"


def breakout_results():
    b = textbook().append(33.0, 33.3, 31.9, 2_000_000)
    return scans.run_scans(panels_from({"X": b}), CFG)


def prev_state(as_of="2026-01-01", watch=None):
    return MarketState(market="us", as_of=date.fromisoformat(as_of), active_count=1, watchlist=watch or {})


# --- chart links ------------------------------------------------------------------------------


def test_chart_url():
    assert chart_url("XERS", "NASDAQ", "XERS") == "https://www.tradingview.com/chart/?symbol=NASDAQ:XERS"
    assert chart_url("BRK-B", "NYSE", "BRK.B").endswith("symbol=NYSE:BRK.B")
    assert chart_url("ACCS", "NYSE_AMERICAN", "ACCS").endswith("symbol=AMEX:ACCS")
    assert chart_url("ODD-A").endswith("symbol=ODD.A")  # unknown exchange: no prefix


# --- message content --------------------------------------------------------------------------


def test_breakout_alert_has_every_spec_field():
    res = breakout_results()
    d = diff_results(prev_state(), res, "us", CFG.alerts)
    msgs = format_alerts(d, res, CFG, UNIVERSE, chart_url)
    text = "\n".join(msgs)
    row = res.breakouts.table.loc["X"]
    assert "🚀 <b>Breakouts</b>" in text
    assert "<b>X</b> · X Corp\n33.00 › trig 31.79 · stop 31.90" in text
    assert f"risk {row['risk_pct']:.1f}%" in text and "ADR 5.9" in text and "vol 1.9×" in text
    for w in ("1M", "3M", "6M"):
        assert f"{w} " in text
    assert "$" in text  # average dollar volume
    assert '<a href="https://www.tradingview.com/chart/?symbol=NASDAQ:X">chart</a>' in text
    assert "Breakouts 1 ok, 0 too wide" in text
    assert "not investment advice" in text


def test_new_watchlist_alert_and_wide_marker():
    res = scans.run_scans(panels_from({"X": textbook()}), CFG)
    d = diff_results(prev_state(), res, "us", CFG.alerts)
    assert d.new_watchlist == ["X"]
    text = "\n".join(format_alerts(d, res, CFG, UNIVERSE, chart_url))
    assert "👀 <b>New on watchlist</b>" in text
    assert "<b>X</b> · X Corp\n31.40 · trig 31.79 (+1.2%) · stop 31.02" in text
    assert "Watchlist 1 (1 new)" in text
    assert "(wide)" not in text


def test_first_run_message_explains_baseline():
    res = scans.run_scans(panels_from({"X": textbook()}), CFG)
    d = diff_results(None, res, "us", CFG.alerts)
    text = "\n".join(format_alerts(d, res, CFG, UNIVERSE, chart_url))
    assert "recorded as the baseline" in text
    assert "New on watchlist" not in text


def test_quiet_day_still_sends_summary():
    res = scans.run_scans(panels_from({"X": textbook()}), CFG)
    d = diff_results(prev_state(watch={"X": {"trigger": 1, "first_seen": "2026-01-01"}}), res, "us", CFG.alerts)
    msgs = format_alerts(d, res, CFG, UNIVERSE, chart_url)
    assert len(msgs) == 1
    assert "Nothing new today." in msgs[0]
    assert "Watchlist 1 (0 new)" in msgs[0]


def test_warning_and_failure_are_escaped():
    d = Diff("unhealthy", date(2026, 11, 12), "only 10 < 5,000 symbols & falling")
    assert "&lt; 5,000 symbols &amp; falling" in format_warning("us", d)
    assert "<code>KeyError: &lt;bad&gt;</code>" in format_failure("us", "KeyError: <bad>")


# --- splitting --------------------------------------------------------------------------------


def test_pack_respects_budget_and_keeps_order():
    blocks = [f"entry {i} " + "x" * 300 for i in range(40)]  # ~12k characters
    msgs = pack(blocks)
    assert len(msgs) > 1
    assert all(len(m) <= MESSAGE_BUDGET for m in msgs)
    joined = "\n\n".join(msgs)
    assert joined == "\n\n".join(blocks)  # nothing lost, nothing reordered, no entry split


def test_pack_hard_splits_a_giant_block():
    msgs = pack(["y" * (MESSAGE_BUDGET * 2 + 10)])
    assert [len(m) for m in msgs] == [MESSAGE_BUDGET, MESSAGE_BUDGET, 10]


def test_dry_run_prints_readable_text():
    out = io.StringIO()
    PrintNotifier(out).send(['<b>US scan</b>\n<a href="https://t.example/x?a=1&amp;b=2">chart</a>'])
    printed = out.getvalue()
    assert "US scan" in printed and "<b>" not in printed
    assert "chart https://t.example/x?a=1&b=2" in printed
    assert "nothing sent" in printed
    assert html_to_text("a &amp; b") == "a & b"


# --- Telegram delivery (fake HTTP session) ----------------------------------------------------


class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    get = post


def test_send_posts_html_messages():
    s = FakeSession([FakeResp(200, {"ok": True}), FakeResp(200, {"ok": True})])
    TelegramNotifier(TOKEN, "42", session=s).send(["one", "two"])
    assert [c[1]["json"]["text"] for c in s.calls] == ["one", "two"]
    assert s.calls[0][1]["json"]["parse_mode"] == "HTML"
    assert s.calls[0][1]["json"]["chat_id"] == "42"


def test_rate_limit_waits_retry_after():
    waits = []
    s = FakeSession([FakeResp(429, {"ok": False, "parameters": {"retry_after": 7}}), FakeResp(200, {"ok": True})])
    TelegramNotifier(TOKEN, "42", session=s, sleep=waits.append).send(["hi"])
    assert waits == [7.0] and len(s.calls) == 2


def test_errors_never_contain_the_token():
    s = FakeSession([ConnectionError(f"HTTPSConnectionPool: /bot{TOKEN}/sendMessage refused")])
    with pytest.raises(NotifyError) as ei:
        TelegramNotifier(TOKEN, "42", session=s).send(["hi"])
    assert TOKEN not in str(ei.value) and "***" in str(ei.value)
    assert ei.value.__suppress_context__  # original exception (with the URL) is not printed

    s = FakeSession([FakeResp(401, {"ok": False, "description": f"Unauthorized {TOKEN}"})])
    with pytest.raises(NotifyError) as ei:
        TelegramNotifier(TOKEN, "42", session=s).send(["hi"])
    assert TOKEN not in str(ei.value) and "401" in str(ei.value)


def test_missing_secrets_fail_clearly():
    with pytest.raises(NotifyError, match="TELEGRAM_BOT_TOKEN"):
        TelegramNotifier("", "42", session=FakeSession([]))


def test_chat_ids_from_updates():
    body = {"ok": True, "result": [
        {"message": {"chat": {"id": 111, "type": "private", "first_name": "A"}}},
        {"message": {"chat": {"id": 111, "type": "private", "first_name": "A"}}},
        {"channel_post": {"chat": {"id": -100222, "type": "channel", "title": "Alerts"}}},
    ]}
    chats = TelegramNotifier(TOKEN, "x", session=FakeSession([FakeResp(200, body)])).chat_ids()
    assert chats == [{"id": 111, "type": "private", "name": "A"}, {"id": -100222, "type": "channel", "name": "Alerts"}]


# --- company names and the full watchlist -------------------------------------------------------


@pytest.mark.parametrize(
    "raw, short",
    [
        ("NVIDIA Corporation - Common Stock", "NVIDIA Corporation"),
        ("Agilent Technologies, Inc. Common Stock", "Agilent Technologies, Inc."),
        ("Ambev S.A. American Depositary Shares (Each representing 1 Common Share)", "Ambev S.A."),
        ("Berkshire Hathaway Inc. New Common Stock", "Berkshire Hathaway Inc."),
        ("A Very Long Company Name Holdings International Group plc", "A Very Long Company Name Holdin…"),
    ],
)
def test_short_name(raw, short):
    assert short_name(raw) == short
    assert len(short_name(raw)) <= 32


def test_entry_without_known_name_shows_ticker_only():
    res = scans.run_scans(panels_from({"X": textbook()}), CFG)
    d = diff_results(prev_state(), res, "us", CFG.alerts)
    text = "\n".join(format_alerts(d, res, CFG, UNIVERSE.iloc[0:0], chart_url))
    assert "<b>X</b>\n31.40 · trig" in text


def test_full_watchlist_lists_every_name_with_company():
    a, b = textbook(), textbook()
    b.set_bar(159, close=31.0)  # a bit further from its trigger, so it sorts second
    res = scans.run_scans(panels_from({"X": a, "BRK-B": b}), CFG)
    msgs = format_watchlist(res, CFG, UNIVERSE, chart_url)
    text = "\n".join(msgs)
    assert "<b>US watchlist ·" in text and "2 names, closest to trigger first" in text
    assert text.index("<b>X</b> · X Corp") < text.index("<b>BRK-B</b> · Berkshire")
    assert "symbol=NYSE:BRK.B" in text
    assert "not investment advice" in text
