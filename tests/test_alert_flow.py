"""End-to-end alert flow on synthetic data: diff -> messages -> delivery -> state, with a fake sender."""

import io

import pandas as pd
import pytest

from scanner import scans
from scanner.data.us_loader import chart_url
from scanner.notify import NotifyError, PrintNotifier
from scanner.run import alert_flow, main
from scanner.state import MarketState, state_path
from tests.synth import panels_from, textbook, us_cfg

CFG = us_cfg()
UNIVERSE = pd.DataFrame({"symbol": ["X"], "source_symbol": ["X"], "name": ["X Corp"], "exchange": ["NASDAQ"]})


class Capture:
    def __init__(self, fail=False):
        self.sent: list[list[str]] = []
        self.fail = fail

    def send(self, messages):
        if self.fail:
            raise NotifyError("Telegram sendMessage failed (502): Bad Gateway")
        self.sent.append(list(messages))


def results_for(bars):
    return scans.run_scans(panels_from({"X": bars}), CFG)


def flow(res, tmp_path, notifier, dry_run=False):
    return alert_flow(res, CFG, UNIVERSE, chart_url, dry_run, tmp_path, notifier)


def test_three_day_sequence(tmp_path):
    day1 = textbook()  # bar 159: X qualifies (watchlist)
    day2 = day1.copy().append(33.0, 33.3, 31.9, 2_000_000)  # bar 160: breakout
    path = state_path("us", tmp_path)

    cap = Capture()
    assert flow(results_for(day1), tmp_path, cap) == 0  # first run: baseline
    assert len(cap.sent) == 1 and "recorded as the baseline" in cap.sent[0][0]
    assert "X" in MarketState.load(path).watchlist

    cap = Capture()
    assert flow(results_for(day1), tmp_path, cap) == 0  # rerun, same market date: silent
    assert cap.sent == []

    cap = Capture()
    assert flow(results_for(day2), tmp_path, cap) == 0  # next session: breakout alert
    text = "\n".join(cap.sent[0])
    assert "🚀 <b>Breakouts</b>" in text and "<b>X</b> · X Corp\n33.00 › trig" in text
    st = MarketState.load(path)
    assert st.breakouts == ["X"] and "X" not in st.watchlist  # closed above trigger: left the watchlist


def test_dry_run_prints_and_never_saves(tmp_path):
    out = io.StringIO()
    assert flow(results_for(textbook()), tmp_path, PrintNotifier(out), dry_run=True) == 0
    assert "nothing sent" in out.getvalue()
    assert not state_path("us", tmp_path).exists()


def test_failed_delivery_does_not_save_state(tmp_path):
    with pytest.raises(NotifyError):
        flow(results_for(textbook()), tmp_path, Capture(fail=True))
    assert not state_path("us", tmp_path).exists()


def test_unhealthy_data_warns_and_keeps_state(tmp_path):
    day1 = textbook()
    flow(results_for(day1), tmp_path, Capture())
    st = MarketState.load(state_path("us", tmp_path))
    st.active_count = 100  # pretend yesterday had 100 symbols; today only 1
    st.save(state_path("us", tmp_path))
    cap = Capture()
    day2 = day1.copy().append(31.5, 31.7, 31.2, 500_000)
    assert flow(results_for(day2), tmp_path, cap) == 1
    assert "No alerts sent; state unchanged" in cap.sent[0][0]
    assert MarketState.load(state_path("us", tmp_path)).as_of == st.as_of


@pytest.mark.parametrize("extra", [["--symbols", "NVDA"], ["--limit", "5"], ["--as-of", "2026-11-01"]])
def test_real_alert_refuses_partial_runs(extra):
    assert main(["--market", "us", "--no-update", "--alert", *extra]) == 2


def test_dry_run_requires_alert_or_send_watchlist():
    assert main(["--market", "us", "--no-update", "--dry-run"]) == 2


# --- catch-up: sessions missed since the last sent run ------------------------------------------


def breakout_then_quiet():
    b = textbook().append(33.0, 33.3, 31.9, 2_000_000)  # bar 160: breakout
    return b.append(33.2, 33.5, 32.9, 1_200_000)  # bar 161: quiet day above the trigger


def test_missed_breakouts_finds_the_skipped_session():
    p = panels_from({"X": breakout_then_quiet()})
    found, skipped = scans.missed_breakouts(p, CFG, p.dates[159], p.dates[161], max_sessions=5)
    assert skipped == 0
    assert [d for d, _ in found] == [p.dates[160]]
    assert found[0][1].table.loc["X", "status"] == "ok"
    found, skipped = scans.missed_breakouts(p, CFG, p.dates[159], p.dates[161], max_sessions=0)
    assert found == [] and skipped == 1


def test_alert_after_a_missed_session_reports_its_breakout(tmp_path):
    bars = breakout_then_quiet()
    p = panels_from({"X": bars})
    flow(scans.run_scans(p.as_of(p.dates[159]), CFG), tmp_path, Capture())  # last sent run: bar 159
    cap = Capture()
    rc = alert_flow(scans.run_scans(p, CFG), CFG, UNIVERSE, chart_url, False, tmp_path, cap, panels=p)
    assert rc == 0
    text = "\n".join(cap.sent[0])
    assert f"⏪ <b>Breakouts on {p.dates[160]:%a %d %b}</b>" in text
    assert "<b>X</b> · X Corp\n33.00 › trig 31.79 · stop 31.90" in text
    assert "Nothing new today." not in text
    assert MarketState.load(state_path("us", tmp_path)).as_of == p.dates[161].date()


def test_no_catch_up_section_when_nothing_was_missed(tmp_path):
    bars = breakout_then_quiet()
    p = panels_from({"X": bars})
    flow(scans.run_scans(p.as_of(p.dates[160]), CFG), tmp_path, Capture())  # last sent run: bar 160
    cap = Capture()
    alert_flow(scans.run_scans(p, CFG), CFG, UNIVERSE, chart_url, False, tmp_path, cap, panels=p)
    assert "⏪" not in "\n".join(cap.sent[0])
