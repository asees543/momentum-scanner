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
    assert "🚀 <b>Breakouts</b>" in text and "<b>X</b> 33.00" in text
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


def test_dry_run_requires_alert():
    assert main(["--market", "us", "--no-update", "--dry-run"]) == 2
