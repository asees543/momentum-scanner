"""India adapter (NSE universe, Yahoo/TradingView symbols) and the circuit-lock Stage 1 rule."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scanner import indicators as ind
from scanner import scans
from scanner.config import load_config
from scanner.data import india_loader
from tests.synth import Bars, panels_from

FIX = Path(__file__).parent / "fixtures"
NSE_TXT = (FIX / "nse_equity_sample.csv").read_text()
INDIA = load_config("india")


# --- universe ---------------------------------------------------------------------------------


def test_parse_strips_padded_headers_and_values():
    df = india_loader.parse_equity_list(NSE_TXT)
    assert set(df.columns) == {"symbol", "source_symbol", "name", "exchange", "series"}
    assert dict(zip(df["source_symbol"], df["series"]))["21STCENMGM"] == "BE"
    assert set(df["exchange"]) == {"NSE"}


def test_filter_keeps_eq_series_only():
    uni = india_loader.filter_universe(india_loader.parse_equity_list(NSE_TXT), INDIA.universe)
    assert uni["symbol"].tolist() == [
        "20MICRONS.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "J&KBANK.NS", "M&M.NS", "RELIANCE.NS"
    ]  # BE (21STCENMGM) and BZ (AGSTRA) dropped
    assert uni.set_index("symbol").loc["M&M.NS", "name"] == "Mahindra & Mahindra Limited"


def test_yahoo_and_tradingview_symbols():
    assert india_loader.to_yahoo_symbol("M&M") == "M&M.NS"
    url = india_loader.chart_url
    assert url("RELIANCE.NS", "NSE", "RELIANCE").endswith("symbol=NSE:RELIANCE")
    assert url("M&M.NS", "NSE", "M&M").endswith("symbol=NSE:M_M")
    assert url("BAJAJ-AUTO.NS", "NSE", "BAJAJ-AUTO").endswith("symbol=NSE:BAJAJ_AUTO")
    assert url("J&KBANK.NS").endswith("symbol=NSE:J_KBANK")  # no universe row: strip .NS


def test_load_universe_caches(tmp_path):
    cfg = INDIA.model_copy(update={"data": INDIA.data.model_copy(update={"cache_dir": str(tmp_path)})})
    calls = []

    def fetch(url):
        calls.append(url)
        return NSE_TXT

    t0 = datetime(2026, 11, 13, 12, 30, tzinfo=timezone.utc)
    uni = india_loader.load_universe(cfg, now=t0, fetch_text=fetch)
    assert len(uni) == 6 and list(uni.columns) == ["symbol", "source_symbol", "name", "exchange"]
    india_loader.load_universe(cfg, now=t0 + timedelta(hours=1), fetch_text=fetch)
    assert len(calls) == 1  # served from universe.csv


# --- circuit lock -----------------------------------------------------------------------------


def test_run_length():
    m = pd.Series([True, True, False, True, True, True, False])
    assert ind.run_length(m).tolist() == [1, 2, 0, 1, 2, 3, 0]


def test_circuit_locked_needs_frozen_range_and_band_move():
    close = pd.Series([100.0, 105.0, 106.0, 112.0, 113.0])
    high = pd.Series([101.0, 105.0, 106.0, 112.5, 113.0])
    low = pd.Series([99.0, 105.0, 106.0, 111.5, 113.0])
    locked = ind.circuit_locked(high, low, close, min_move_pct=1.9, max_range_pct=0.1)
    # bar 1: frozen at +5% -> locked; bar 2: frozen but only +0.95% -> not; bar 3: +5.7% but traded
    # in a 0.9% range -> not; bar 4: frozen at +0.9% -> not
    assert locked.tolist() == [False, True, False, False, False]


N = 60


def bars_locked_at(days, move=1.05, rng=0.0) -> Bars:
    """Liquid India stock (~Rs 200, 6% daily range, Rs 20 crore/day) with chosen days frozen at a band."""
    b = Bars.from_close(np.linspace(200, 230, N), 6.0, 1_000_000)
    for i in sorted(days):
        c = b.close[i - 1] * move
        b.set_bar(i, close=c, range_pct=rng)
    return b


@pytest.mark.parametrize(
    "days, move, rng, stuck",
    [
        ([57, 58], 1.05, 0.0, True),  # two locked days in a row, 1-2 sessions ago
        ([58], 1.05, 0.0, False),  # one locked day is not "consecutive days"
        ([40, 41], 1.05, 0.0, False),  # streak ended 18 sessions ago, outside the 10-session lookback
        ([57, 58], 1.01, 0.0, False),  # frozen but only +1%: not a band move
        ([57, 58], 1.05, 0.5, False),  # +5% but traded in a 0.5% range: not frozen
        ([56, 57, 58, 59], 0.95, 0.0, True),  # lower circuit works too
    ],
)
def test_circuit_stuck_stocks_fail_stage1(days, move, rng, stuck):
    p = panels_from({"X": bars_locked_at(days, move, rng)})
    liq = scans.liquidity_frames(p, INDIA.liquidity)
    assert bool(liq.circuit_stuck["X"].iloc[-1]) is stuck
    assert bool(liq.liquid["X"].iloc[-1]) is (not stuck)


def test_unlocked_stock_is_liquid_under_india_thresholds():
    p = panels_from({"X": bars_locked_at([])})
    snap = scans.liquidity_snapshot(p, INDIA.liquidity)
    assert snap.loc["X", "close"] >= 100 and snap.loc["X", "avg_traded_value"] >= 5e7
    assert bool(snap.loc["X", "liquid"])


def test_circuit_rule_is_off_for_us():
    p = panels_from({"X": bars_locked_at([57, 58])})
    assert scans.liquidity_frames(p, load_config("us").liquidity).circuit_stuck is None
