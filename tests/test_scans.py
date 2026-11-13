import numpy as np
import pandas as pd
import pytest

from scanner.config import LeadersConfig, LiquidityConfig, WindowConfig
from scanner.data.base import Panels
from scanner.scans import liquidity_snapshot, momentum_leaders

L = 140  # sessions of synthetic history
DATES = pd.bdate_range("2026-04-01", periods=L)

LIQ = LiquidityConfig(min_price=1.0, min_avg_traded_value=3_000_000, min_adr_pct=5.0, window=20)
LEAD = LeadersConfig(
    mode="fixed",
    rank_top_pct=7,
    windows={
        "1m": WindowConfig(lookback=22, min_multiple=1.25),
        "3m": WindowConfig(lookback=67, min_multiple=1.50),
        "6m": WindowConfig(lookback=126, min_multiple=2.50),
    },
    target_count=(25, 75),
    sort_by="3m",
    flag_day_move_multiple=3.0,
)


def path(anchors: dict[int, float]) -> np.ndarray:
    """Close series through anchor points given as {bars_before_last: price}; linear in between,
    flat before the oldest anchor."""
    xs = sorted((L - 1 - k for k in anchors), key=int)
    ys = [anchors[L - 1 - x] for x in xs]
    return np.interp(np.arange(L), xs, ys)


def build(symbols: dict[str, dict]) -> Panels:
    """symbols[name] = dict(close=array, adr=pct, shares=volume per bar, start=first bar index,
    drop_last=bool)."""
    fields = {f: {} for f in ["open", "high", "low", "close", "volume"]}
    for name, spec in symbols.items():
        c = np.asarray(spec["close"], dtype=float).copy()
        adr = spec.get("adr", 6.0)
        # high/low = 1 + adr/100 exactly, so ADR% equals `adr` (SPEC formula); close sits inside.
        high = c * (1 + adr / 200)
        low = high / (1 + adr / 100)
        vol = np.full(L, float(spec.get("shares", 1_000_000)))
        start = spec.get("start", 0)
        for arr in (c, high, low, vol):
            arr[:start] = np.nan
        if spec.get("drop_last"):
            for arr in (c, high, low, vol):
                arr[-1] = np.nan
        fields["open"][name] = c
        fields["high"][name] = high
        fields["low"][name] = low
        fields["close"][name] = c
        fields["volume"][name] = vol
    return Panels(**{f: pd.DataFrame(v, index=DATES) for f, v in fields.items()})


def standard_universe() -> Panels:
    return build(
        {
            # exactly +25% over 22 bars -> passes 1m on the boundary (>= is inclusive); flat before
            "EDGE": {"close": path({0: 125.0, 22: 100.0})},
            # one cent short of the boundary
            "MISS": {"close": path({0: 124.99, 22: 100.0})},
            # passes all three: 26/20 = 1.3, 26/15 = 1.73, 26/10 = 2.6
            "ALL3": {"close": path({0: 26.0, 22: 20.0, 67: 15.0, 126: 10.0}), "shares": 5_000_000},
            # 3m and 6m but not 1m: 30/28 = 1.07, 30/18 = 1.67, 30/11 = 2.73
            "OLDER": {"close": path({0: 30.0, 22: 28.0, 67: 18.0, 126: 11.0}), "shares": 5_000_000},
            # would pass 1m but trades only ~$12.5k/day
            "ILLIQ": {"close": path({0: 125.0, 22: 100.0}), "shares": 100},
            # would pass 1m but ADR% is 4
            "LOWADR": {"close": path({0: 125.0, 22: 100.0}), "adr": 4.0},
            # would pass 1m but trades under $1
            "CHEAP": {"close": path({0: 0.95, 22: 0.70}), "shares": 50_000_000},
            # listed 30 bars ago: enough history for 1m only
            "IPO": {"close": path({0: 50.0, 22: 30.0}), "start": L - 30},
            # no bar on the as-of date
            "STALE": {"close": path({0: 125.0, 22: 100.0}), "drop_last": True},
        }
    )


def test_liquidity_snapshot_filters():
    snap = liquidity_snapshot(standard_universe(), LIQ)
    assert "STALE" not in snap.index
    assert snap.loc["LOWADR", "adr_pct"] == pytest.approx(4.0)
    assert snap.loc["EDGE", "adr_pct"] == pytest.approx(6.0)
    liquid = set(snap.index[snap["liquid"]])
    assert liquid == {"EDGE", "MISS", "ALL3", "OLDER", "IPO"}


def test_liquidity_boundaries_are_inclusive():
    # avg traded value exactly 3,000,000 and ADR exactly 5.0 and price exactly 1.0 -> liquid
    p = build({"X": {"close": np.full(L, 1.0), "shares": 3_000_000, "adr": 5.0}})
    snap = liquidity_snapshot(p, LIQ)
    assert snap.loc["X", "avg_traded_value"] == pytest.approx(3_000_000)
    assert bool(snap.loc["X", "liquid"])


def test_fixed_mode_leaders():
    res = momentum_leaders(standard_universe(), LIQ, LEAD)
    t = res.table
    assert res.as_of == DATES[-1]
    assert res.stale_count == 1 and res.active_count == 8 and res.liquid_count == 5
    assert set(t.index) == {"EDGE", "ALL3", "OLDER", "IPO"}
    assert res.counts == {"1m": 3, "3m": 2, "6m": 2}  # 1m: EDGE, ALL3, IPO
    assert t.loc["EDGE", "ret_1m"] == pytest.approx(0.25)
    assert t.loc["EDGE", ["pass_1m", "pass_3m", "pass_6m"]].tolist() == [True, False, False]
    assert t.loc["ALL3", "n_passed"] == 3
    assert t.loc["OLDER", ["pass_1m", "pass_3m", "pass_6m"]].tolist() == [False, True, True]
    assert t.loc["IPO", "pass_1m"] and np.isnan(t.loc["IPO", "ret_3m"])
    # sorted by number of scans passed, then 3m return
    assert t.index.tolist()[:2] == ["ALL3", "OLDER"]


def test_no_lookahead_with_as_of():
    p = standard_universe()
    # EDGE reaches +25% only on the last bar; one session earlier it is just short
    res = momentum_leaders(p.as_of(DATES[-2]), LIQ, LEAD)
    assert res.as_of == DATES[-2]
    assert "EDGE" not in res.table.index


def test_rank_mode_keeps_top_7_percent():
    # 100 liquid symbols; 1m return increases with the symbol number
    syms = {f"S{i:03d}": {"close": path({0: 100.0 + i, 22: 100.0})} for i in range(100)}
    res = momentum_leaders(build(syms), LIQ, LEAD, mode="rank")
    assert res.mode == "rank"
    assert res.counts["1m"] == 7
    passed_1m = set(res.table.index[res.table["pass_1m"]])
    assert passed_1m == {f"S{i:03d}" for i in range(93, 100)}
    assert res.table.loc["S099", "rs_1m"] == 100


def test_rank_mode_ignores_illiquid_names():
    syms = {f"S{i:03d}": {"close": path({0: 100.0 + i, 22: 100.0})} for i in range(100)}
    syms["ROCKET"] = {"close": path({0: 500.0, 22: 100.0}), "shares": 10}  # huge return, illiquid
    res = momentum_leaders(build(syms), LIQ, LEAD, mode="rank")
    assert "ROCKET" not in res.table.index
    assert res.counts["1m"] == 7


def test_empty_panels_raise():
    e = pd.DataFrame()
    with pytest.raises(ValueError):
        momentum_leaders(Panels(e, e, e, e, e), LIQ, LEAD)


def test_one_day_jump_is_flagged():
    # Looks like an un-adjusted 1:10 reverse split: 0.20 -> 2.00 overnight, 10 bars ago.
    c = np.r_[np.full(L - 10, 0.20), np.full(10, 2.00)]
    p = build({"JUMP": {"close": c, "shares": 50_000_000}, "EDGE": {"close": path({0: 125.0, 22: 100.0})}})
    t = momentum_leaders(p, LIQ, LEAD).table
    assert t.loc["JUMP", "max_day_move"] == pytest.approx(10.0)
    assert bool(t.loc["JUMP", "flag_day_move"])
    assert not bool(t.loc["EDGE", "flag_day_move"])


def test_one_day_crash_is_flagged_too():
    c = np.r_[np.full(L - 40, 10.0), np.full(10, 2.0), np.linspace(2.0, 4.0, 30)]
    p = build({"DROP": {"close": c, "shares": 5_000_000}})
    t = momentum_leaders(p, LIQ, LEAD).table
    assert t.loc["DROP", "max_day_move"] == pytest.approx(5.0)
    assert bool(t.loc["DROP", "flag_day_move"])
