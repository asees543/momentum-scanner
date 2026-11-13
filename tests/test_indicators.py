import math

import numpy as np
import pandas as pd
import pytest

from scanner import indicators as ind


def s(values, start="2025-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


# --- pct_return -------------------------------------------------------------------------------


def test_pct_return_hand_computed():
    close = s([10, 11, 12, 15])
    out = ind.pct_return(close, 2)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(12 / 10 - 1)  # 0.20
    assert out.iloc[3] == pytest.approx(15 / 11 - 1)


def test_pct_return_matches_spec_rule():
    # 22 bars at 100 then a close of 125 => exactly +25%, which passes "close >= 1.25 x close 22 ago"
    close = s([100.0] * 22 + [125.0])
    r = ind.pct_return(close, 22).iloc[-1]
    assert r == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_window_must_be_positive_int(bad):
    with pytest.raises(ValueError):
        ind.pct_return(s([1, 2, 3]), bad)


# --- adr_pct ----------------------------------------------------------------------------------


def test_adr_pct_hand_computed():
    high = s([110, 105, 120])
    low = s([100, 100, 100])
    out = ind.adr_pct(high, low, n=3)
    assert out.iloc[:2].isna().all()  # needs a full window
    # ratios 1.10, 1.05, 1.20 -> mean 1.116666... -> 11.6666...%
    assert out.iloc[2] == pytest.approx(100 * ((1.10 + 1.05 + 1.20) / 3 - 1))


def test_adr_pct_constant_range_default_window():
    high = s([105.0] * 25)
    low = s([100.0] * 25)
    out = ind.adr_pct(high, low)  # n = 20 default
    assert out.iloc[:19].isna().all()
    assert out.iloc[19:].tolist() == pytest.approx([5.0] * 6)


def test_adr_pct_zero_low_is_nan_not_inf():
    high = s([1.0, 1.0])
    low = s([0.0, 0.5])
    out = ind.adr_pct(high, low, n=1)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(100.0)


# --- sma --------------------------------------------------------------------------------------


def test_sma_hand_computed():
    out = ind.sma(s([1, 2, 3, 4, 5]), 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_sma_nan_inside_window_gives_nan():
    out = ind.sma(s([1, np.nan, 3, 4, 5]), 3)
    assert out.iloc[2:4].isna().all()  # windows containing the gap
    assert out.iloc[4] == pytest.approx(4.0)


# --- traded value / dollar volume ---------------------------------------------------------------


def test_traded_value_and_average():
    close = s([10, 20, 30])
    vol = s([1_000, 2_000, 3_000])
    assert ind.traded_value(close, vol).tolist() == [10_000, 40_000, 90_000]
    avg = ind.avg_traded_value(close, vol, n=2)
    assert math.isnan(avg.iloc[0])
    assert avg.iloc[1:].tolist() == pytest.approx([25_000, 65_000])


def test_dollar_volume_aliases():
    assert ind.dollar_volume is ind.traded_value
    assert ind.avg_dollar_volume is ind.avg_traded_value


# --- rs_rank_pct ------------------------------------------------------------------------------


def test_rs_rank_pct_top_7_of_100_keeps_exactly_7():
    values = pd.Series(np.arange(100, dtype=float), index=[f"S{i}" for i in range(100)])
    pct = ind.rs_rank_pct(values)
    assert pct["S99"] == 100
    assert pct["S0"] == 1
    kept = pct[pct > 100 - 7]
    assert sorted(kept.index) == sorted(f"S{i}" for i in range(93, 100))


def test_rs_rank_pct_ignores_nan_and_keeps_ties_together():
    values = pd.Series({"A": 0.5, "B": 0.5, "C": 0.1, "D": np.nan})
    pct = ind.rs_rank_pct(values)
    assert math.isnan(pct["D"])
    # N = 3 ranked names; A and B tie for the top and both get rank 3 -> 100
    assert pct["A"] == pct["B"] == 100
    assert pct["C"] == pytest.approx(100 / 3)


def test_rs_rank_pct_panel_ranks_each_row_across_symbols():
    panel = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 2.0], "C": [3.0, 1.0]})
    pct = ind.rs_rank_pct(panel)
    assert pct.loc[0].tolist() == pytest.approx([100 / 3, 200 / 3, 100])
    assert pct.loc[1].tolist() == pytest.approx([100, 200 / 3, 100 / 3])


# --- panel input gives the same answer as per-symbol input -------------------------------------


def test_panel_matches_series_for_every_time_series_indicator():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2025-01-01", periods=60)
    close = pd.DataFrame(rng.uniform(10, 20, (60, 3)), index=idx, columns=["A", "B", "C"])
    high = close * 1.03
    low = close * 0.97
    vol = pd.DataFrame(rng.integers(1_000, 9_000, (60, 3)), index=idx, columns=close.columns).astype(float)

    panel_results = {
        "ret": ind.pct_return(close, 22),
        "adr": ind.adr_pct(high, low, 20),
        "sma": ind.sma(close, 10),
        "atv": ind.avg_traded_value(close, vol, 20),
    }
    for col in close.columns:
        series_results = {
            "ret": ind.pct_return(close[col], 22),
            "adr": ind.adr_pct(high[col], low[col], 20),
            "sma": ind.sma(close[col], 10),
            "atv": ind.avg_traded_value(close[col], vol[col], 20),
        }
        for name, ser in series_results.items():
            pd.testing.assert_series_equal(panel_results[name][col], ser, check_names=False)
