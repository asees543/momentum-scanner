"""Data-quality report on synthetic India series with planted problems."""

import numpy as np
import pandas as pd
import pytest

from scanner.config import load_config
from scanner.quality import nearest_clean_ratio, quality_report
from tests.synth import Bars, panels_from

INDIA = load_config("india")
N = 160


def clean() -> Bars:
    return Bars.from_close(np.linspace(200, 240, N), 6.0, 1_000_000)


def unadjusted_bonus() -> Bars:
    """1:1 bonus Yahoo has not adjusted: from bar 140 the price halves and share volume doubles."""
    b = clean()
    for i in range(140, N):
        b.set_bar(i, close=b.close[i] / 2, range_pct=6.0, volume=2_000_000)
    return b


def report(bars: dict, **kw):
    p = panels_from(bars)
    return quality_report(p, INDIA, **kw), p


def test_clean_ratio_labels():
    assert nearest_clean_ratio(0.5).startswith("1/2")
    assert nearest_clean_ratio(0.52).startswith("1/2")  # within 6%
    assert nearest_clean_ratio(0.1).startswith("1/10")
    assert nearest_clean_ratio(10.3).startswith("10 ")
    assert nearest_clean_ratio(0.75) == ""
    assert nearest_clean_ratio(float("nan")) == ""


def test_unadjusted_bonus_is_reported_with_split_signature():
    rep, p = report({"BONUS": unadjusted_bonus(), "OK": clean()},
                    split_lookup=lambda s: [p_dates[140]] if s == "BONUS" else [])
    j = rep.jumps
    assert j["symbol"].tolist() == ["BONUS"]
    row = j.iloc[0]
    assert row["date"] == p.dates[140]
    assert row["price_ratio"] == pytest.approx(0.5, rel=0.01)
    assert row["clean_ratio"].startswith("1/2")
    assert row["vol_change"] == pytest.approx(2.0)  # price / 2, volume x 2
    assert row["liquid"]
    assert row["yahoo_split"] == "yes"


p_dates = panels_from({"X": clean()}).dates  # shared calendar for the split_lookup above


def test_missing_frozen_zero_volume_and_stale():
    gap = clean()
    gap.close[120] = gap.high[120] = gap.low[120] = gap.volume[120] = np.nan  # one missing session
    frozen = clean()
    for i in (150, 151, 152):  # three sessions frozen at the upper band
        frozen.set_bar(i, close=frozen.close[i - 1] * 1.05, range_pct=0.0)
    zero = clean().set_bar(130, volume=0.0)
    stale = clean()
    stale.close[-1] = stale.high[-1] = stale.low[-1] = stale.volume[-1] = np.nan

    rep, p = report({"GAP": gap, "FROZEN": frozen, "ZERO": zero, "STALE": stale, "OK": clean()})
    miss = rep.missing.set_index("symbol")
    assert miss.loc["GAP", "missing_sessions"] == 1 and miss.loc["GAP", "first_gap"] == p.dates[120]
    assert "STALE" not in miss.index  # a missing LAST bar is "stale", not a gap
    fr = rep.frozen.set_index("symbol")
    assert fr.loc["FROZEN", "longest_streak"] == 3 and fr.loc["FROZEN", "last_frozen"] == p.dates[152]
    assert rep.zero_volume.set_index("symbol").loc["ZERO", "zero_volume_days"] == 1
    assert rep.stale["symbol"].tolist() == ["STALE"]
    assert rep.jumps.empty  # nothing moved more than +30% / -23%
    assert set(miss.index) | set(fr.index) <= {"GAP", "FROZEN"}  # the clean series is never reported
