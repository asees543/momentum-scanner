"""Synthetic price series for setup/breakout tests.

`textbook()` is a hand-checked Qullamaggie-style setup, 160 sessions long:
  bars   0..119  flat at 20, 8% daily range, volume 1.0M
  bars 119..149  run-up 20 -> 32 (+60%), 8% range, volume 1.5M          (a 1M and 3M leader)
  bars 150..159  base: closes 31.0, 30.6, 30.8, 30.9, 31.0, 31.1, 31.2, 31.2, 31.3, 31.4
                 range 5% on bars 150-154, 2.5% on bars 155-159; volume 0.6M
On bar 159 every Stage 3 rule holds (worked through in tests/test_setup.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scanner.config import MarketConfig, load_config
from scanner.data.base import Panels
from scanner import scans

BASE_CLOSES = [31.0, 30.6, 30.8, 30.9, 31.0, 31.1, 31.2, 31.2, 31.3, 31.4]


@dataclass
class Bars:
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_close(cls, close, range_pct, volume) -> Bars:
        """high/low straddle the close so that high/low - 1 == range_pct / 100 exactly."""
        c = np.asarray(close, dtype=float)
        r = np.broadcast_to(np.asarray(range_pct, dtype=float), c.shape)
        high = c * (1 + r / 200)
        low = high / (1 + r / 100)
        return cls(c, high, low, np.broadcast_to(np.asarray(volume, dtype=float), c.shape).copy())

    def copy(self) -> Bars:
        return Bars(self.close.copy(), self.high.copy(), self.low.copy(), self.volume.copy())

    def set_bar(self, i: int, close=None, range_pct=None, high=None, low=None, volume=None) -> Bars:
        if close is not None or range_pct is not None:
            c = self.close[i] if close is None else close
            r = 100 * (self.high[i] / self.low[i] - 1) if range_pct is None else range_pct
            one = Bars.from_close([c], [r], [self.volume[i]])
            self.close[i], self.high[i], self.low[i] = one.close[0], one.high[0], one.low[0]
        if high is not None:
            self.high[i] = high
        if low is not None:
            self.low[i] = low
        if volume is not None:
            self.volume[i] = volume
        return self

    def append(self, close: float, high: float, low: float, volume: float) -> Bars:
        self.close = np.append(self.close, close)
        self.high = np.append(self.high, high)
        self.low = np.append(self.low, low)
        self.volume = np.append(self.volume, volume)
        return self

    def __len__(self) -> int:
        return len(self.close)


def textbook() -> Bars:
    n = 160
    c = np.empty(n)
    c[:120] = 20.0
    c[119:150] = np.linspace(20.0, 32.0, 31)
    c[150:] = BASE_CLOSES
    r = np.full(n, 8.0)
    r[150:155] = 5.0
    r[155:] = 2.5
    v = np.full(n, 1_000_000.0)
    v[120:150] = 1_500_000.0
    v[150:] = 600_000.0
    return Bars.from_close(c, r, v)


def panels_from(symbols: dict[str, Bars]) -> Panels:
    n = max(len(b) for b in symbols.values())
    dates = pd.bdate_range("2026-03-02", periods=n)

    def frame(attr: str) -> pd.DataFrame:
        cols = {}
        for sym, b in symbols.items():
            arr = np.full(n, np.nan)
            arr[n - len(b):] = getattr(b, attr)
            cols[sym] = arr
        return pd.DataFrame(cols, index=dates)

    return Panels(open=frame("close"), high=frame("high"), low=frame("low"), close=frame("close"),
                  volume=frame("volume"))


def us_cfg(**section_updates) -> MarketConfig:
    """SPEC defaults from config/us.yaml, with optional per-section overrides:
    us_cfg(setup={"leader_lookback_sessions": 5}, leaders={"windows": {...}})."""
    cfg = load_config("us")
    updates = {}
    for section, fields in section_updates.items():
        updates[section] = getattr(cfg, section).model_copy(update=fields)
    return cfg.model_copy(update=updates)


def frames(p: Panels, cfg: MarketConfig):
    liq = scans.liquidity_frames(p, cfg.liquidity)
    lf = scans.leader_frames(p, liq, cfg.leaders, cfg.leaders.mode)
    sf = scans.setup_frames(p, liq, lf, cfg.setup)
    return liq, lf, sf
