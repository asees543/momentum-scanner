"""Load and validate per-market YAML config (config/<market>.yaml).

Every threshold the scans use comes from these models. Unknown keys are rejected so a typo in the
YAML fails loudly instead of silently falling back to a default.
"""

from __future__ import annotations

import re
from datetime import time
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class ConfigError(Exception):
    """Raised when a config file is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    provider: Literal["yfinance"] = "yfinance"
    history_years: int = Field(5, ge=1, le=30)
    adjustment: Literal["split"] = "split"
    cache_dir: str
    batch_size: int = Field(100, ge=1, le=1000)
    batch_pause_seconds: float = Field(1.0, ge=0)
    max_retries: int = Field(3, ge=0)
    incremental_overlap_bars: int = Field(10, ge=2)
    revision_tolerance_pct: float = Field(0.5, gt=0)
    universe_max_age_hours: float = Field(20, gt=0)
    timezone: str
    market_close: str
    close_buffer_minutes: int = Field(30, ge=0)
    # When the price provider has not published the newest sessions yet, fill them from another
    # source. "nse_bhavcopy" = NSE's official end-of-day file (India). "none" = wait for the provider.
    latest_day_fallback: Literal["none", "nse_bhavcopy"] = "none"
    fallback_max_days: int = Field(5, ge=1)  # look back at most this many calendar days for missing sessions

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as e:
            raise ValueError(f"unknown timezone {v!r}") from e
        return v

    @field_validator("market_close")
    @classmethod
    def _valid_close(cls, v: str) -> str:
        time.fromisoformat(v)  # raises ValueError on bad format
        return v

    @property
    def close_time(self) -> time:
        return time.fromisoformat(self.market_close)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def cache_path(self, root: Path = PROJECT_ROOT) -> Path:
        p = Path(self.cache_dir)
        return p if p.is_absolute() else root / p


class UniverseConfig(_Strict):
    exchanges: list[Literal["NASDAQ", "NYSE", "NYSE_AMERICAN", "NSE"]] = Field(min_length=1)
    series: list[str] = []  # NSE only: series to keep (e.g. EQ); empty = not applicable
    exclude_etfs: bool = True
    exclude_test_issues: bool = True
    exclude_symbol_patterns: list[str] = []
    exclude_name_patterns: list[str] = []

    @field_validator("exclude_symbol_patterns", "exclude_name_patterns")
    @classmethod
    def _valid_regex(cls, v: list[str]) -> list[str]:
        for pat in v:
            try:
                re.compile(pat)
            except re.error as e:
                raise ValueError(f"bad regex {pat!r}: {e}") from e
        return v


class LiquidityConfig(_Strict):
    min_price: float = Field(ge=0)
    # Average of close x volume over `window` days, in the market's currency:
    # "dollar volume" in the US, "turnover" in India.
    min_avg_traded_value: float = Field(ge=0)
    min_adr_pct: float = Field(ge=0)
    window: int = Field(20, ge=1)
    # Circuit-limit lock (SPEC India): a day "locked" at a price band traded in a range of at most
    # circuit_lock_max_range_pct while closing at least circuit_lock_min_move_pct from the prior close.
    # A stock locked on circuit_lock_consecutive_days consecutive sessions within the last
    # circuit_lock_lookback sessions fails Stage 1.
    exclude_circuit_locked: bool = False
    circuit_lock_min_move_pct: float = Field(1.9, gt=0)
    circuit_lock_max_range_pct: float = Field(0.1, ge=0)
    circuit_lock_consecutive_days: int = Field(2, ge=1)
    circuit_lock_lookback: int = Field(10, ge=1)


class WindowConfig(_Strict):
    lookback: int = Field(ge=1)  # bars back, e.g. 22 for ~1 month
    min_multiple: float = Field(gt=0)  # fixed mode: close >= min_multiple x close[lookback bars ago]


class LeadersConfig(_Strict):
    mode: Literal["fixed", "rank"]
    rank_top_pct: float = Field(gt=0, le=100)
    windows: dict[str, WindowConfig] = Field(min_length=1)
    target_count: tuple[int, int] = (25, 75)
    sort_by: str  # window name whose return orders rows (after number of scans passed)
    flag_day_move_multiple: float = Field(3.0, gt=1)  # data-quality flag, see us.yaml

    @model_validator(mode="after")
    def _check(self) -> LeadersConfig:
        lo, hi = self.target_count
        if lo > hi:
            raise ValueError(f"target_count low ({lo}) > high ({hi})")
        if self.sort_by not in self.windows:
            raise ValueError(f"sort_by {self.sort_by!r} is not one of the windows {list(self.windows)}")
        return self


class SetupConfig(_Strict):
    """Stage 3: orderly consolidation (builds the watchlist). See us.yaml for each rule."""

    leader_lookback_sessions: int = Field(20, ge=1)
    require_liquid_today: bool = True  # Stage 1 must also pass on the setup / watchlist day
    sma_trend: int = Field(50, ge=1)
    sma_fast: int = Field(10, ge=1)
    sma_slow: int = Field(20, ge=1)
    high_window: int = Field(20, ge=1)
    max_below_high_pct: float = Field(15, ge=0, lt=100)
    higher_lows_bars: int = Field(5, ge=1)
    tight_min_days: int = Field(3, ge=1)
    tight_max_days: int = Field(5, ge=1)
    tight_vs_adr: float = Field(1.0, gt=0)
    dryup_max_volume_ratio: float = Field(0.70, gt=0)
    volume_avg_window: int = Field(20, ge=1)
    trigger_bars: int = Field(10, ge=1)

    @model_validator(mode="after")
    def _check(self) -> SetupConfig:
        if self.tight_min_days > self.tight_max_days:
            raise ValueError("tight_min_days must be <= tight_max_days")
        if self.sma_fast >= self.sma_slow:
            raise ValueError("sma_fast must be shorter than sma_slow")
        return self


class BreakoutConfig(_Strict):
    """Stage 4: breakout trigger on a recent watchlist name."""

    watchlist_memory_sessions: int = Field(5, ge=1)
    min_volume_ratio: float = Field(1.5, gt=0)
    max_risk_vs_adr: float = Field(1.0, gt=0)


class AlertsConfig(_Strict):
    """Phase 3 alerting: what counts as new, and when not to trust a run."""

    min_active_ratio: float = Field(0.9, gt=0, le=1)  # data-health guard, see us.yaml
    baseline_first_run: bool = True  # first run records the watchlist instead of alerting all of it
    list_wide_breakouts: bool = True  # also name breakouts whose stop is wider than ADR (own section)
    catch_up_max_sessions: int = Field(5, ge=0)  # report breakouts of up to this many missed sessions


class MarketConfig(_Strict):
    market: str
    currency: str
    data: DataConfig
    universe: UniverseConfig
    liquidity: LiquidityConfig
    leaders: LeadersConfig
    setup: SetupConfig
    breakout: BreakoutConfig
    alerts: AlertsConfig


def load_config(market: str, config_dir: Path = CONFIG_DIR) -> MarketConfig:
    """Read config/<market>.yaml and validate it. Raises ConfigError with a readable message."""
    path = Path(config_dir) / f"{market}.yaml"
    if not path.exists():
        raise ConfigError(f"No config for market {market!r}: {path} not found")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at top level")
    try:
        cfg = MarketConfig.model_validate(raw)
    except ValidationError as e:
        lines = [f"{path}: invalid config"]
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from e
    if cfg.market != market:
        raise ConfigError(f"{path}: 'market' is {cfg.market!r}, expected {market!r}")
    return cfg
