from pathlib import Path

import pytest
import yaml

from scanner.config import CONFIG_DIR, ConfigError, load_config


def test_us_config_loads_with_spec_defaults():
    cfg = load_config("us")
    assert cfg.market == "us"
    assert cfg.liquidity.min_price == 1.0
    assert cfg.liquidity.min_avg_traded_value == 3_000_000
    assert cfg.liquidity.min_adr_pct == 5.0
    assert cfg.liquidity.window == 20
    w = cfg.leaders.windows
    assert (w["1m"].lookback, w["1m"].min_multiple) == (22, 1.25)
    assert (w["3m"].lookback, w["3m"].min_multiple) == (67, 1.50)
    assert (w["6m"].lookback, w["6m"].min_multiple) == (126, 2.50)
    assert cfg.leaders.mode == "fixed"
    assert cfg.leaders.rank_top_pct == 7
    assert cfg.leaders.target_count == (25, 75)
    assert cfg.data.close_time.hour == 16
    st, bo = cfg.setup, cfg.breakout
    assert (st.sma_trend, st.sma_fast, st.sma_slow) == (50, 10, 20)
    assert (st.high_window, st.max_below_high_pct) == (20, 15)
    assert (st.higher_lows_bars, st.tight_min_days, st.tight_max_days, st.tight_vs_adr) == (5, 3, 5, 1.0)
    assert (st.dryup_max_volume_ratio, st.volume_avg_window, st.trigger_bars) == (0.70, 20, 10)
    assert st.leader_lookback_sessions == 20
    assert st.require_liquid_today is True
    assert cfg.alerts.list_wide_breakouts is True
    assert (bo.watchlist_memory_sessions, bo.min_volume_ratio, bo.max_risk_vs_adr) == (5, 1.5, 1.0)


def _write_variant(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load((CONFIG_DIR / "us.yaml").read_text())
    mutate(raw)
    (tmp_path / "us.yaml").write_text(yaml.safe_dump(raw))
    return tmp_path


def test_typo_key_is_rejected(tmp_path):
    def mutate(raw):
        raw["liquidity"]["min_prise"] = 5  # typo

    with pytest.raises(ConfigError, match="min_prise"):
        load_config("us", _write_variant(tmp_path, mutate))


def test_wrong_type_is_rejected(tmp_path):
    def mutate(raw):
        raw["leaders"]["windows"]["1m"]["lookback"] = "twenty-two"

    with pytest.raises(ConfigError, match="lookback"):
        load_config("us", _write_variant(tmp_path, mutate))


def test_bad_mode_is_rejected(tmp_path):
    def mutate(raw):
        raw["leaders"]["mode"] = "vibes"

    with pytest.raises(ConfigError, match="mode"):
        load_config("us", _write_variant(tmp_path, mutate))


def test_bad_regex_is_rejected(tmp_path):
    def mutate(raw):
        raw["universe"]["exclude_name_patterns"] = ["(unclosed"]

    with pytest.raises(ConfigError, match="bad regex"):
        load_config("us", _write_variant(tmp_path, mutate))


def test_missing_market_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config("mars", tmp_path)


def test_setup_rejects_inverted_tight_days(tmp_path):
    def mutate(raw):
        raw["setup"]["tight_min_days"] = 6

    with pytest.raises(ConfigError, match="tight_min_days"):
        load_config("us", _write_variant(tmp_path, mutate))


def test_india_config_loads_with_spec_values():
    cfg = load_config("india")
    assert cfg.market == "india" and cfg.currency == "INR"
    assert cfg.universe.exchanges == ["NSE"] and cfg.universe.series == ["EQ"]
    assert cfg.liquidity.min_price == 100
    assert cfg.liquidity.min_avg_traded_value == 5 * 10**7  # Rs 5 crore
    assert cfg.liquidity.min_adr_pct == 4.0
    assert cfg.liquidity.exclude_circuit_locked is True
    assert cfg.leaders.mode == "rank" and cfg.leaders.rank_top_pct == 7
    assert cfg.leaders.flag_day_move_multiple == 1.3
    assert cfg.data.timezone == "Asia/Kolkata" and cfg.data.close_time.hour == 15
    assert cfg.alerts.list_wide_breakouts is True
    assert load_config("us").liquidity.exclude_circuit_locked is False
