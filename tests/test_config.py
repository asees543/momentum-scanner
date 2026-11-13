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
