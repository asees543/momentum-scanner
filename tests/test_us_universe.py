from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scanner.config import load_config
from scanner.data import us_loader

FIX = Path(__file__).parent / "fixtures"
NASDAQ_TXT = (FIX / "nasdaqlisted_sample.txt").read_text()
OTHER_TXT = (FIX / "otherlisted_sample.txt").read_text()

# Kept on purpose (regressions from the real files): PFBC "Preferred Bank", OBAI "Our Bond, Inc.",
# RLX ADR "right to receive", BNS "...Pfd 3 Ordinary Shares", OZ LLC "Class A Units", ET MLP "Common Units".
EXPECTED = sorted(
    ["A", "AA", "AACG", "AAPL", "ABEV", "ACCS", "AGM-A", "BRK-B", "ET", "NVDA", "ZYN",
     "PFBC", "OBAI", "RLX", "BNS", "OZ"]
)


def test_parsers_drop_trailer_and_map_exchanges():
    nas = us_loader.parse_nasdaq_listed(NASDAQ_TXT)
    oth = us_loader.parse_other_listed(OTHER_TXT)
    assert not nas["source_symbol"].str.startswith("File Creation").any()
    assert not oth["source_symbol"].str.startswith("File Creation").any()
    assert set(nas["exchange"]) == {"NASDAQ"}
    ex = dict(zip(oth["source_symbol"], oth["exchange"]))
    assert ex["A"] == "NYSE"
    assert ex["ACCS"] == "NYSE_AMERICAN"
    assert ex["AAA"] == "NYSE_ARCA"
    assert ex["AAAA"] == "CBOE_BZX"
    assert ex["ZZZTX"] == "TXSE"
    # names are stripped (Alcoa's row has a trailing space)
    assert oth.loc[oth["source_symbol"] == "AA", "name"].item() == "Alcoa Corporation Common Stock"


def test_yahoo_symbol_mapping():
    assert us_loader.to_yahoo_symbol("BRK.B") == "BRK-B"
    assert us_loader.to_yahoo_symbol("AAPL") == "AAPL"


def test_filter_keeps_common_stocks_only():
    cfg = load_config("us")
    raw = __import__("pandas").concat(
        [us_loader.parse_nasdaq_listed(NASDAQ_TXT), us_loader.parse_other_listed(OTHER_TXT)],
        ignore_index=True,
    )
    uni = us_loader.filter_universe(raw, cfg.universe)
    assert uni["symbol"].tolist() == EXPECTED


@pytest.mark.parametrize(
    "dropped, why",
    [
        ("AAAP", "ETF"),
        ("AACIW", "warrant"),
        ("AACIU", "units"),
        ("AACPR", "rights"),
        ("ACGLN", "preferred depositary shares"),
        ("ZAZZT", "test issue"),
        ("ADAMG", "notes"),
        ("AAC-U", "unit (symbol suffix)"),
        ("AAC-W", "warrant (symbol suffix)"),
        ("ABR$D", "preferred ($)"),
        ("AHL$E", "preference shares"),
        ("AMUB", "NYSE Arca ETN"),
        ("AAAA", "Cboe ETF"),
        ("AIIA-R", "rights"),
        ("ABXL", "notes"),
        ("ZZZTX", "TXSE test"),
        ("BLUWU", "unit with trailing period"),
        ("CHSCN", "preferred class B"),
    ],
)
def test_specific_exclusions(dropped, why):
    assert dropped not in EXPECTED, why


def test_load_universe_caches_and_reuses(tmp_path, monkeypatch):
    cfg = load_config("us")
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"cache_dir": str(tmp_path)})})
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return NASDAQ_TXT if "nasdaqlisted" in url else OTHER_TXT

    t0 = datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc)
    uni = us_loader.load_universe(cfg, now=t0, fetch_text=fake_fetch)
    assert uni["symbol"].tolist() == EXPECTED
    assert len(calls) == 2
    # within max age: served from universe.csv, no new downloads
    uni2 = us_loader.load_universe(cfg, now=t0 + timedelta(hours=1), fetch_text=fake_fetch)
    assert uni2["symbol"].tolist() == EXPECTED
    assert len(calls) == 2
    # older than max age: downloads again
    us_loader.load_universe(cfg, now=t0 + timedelta(hours=30), fetch_text=fake_fetch)
    assert len(calls) == 4


def test_load_universe_falls_back_to_cache_when_download_fails(tmp_path):
    cfg = load_config("us")
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"cache_dir": str(tmp_path)})})
    t0 = datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc)
    us_loader.load_universe(cfg, now=t0, fetch_text=lambda u: NASDAQ_TXT if "nasdaqlisted" in u else OTHER_TXT)

    def broken(url):
        raise ConnectionError("down")

    uni = us_loader.load_universe(cfg, now=t0 + timedelta(days=3), fetch_text=broken)
    assert uni["symbol"].tolist() == EXPECTED
