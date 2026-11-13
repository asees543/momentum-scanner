"""Live checks against real data sources. Deselected by default; run with:  uv run pytest -m network"""

from datetime import date

import pytest

from scanner.data.yfinance_provider import YFinanceProvider

pytestmark = pytest.mark.network


def test_yfinance_prices_are_split_adjusted_across_nvda_split():
    # NVDA split 10-for-1 effective 2024-06-10. Unadjusted data would show a ~-90% day.
    p = YFinanceProvider(pause_seconds=0)
    batches = list(p.fetch_batches(["NVDA"], date(2024, 6, 1), date(2024, 6, 20)))
    df = batches[0].prices
    assert len(df) >= 10
    daily = df["close"].pct_change().dropna()
    assert daily.min() > -0.2
    # Volume is split-adjusted too: pre-split days are in the hundreds of millions of shares.
    assert df.loc[df["date"] == "2024-06-07", "volume"].item() > 2e8


def test_nasdaq_trader_files_reachable():
    from scanner.data.us_loader import NASDAQ_LISTED_URL, OTHER_LISTED_URL, _http_get_text

    for url in (NASDAQ_LISTED_URL, OTHER_LISTED_URL):
        text = _http_get_text(url)
        assert text.splitlines()[-1].startswith("File Creation Time")
