# Backtest · US

Generated 2026-11-14 by `python -m scanner.backtest --market us`. **Information only, not investment advice. Paper trade before using real money.**

## Setup

- Signals from **2022-07-25 to 2026-11-13** (1,083 sessions), on today's universe of 5,920 stocks; leaders in **fixed** mode; all thresholds from `config/us.yaml`.
- Costs: 0.05% + 0.1% slippage per buy or sell. Benchmark: SPY.
- Point-in-time check: the live scanner re-run on data cut off at 13 random days gave **exactly the same breakouts** as the backtest.

Entry models:
- **A · buy-stop at the trigger (breakout-day entry)**: every day a name from the previous evening's watchlist traded up to its trigger; bought at the trigger (or the open if it gapped above). Includes triggers that **failed** (closed back below the trigger or without 1.5× volume), because a buy-stop fills on those too. Stop = that day's low, from the next day.
- **B · next-day open after the alert**: every confirmed breakout (ok or too wide), bought at the next open; skipped if that open was already at or below the stop.
- Exits (SPEC): stop at the breakout-day low; sell part after N days; trail the rest below an SMA. Headline exit, fixed before running: **sell 1/3 after 3d, trail 10-SMA**.

## Read this first: biases

- **Survivorship.** Only stocks listed *today* are tested. Stocks delisted since 2022 (many failed momentum names among them) are missing from free data, so results are **better than reality**.
- **Universe look-ahead.** Today's lists also decide which tickers count as common stock in the past.
- **Data quality.** Unadjusted splits/bonuses create fake moves; results are also shown without ⚠-flagged names.
- **† Hindsight groups.** Model A's split into ok / too wide / failed uses the breakout day's close and volume, which aren't known when a buy-stop fills. Those rows explain where A's results come from but can't be traded; **only “A · all” is a tradeable strategy**. Model B's groups are known before its entry.
- **R multiples** use a minimum stop distance of 1%: a few near-zero stops otherwise dominate the average. Median R is shown next to it.
- **Daily bars.** They can't show whether the day's low came before or after a buy-stop filled; Model A's stop only starts the next day, which flatters A slightly. Gaps through stops are filled at the open.
- **Costs and slippage are estimates**; real fills on breakouts are often worse.
- **One regime.** 2022–2026 only. **8 exit combinations** were tested, so the best one looks good partly by luck; judge by the pre-chosen headline. Results don't transfer automatically to other years.

## Signals

|  | Count |
|---|---|
| Trigger touches (Model A fills) | 11,362 |
| …of which ok breakouts | 449 |
| …of which too wide (stop > 1 ADR) | 731 |
| …of which failed (no confirmed close) | 10,182 |
| Confirmed breakouts (Model B candidates) | 1,180 |
| …entered at the next open | 1,162 |
| …skipped: opened at or below the stop | 18 |

## Forward returns (no costs)

Mean return from the entry price to the close k sessions later, and the benchmark over the same sessions. *Beat* = share of trades ahead of the benchmark.

| Model · group | Trades | 5d | 5d bench | 5d beat | 10d | 10d bench | 10d beat | 20d | 20d bench | 20d beat |
|---|---|---|---|---|---|---|---|---|---|---|
| A · all | 11,362 | +0.0% | +0.4% | 44% | -0.1% | +0.7% | 43% | -0.2% | +1.2% | 43% |
| A · ok † | 449 | +2.7% | +0.5% | 56% | +2.5% | +0.7% | 51% | +2.1% | +0.8% | 46% |
| A · too wide † | 731 | +7.8% | +0.5% | 68% | +7.3% | +0.7% | 61% | +6.6% | +1.2% | 55% |
| A · failed † | 10,182 | -0.6% | +0.3% | 42% | -0.8% | +0.7% | 42% | -0.7% | +1.2% | 42% |
| B · all | 1,162 | +0.0% | +0.3% | 45% | -0.4% | +0.5% | 44% | -1.4% | +0.9% | 40% |
| B · ok | 434 | +0.4% | +0.4% | 46% | +0.1% | +0.5% | 46% | -0.7% | +0.7% | 40% |
| B · too wide | 728 | -0.2% | +0.3% | 44% | -0.6% | +0.5% | 43% | -1.9% | +0.9% | 40% |

## Trade simulation: headline exit (sell 1/3 after 3d, trail 10-SMA)

**A · buy-stop at the trigger (breakout-day entry)**

| Group | Trades | Win rate | Avg win | Avg loss | Mean / trade | Mean R | Median R | Profit factor | Median days |
|---|---|---|---|---|---|---|---|---|---|
| A · all | 11,362 | 26% | +12.0% | -4.9% | -0.56% | 0.14 | -1.04 | 0.85 | 2 |
| A · ok † | 449 | 35% | +15.3% | -4.1% | +2.57% | 1.17 | -1.03 | 1.95 | 3 |
| A · too wide † | 731 | 60% | +14.7% | -5.0% | +6.85% | 2.55 | 0.49 | 4.42 | 7 |
| A · failed † | 10,182 | 23% | +11.2% | -4.9% | -1.23% | -0.08 | -1.04 | 0.68 | 2 |

**B · next-day open after the alert**

| Group | Trades | Win rate | Avg win | Avg loss | Mean / trade | Mean R | Median R | Profit factor | Median days |
|---|---|---|---|---|---|---|---|---|---|
| B · all | 1,162 | 32% | +13.6% | -7.5% | -0.75% | -0.08 | -0.57 | 0.85 | 5 |
| B · ok | 434 | 27% | +15.4% | -5.7% | +0.09% | -0.10 | -1.03 | 1.02 | 2 |
| B · too wide | 728 | 35% | +12.8% | -8.7% | -1.25% | -0.06 | -0.37 | 0.78 | 6 |

## All 8 exit combinations

Mean return per trade (after costs) and profit factor. The headline is marked ★.

| Exit | A all: mean | PF | B all: mean | PF | B ok: mean | PF |
|---|---|---|---|---|---|---|
| ★ sell 1/3 after 3d, trail 10-SMA | -0.56% | 0.85 | -0.75% | 0.85 | +0.09% | 1.02 |
| sell 1/3 after 3d, trail 20-SMA | -0.54% | 0.86 | -1.05% | 0.82 | -0.42% | 0.91 |
| sell 1/3 after 5d, trail 10-SMA | -0.57% | 0.85 | -0.63% | 0.88 | +0.37% | 1.09 |
| sell 1/3 after 5d, trail 20-SMA | -0.56% | 0.86 | -0.91% | 0.85 | -0.16% | 0.96 |
| sell 1/2 after 3d, trail 10-SMA | -0.51% | 0.86 | -0.70% | 0.86 | -0.10% | 0.98 |
| sell 1/2 after 3d, trail 20-SMA | -0.49% | 0.86 | -0.93% | 0.82 | -0.48% | 0.89 |
| sell 1/2 after 5d, trail 10-SMA | -0.53% | 0.86 | -0.52% | 0.90 | +0.30% | 1.07 |
| sell 1/2 after 5d, trail 20-SMA | -0.52% | 0.87 | -0.73% | 0.87 | -0.09% | 0.98 |

## By year (headline exit)

| Model · group | Year | Trades | Win rate | Mean / trade | Profit factor |
|---|---|---|---|---|---|
| A · all | 2022 | 1,078 | 23% | -1.11% | 0.69 |
| A · all | 2023 | 1,870 | 25% | -0.75% | 0.78 |
| A · all | 2024 | 2,050 | 24% | -1.18% | 0.71 |
| A · all | 2025 | 3,274 | 28% | +0.17% | 1.05 |
| A · all | 2026 | 3,090 | 26% | -0.62% | 0.83 |
| B · all | 2022 | 111 | 32% | -1.30% | 0.72 |
| B · all | 2023 | 164 | 29% | -2.12% | 0.59 |
| B · all | 2024 | 217 | 28% | -2.47% | 0.57 |
| B · all | 2025 | 364 | 33% | +0.29% | 1.06 |
| B · all | 2026 | 306 | 35% | +0.17% | 1.04 |
| B · ok | 2022 | 39 | 31% | +2.20% | 1.84 |
| B · ok | 2023 | 70 | 27% | -1.34% | 0.64 |
| B · ok | 2024 | 76 | 22% | -2.27% | 0.52 |
| B · ok | 2025 | 145 | 28% | +0.01% | 1.00 |
| B · ok | 2026 | 104 | 29% | +2.09% | 1.53 |

## Without ⚠-flagged names (headline exit)

Names whose 6-month window had a one-day move beyond the flag threshold (possible unadjusted split/bonus).

| Group | Trades | Win rate | Avg win | Avg loss | Mean / trade | Mean R | Median R | Profit factor | Median days |
|---|---|---|---|---|---|---|---|---|---|
| A · all (n flagged: 199) | 11,163 | 26% | +11.7% | -4.8% | -0.59% | 0.12 | -1.04 | 0.84 | 2 |
| A · ok † (n flagged: 6) | 443 | 35% | +14.6% | -4.0% | +2.40% | 1.15 | -1.03 | 1.91 | 3 |
| A · too wide † (n flagged: 6) | 725 | 60% | +14.5% | -5.0% | +6.71% | 2.51 | 0.49 | 4.34 | 7 |
| A · failed † (n flagged: 187) | 9,995 | 23% | +11.0% | -4.9% | -1.25% | -0.10 | -1.05 | 0.67 | 2 |
| B · all (n flagged: 12) | 1,150 | 32% | +13.0% | -7.4% | -0.89% | -0.09 | -0.57 | 0.82 | 5 |
| B · ok (n flagged: 6) | 428 | 28% | +14.4% | -5.7% | -0.16% | -0.11 | -1.03 | 0.96 | 2 |
| B · too wide (n flagged: 6) | 722 | 35% | +12.3% | -8.6% | -1.32% | -0.07 | -0.37 | 0.76 | 6 |

## Equity curve (headline exit, realized)

Risking 0.5% of equity per trade (stop distance at least 1%; no position above 25% of equity; never more than 100% invested), at most 10 open. Model A takes same-day fills in a fixed random order (the day's volume is only known after the fill); Model B by volume. P&L is counted when a trade closes, so drawdowns while trades are open are not shown. Only tradeable groups are listed.

| Model · group | Trades taken | Skipped (full or no cash) | CAGR | Max drawdown |
|---|---|---|---|---|
| A · all | 1,944 | 9,362 | -9.8% | -59.7% |
| B · all | 848 | 314 | -5.7% | -38.9% |
| B · ok | 424 | 10 | -4.4% | -32.2% |
| SPY buy and hold | – | – | +16.2% | -19.0% |
