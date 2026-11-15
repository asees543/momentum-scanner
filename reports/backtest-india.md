# Backtest · INDIA

Generated 2026-11-15 by `python -m scanner.backtest --market india`. **Information only, not investment advice. Paper trade before using real money.**

## Setup

- Signals from **2022-07-25 to 2026-11-13** (1,063 sessions), on today's universe of 2,391 stocks; leaders in **rank** mode; all thresholds from `config/india.yaml`.
- Costs: 0.15% + 0.1% slippage per buy or sell. Benchmark: ^NSEI.
- Point-in-time check: the live scanner re-run on data cut off at 13 random days gave **exactly the same breakouts** as the backtest.

Entry models:
- **A · buy-stop at the trigger (breakout-day entry)**: every day a name from the previous evening's watchlist traded up to its trigger; bought at the trigger (or the open if it gapped above). Includes triggers that **failed** (closed back below the trigger or without 1.5× volume), because a buy-stop fills on those too. Stop = that day's low, from the next day.
- **C · buy-stop at the trigger, failed breakouts sold at that day's close**: Model A's entries, but a fill that hasn't confirmed by the close (closed at or below the trigger, or under 1.5× volume) is sold at that close; confirmed ones continue with the normal exits. Slightly optimistic: in practice you decide minutes before the close.
- **B · next-day open after the alert**: every confirmed breakout (ok or too wide), bought at the next open; skipped if that open was already at or below the stop.
- Exits (SPEC): stop at the breakout-day low; sell part after N days; trail the rest below an SMA. Headline exit, fixed before running: **sell 1/3 after 3d, trail 10-SMA**.

## Read this first: biases

- **Survivorship.** Only stocks listed *today* are tested. Stocks delisted since 2022 (many failed momentum names among them) are missing from free data, so results are **better than reality**.
- **Universe look-ahead.** Today's lists also decide which tickers count as common stock in the past.
- **Data quality.** Unadjusted splits/bonuses create fake moves; results are also shown without ⚠-flagged names.
- **† Hindsight groups.** Models A and C split into ok / too wide / failed using the breakout day's close and volume, which aren't known when a buy-stop fills. Those rows explain where A's results come from but can't be traded; **only “A · all” and “C · all” are tradeable strategies**. Model B's groups are known before its entry.
- **Position size and R use only what's known at entry.** For the buy-stop models (A, C) that is the previous day's low (the watchlist stop); the breakout day's low is only known after the fill. (A first version sized on the day's low and showed C at +53% a year; that was look-ahead.) Exits still use the breakout day's low.
- **R multiples** use a minimum stop distance of 1%: a few near-zero stops otherwise dominate the average. Median R is shown next to it.
- **Daily bars.** They can't show whether the day's low came before or after a buy-stop filled; Model A's stop only starts the next day, which flatters A slightly. Gaps through stops are filled at the open.
- **Costs and slippage are estimates**; real fills on breakouts are often worse.
- **One regime.** 2022–2026 only. **8 exit combinations** were tested, so the best one looks good partly by luck; judge by the pre-chosen headline. Results don't transfer automatically to other years.

## Signals

|  | Count |
|---|---|
| Trigger touches (Model A and C fills) | 5,853 |
| …of which ok breakouts | 379 |
| …of which too wide (stop > 1 ADR) | 725 |
| …of which failed (no confirmed close) | 4,749 |
| Confirmed breakouts (Model B candidates) | 1,104 |
| …entered at the next open | 1,100 |
| …skipped: opened at or below the stop | 4 |

## Forward returns (no costs)

Mean return from the entry price to the close k sessions later, and the benchmark over the same sessions. *Beat* = share of trades ahead of the benchmark.

| Model · group | Trades | 5d | 5d bench | 5d beat | 10d | 10d bench | 10d beat | 20d | 20d bench | 20d beat |
|---|---|---|---|---|---|---|---|---|---|---|
| A · all | 5,853 | -0.2% | +0.1% | 42% | -0.0% | +0.2% | 43% | +0.8% | +0.6% | 44% |
| A · ok † | 379 | +1.4% | +0.1% | 55% | +1.7% | +0.2% | 52% | +2.4% | +0.7% | 46% |
| A · too wide † | 725 | +4.5% | +0.3% | 65% | +5.3% | +0.4% | 63% | +6.1% | +0.8% | 59% |
| A · failed † | 4,749 | -1.1% | +0.1% | 38% | -1.0% | +0.2% | 39% | -0.1% | +0.6% | 42% |
| B · all | 1,100 | -0.6% | +0.1% | 41% | +0.2% | +0.2% | 43% | +0.5% | +0.6% | 43% |
| B · ok | 376 | -0.5% | +0.1% | 41% | -0.4% | +0.1% | 42% | +0.1% | +0.6% | 41% |
| B · too wide | 724 | -0.6% | +0.1% | 40% | +0.5% | +0.3% | 44% | +0.7% | +0.6% | 43% |

## Trade simulation: headline exit (sell 1/3 after 3d, trail 10-SMA)

**A · buy-stop at the trigger (breakout-day entry)**

| Group | Trades | Win rate | Avg win | Avg loss | Mean / trade | Mean R | Median R | Profit factor | Median days |
|---|---|---|---|---|---|---|---|---|---|
| A · all | 5,853 | 25% | +8.2% | -3.8% | -0.86% | -0.18 | -0.57 | 0.70 | 3 |
| A · ok † | 379 | 32% | +8.5% | -2.8% | +0.82% | 0.20 | -0.28 | 1.43 | 3 |
| A · too wide † | 725 | 57% | +9.6% | -3.7% | +3.85% | 0.73 | 0.23 | 3.41 | 7 |
| A · failed † | 4,749 | 19% | +7.5% | -3.9% | -1.72% | -0.35 | -0.63 | 0.46 | 2 |

**C · buy-stop at the trigger, failed breakouts sold at that day's close**

| Group | Trades | Win rate | Avg win | Avg loss | Mean / trade | Mean R | Median R | Profit factor | Median days |
|---|---|---|---|---|---|---|---|---|---|
| C · all | 5,853 | 27% | +4.2% | -2.5% | -0.72% | -0.15 | -0.25 | 0.61 | 0 |
| C · ok † | 379 | 32% | +8.5% | -2.8% | +0.82% | 0.20 | -0.28 | 1.43 | 3 |
| C · too wide † | 725 | 57% | +9.6% | -3.7% | +3.85% | 0.73 | 0.23 | 3.41 | 7 |
| C · failed † | 4,749 | 22% | +1.6% | -2.4% | -1.54% | -0.31 | -0.27 | 0.18 | 0 |

**B · next-day open after the alert**

| Group | Trades | Win rate | Avg win | Avg loss | Mean / trade | Mean R | Median R | Profit factor | Median days |
|---|---|---|---|---|---|---|---|---|---|
| B · all | 1,100 | 29% | +8.5% | -5.3% | -1.26% | -0.22 | -0.58 | 0.67 | 5 |
| B · ok | 376 | 24% | +8.4% | -4.2% | -1.10% | -0.28 | -1.07 | 0.65 | 2 |
| B · too wide | 724 | 32% | +8.5% | -6.0% | -1.34% | -0.18 | -0.40 | 0.67 | 6 |

## All 8 exit combinations

Mean return per trade (after costs) and profit factor. The headline is marked ★.

| Exit | A all: mean | PF | C all: mean | PF | B all: mean | PF | B ok: mean | PF |
|---|---|---|---|---|---|---|---|---|
| ★ sell 1/3 after 3d, trail 10-SMA | -0.86% | 0.70 | -0.72% | 0.61 | -1.26% | 0.67 | -1.10% | 0.65 |
| sell 1/3 after 3d, trail 20-SMA | -0.88% | 0.71 | -0.71% | 0.63 | -1.19% | 0.72 | -1.16% | 0.66 |
| sell 1/3 after 5d, trail 10-SMA | -0.92% | 0.70 | -0.73% | 0.61 | -1.32% | 0.67 | -1.07% | 0.68 |
| sell 1/3 after 5d, trail 20-SMA | -0.93% | 0.71 | -0.71% | 0.63 | -1.23% | 0.72 | -1.12% | 0.68 |
| sell 1/2 after 3d, trail 10-SMA | -0.81% | 0.71 | -0.71% | 0.61 | -1.24% | 0.65 | -1.16% | 0.62 |
| sell 1/2 after 3d, trail 20-SMA | -0.82% | 0.71 | -0.71% | 0.62 | -1.20% | 0.69 | -1.20% | 0.62 |
| sell 1/2 after 5d, trail 10-SMA | -0.88% | 0.71 | -0.73% | 0.61 | -1.32% | 0.66 | -1.09% | 0.67 |
| sell 1/2 after 5d, trail 20-SMA | -0.89% | 0.71 | -0.71% | 0.62 | -1.26% | 0.70 | -1.13% | 0.67 |

## By year (headline exit)

| Model · group | Year | Trades | Win rate | Mean / trade | Profit factor |
|---|---|---|---|---|---|
| A · all | 2022 | 357 | 24% | -0.81% | 0.71 |
| A · all | 2023 | 984 | 25% | -0.45% | 0.83 |
| A · all | 2024 | 1,801 | 27% | -0.55% | 0.81 |
| A · all | 2025 | 1,153 | 21% | -1.53% | 0.50 |
| A · all | 2026 | 1,558 | 25% | -1.01% | 0.65 |
| C · all | 2022 | 357 | 27% | -0.40% | 0.76 |
| C · all | 2023 | 984 | 28% | -0.68% | 0.62 |
| C · all | 2024 | 1,801 | 27% | -0.55% | 0.71 |
| C · all | 2025 | 1,153 | 25% | -1.00% | 0.47 |
| C · all | 2026 | 1,558 | 26% | -0.79% | 0.56 |
| B · all | 2022 | 74 | 27% | -0.41% | 0.88 |
| B · all | 2023 | 181 | 27% | -1.56% | 0.59 |
| B · all | 2024 | 403 | 33% | -1.02% | 0.74 |
| B · all | 2025 | 200 | 24% | -2.01% | 0.47 |
| B · all | 2026 | 242 | 31% | -1.06% | 0.69 |
| B · ok | 2022 | 22 | 23% | -1.73% | 0.51 |
| B · ok | 2023 | 51 | 25% | +0.29% | 1.10 |
| B · ok | 2024 | 132 | 27% | -0.77% | 0.75 |
| B · ok | 2025 | 89 | 21% | -1.77% | 0.45 |
| B · ok | 2026 | 82 | 23% | -1.60% | 0.49 |

## Without ⚠-flagged names (headline exit)

Names whose 6-month window had a one-day move beyond the flag threshold (possible unadjusted split/bonus).

| Group | Trades | Win rate | Avg win | Avg loss | Mean / trade | Mean R | Median R | Profit factor | Median days |
|---|---|---|---|---|---|---|---|---|---|
| A · all (n flagged: 16) | 5,837 | 25% | +8.2% | -3.8% | -0.86% | -0.18 | -0.57 | 0.70 | 3 |
| A · ok † (n flagged: 1) | 378 | 32% | +8.5% | -2.8% | +0.83% | 0.20 | -0.28 | 1.44 | 3 |
| A · too wide † (n flagged: 1) | 724 | 57% | +9.6% | -3.7% | +3.85% | 0.73 | 0.23 | 3.41 | 7 |
| A · failed † (n flagged: 14) | 4,735 | 19% | +7.5% | -3.9% | -1.72% | -0.35 | -0.63 | 0.46 | 2 |
| C · all (n flagged: 16) | 5,837 | 27% | +4.2% | -2.5% | -0.71% | -0.15 | -0.25 | 0.61 | 0 |
| C · ok † (n flagged: 1) | 378 | 32% | +8.5% | -2.8% | +0.83% | 0.20 | -0.28 | 1.44 | 3 |
| C · too wide † (n flagged: 1) | 724 | 57% | +9.6% | -3.7% | +3.85% | 0.73 | 0.23 | 3.41 | 7 |
| C · failed † (n flagged: 14) | 4,735 | 22% | +1.6% | -2.4% | -1.54% | -0.31 | -0.27 | 0.18 | 0 |
| B · all (n flagged: 2) | 1,098 | 29% | +8.5% | -5.3% | -1.26% | -0.22 | -0.58 | 0.67 | 5 |
| B · ok (n flagged: 1) | 375 | 25% | +8.4% | -4.2% | -1.09% | -0.28 | -1.07 | 0.65 | 3 |
| B · too wide (n flagged: 1) | 723 | 32% | +8.6% | -6.0% | -1.34% | -0.18 | -0.40 | 0.67 | 6 |

## Equity curve (headline exit, realized)

Risking 0.5% of equity per trade (stop distance at least 1%; no position above 25% of equity; never more than 100% invested), at most 10 open. Models A and C take same-day fills in a fixed random order (the day's volume is only known after the fill); Model B by volume. P&L is counted when a trade closes, so drawdowns while trades are open are not shown. Every curve and the benchmark start on the first signal day. Only tradeable groups are listed.

| Model · group | Trades taken | Skipped (full or no cash) | CAGR | Max drawdown |
|---|---|---|---|---|
| A · all | 1,818 | 4,034 | -31.9% | -82.2% |
| C · all | 3,388 | 2,464 | -40.4% | -90.0% |
| B · all | 813 | 287 | -18.0% | -59.4% |
| B · ok | 357 | 19 | -9.2% | -36.7% |
| ^NSEI buy and hold | – | – | +10.1% | -15.8% |

## Market filter test (^NSEI above its 50-day average)

**Rule, fixed before running:** take a trade only if ^NSEI closed above its 50-day simple moving average on the session before entry, the last close known when the order is placed (A and C: the evening the buy-stop is set from the watchlist; B: the breakout evening). No other averages or filters were tried.

**Pass mark, also fixed before running:** after costs, a positive mean return per trade **and** an equity-curve CAGR above ^NSEI buy and hold over the same period. Headline exit only.

The filter was on for 62% of sessions. Trades below are split by whether the filter allowed them; the *removed* rows are what the filter avoided.

| Group | Trades | Win rate | Avg win | Avg loss | Mean / trade | Mean R | Median R | Profit factor | Median days |
|---|---|---|---|---|---|---|---|---|---|
| A · all · kept | 4,045 | 23% | +8.5% | -3.8% | -0.94% | -0.21 | -0.58 | 0.68 | 3 |
| A · all · removed | 1,808 | 28% | +7.6% | -3.9% | -0.68% | -0.12 | -0.53 | 0.76 | 3 |
| C · all · kept | 4,045 | 26% | +4.2% | -2.5% | -0.72% | -0.16 | -0.25 | 0.61 | 0 |
| C · all · removed | 1,808 | 28% | +4.2% | -2.6% | -0.70% | -0.13 | -0.24 | 0.62 | 0 |
| B · all · kept | 766 | 28% | +8.3% | -5.3% | -1.51% | -0.27 | -0.62 | 0.61 | 5 |
| B · all · removed | 334 | 32% | +8.9% | -5.2% | -0.68% | -0.09 | -0.47 | 0.81 | 5 |
| B · ok · kept | 259 | 23% | +7.4% | -4.2% | -1.54% | -0.41 | -1.08 | 0.52 | 2 |
| B · ok · removed | 117 | 28% | +10.2% | -4.2% | -0.13% | 0.01 | -1.06 | 0.96 | 3 |

Equity curve with the filter (same sizing rules as above):

| Model · group | Trades taken | Mean / trade | CAGR | Max drawdown | Pass mark |
|---|---|---|---|---|---|
| A · all · filtered | 1,383 | -0.94% | -30.1% | -81.5% | **fail** |
| C · all · filtered | 2,449 | -0.72% | -33.2% | -84.0% | **fail** |
| B · all · filtered | 608 | -1.51% | -16.8% | -56.9% | **fail** |
| B · ok · filtered | 250 | -1.54% | -10.5% | -40.5% | **fail** |
| ^NSEI buy and hold | – | – | +10.1% | -15.8% | – |
