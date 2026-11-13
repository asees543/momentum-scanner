# Momentum Scanner (Qullamaggie-style): US + India

Put this file in the project root. Tell Claude Code: "Read SPEC.md, then work through the phases one at a time. Use plan mode before each phase."

## Goal
A scanner that runs automatically after each market close, finds stocks matching Qullamaggie-style momentum-breakout rules, and sends me a message only about NEW names. One shared engine, two market adapters (US, India). All thresholds live in config files, not in code.

## Ground rules for Claude Code
- Read-only market data. No broker integration, no order placement, no credentials in the repo. Secrets go in `.env` (gitignored) and GitHub Actions secrets.
- Rules differ between public write-ups of this method. Treat every threshold below as a tunable default, and tell me which numbers are my assumptions.
- Explain every formula in a code comment. Write unit tests for every indicator.
- Commit after each working step. Never rewrite working code without telling me why.
- Output is information, not investment advice. Paper trade before using real money.

## Architecture
```
momentum-scanner/
  CLAUDE.md                # project rules + commands (create via /init, then add the ground rules above)
  SPEC.md                  # this file
  config/us.yaml
  config/india.yaml
  scanner/
    data/us_loader.py      # universe + daily OHLCV for US
    data/india_loader.py   # universe + daily OHLCV for NSE
    data/cache.py          # parquet cache, incremental updates
    indicators.py          # returns, ADR, SMAs, dollar volume, RS rank
    scans.py               # leaders, setup, breakout, episodic pivot
    state.py               # saves last run's results, computes what is NEW
    notify.py              # Telegram (primary), email (optional)
    run.py                 # CLI: python -m scanner.run --market us
  tests/
  .github/workflows/scan-us.yml
  .github/workflows/scan-india.yml
```
Every loader returns the same shape: per symbol, a DataFrame with `date, open, high, low, close, volume` (prices adjusted for splits/bonuses). Scans never know which market they are running on.

## Default rules (US)

### Stage 1: universe and liquidity
- Listed stocks only (NYSE, Nasdaq, NYSE American). Exclude OTC, ETFs, warrants, preferreds.
- Price >= $1 (configurable; many people use a higher floor).
- 20-day average dollar volume (close x volume) >= $3,000,000.
- ADR% >= 5. Define ADR% = 100 x (mean over 20 days of high/low - 1).

### Stage 2: momentum leaders (run all three scans)
- 1 month: close >= 1.25 x close 22 trading days ago
- 3 month: close >= 1.50 x close 67 trading days ago
- 6 month: close >= 2.50 x close 126 trading days ago
- Alternative mode (config switch): instead of fixed percentages, keep the top 7% of the universe by return over each window (relative strength rank).
- Target: roughly 25 to 75 results per scan. If far outside that, tune thresholds, not the logic.

### Stage 3: setup = orderly consolidation (builds the WATCHLIST)
A leader qualifies when all hold:
- Close > 50-day SMA, and 10-day SMA > 20-day SMA
- Close within 15% of the 20-day high
- Higher lows: the lowest low of the last 5 days > lowest low of the 5 days before that
- Tight range: average daily range of the last 3 to 5 days < ADR
- Volume drying up: today's volume <= 70% of the 20-day average
For each watchlist name store: `trigger_price` = highest high of the consolidation window.

### Stage 4: breakout trigger (BREAKOUT alert)
On a watchlist name, alert when today:
- Close > trigger_price, AND
- Volume >= 1.5 x 20-day average volume
- Stop reference = low of the breakout day. Reject the alert if (close - stop) / close > ADR% (stop wider than one average daily range means the risk is too large).
- Show risk % = (close - stop) / close.

### Stage 5 (later): episodic pivot
Gap up >= 10% at the open versus the prior close, on large volume, passing the liquidity filter. Fundamentals/earnings data is optional, add later.

Not implementing parabolic shorts.

## India adaptations (config/india.yaml)
- Currency: replace dollar volume with turnover in rupees. Start at 20-day average turnover >= Rs 5 crore, then tune.
- ADR threshold: start at 4% (Indian large and mid caps often have lower ADR). Tune so each scan returns 25 to 75 names.
- Price floor: Rs 100 (my original screen).
- Universe: NSE equity list, series EQ only. Exclude other series.
- Exclude stocks stuck at circuit limits for consecutive days (illiquid, price moves are not tradable).
- Optional: exclude names on NSE surveillance lists (ASM/GSM).
- Prefer RELATIVE strength rank mode over fixed percentages. Indian small caps can double on thin liquidity.
- IMPORTANT: prices must be adjusted for splits and bonus issues. Unadjusted prices create fake "gainers" and "losers" on corporate-action days.

## Data sources (verify these exist and work before relying on them)
- US universe: Nasdaq Trader symbol directory files (nasdaqlisted / otherlisted), filter out ETFs and test issues.
- US prices: start with `yfinance` batch downloads, adjusted prices, cached to parquet, incremental daily updates. It is free but unofficial and can be rate-limited or break. Design `us_loader.py` so a paid provider (Polygon, Tiingo, Alpaca) can be swapped in.
- India universe: NSE's equity list CSV.
- India prices: start with `yfinance` using `.NS` tickers because they come adjusted. Later option: NSE official bhavcopy archives (free, end of day). NSE moved to the UDiFF file format, so check that whatever library you pick (aynse, jugaad-data, nse-archives) handles the current format. Bhavcopy is UNADJUSTED, so you must apply corporate actions yourself.

## Alerts and state
- `state.json` stores the previous run's WATCHLIST and BREAKOUT symbols per market.
- Send only what is new: "New on watchlist", "Breakout today", plus a one-line summary (counts, market date).
- Each alert line: symbol, 1M/3M/6M return, ADR%, avg $ or Rs volume, trigger price, stop, risk %, chart link.
- Telegram: create a bot with @BotFather, store the token and chat id as secrets.
- If the market was closed (no new bar), skip silently.

## Scheduling (GitHub Actions, free tier)
- Cron runs in UTC. US: run after the close, for example 22:00 UTC Mon-Fri (covers both daylight-saving states).
- India: NSE closes 15:30 IST = 10:00 UTC. Run around 12:30 UTC Mon-Fri and retry once if data is not yet available.
- Commit `state.json` and the cache back to the repo, or use the Actions cache.

## Backtest (do this before trusting any signal)
- Replay the scans over history using only data available on each date (no lookahead).
- Record forward returns at 5, 10, and 20 days versus the benchmark (S&P 500 / Nifty 50).
- Simple trade simulation: stop at the breakout day low, sell a third to half after 3-5 days, trail the rest below the 10 or 20-day SMA.
- Be honest about biases: survivorship (delisted names missing from free data), costs and slippage, and the fact that this method's reported results do not automatically transfer to other years or to India.

## Build phases (paste one at a time into Claude Code, in plan mode)

**Phase 1: skeleton + US leaders.** "Read SPEC.md. Plan Phase 1: project structure, config loading, US data loader with caching, indicators (returns, ADR, SMAs, dollar volume) with unit tests, and the momentum-leaders scan. A CLI command should print a table of results. Ask me questions first."

**Phase 2: setup + breakout.** "Plan Phase 2: implement the consolidation setup (watchlist) and breakout trigger from SPEC.md, with tests on synthetic price series. Show me a table of watchlist names with trigger price and stop."

**Phase 3: alerts + automation.** "Plan Phase 3: state diffing, Telegram notifications, and a GitHub Actions workflow that runs after the US close and only sends new names. Include a dry-run mode that prints instead of sending."

**Phase 4: India.** "Plan Phase 4: India adapter and india.yaml per SPEC.md, using yfinance .NS first. Flag any data-quality issues you find (splits, bonuses, missing days, circuit-limit stocks)."

**Phase 5: backtest + dashboard.** "Plan Phase 5: a point-in-time backtest of the scans, then a simple static dashboard page of today's watchlist and breakouts. Report biases clearly."
