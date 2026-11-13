# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Qullamaggie-style momentum scanner for US (and, from Phase 4, India) equities. `SPEC.md` is the source of truth for rules, thresholds and the build phases; work through the phases one at a time, using plan mode before each.

## Ground rules (from SPEC.md)

- Read-only market data. No broker integration, no order placement, no credentials in the repo. Secrets go in `.env` (gitignored) and GitHub Actions secrets.
- Every threshold is a tunable default and lives in `config/<market>.yaml`, never in code. In the YAML, `(SPEC)` marks the user's numbers and `(assumption)` marks Claude's choices. Keep labelling new values that way and tell the user which numbers are assumptions.
- Explain every formula in a code comment. Write unit tests for every indicator.
- Commit after each working step. Never rewrite working code without saying why.
- Output is information, not investment advice.

## Commands

Python 3.12 managed by uv (`uv sync` to install).

```bash
uv run pytest                                    # offline test suite (network tests deselected)
uv run pytest tests/test_scans.py::test_fixed_mode_leaders   # single test
uv run pytest -m network                         # live checks: yfinance split adjustment, Nasdaq Trader files
uv run python -m scanner.run --market us         # update cache; print breakouts, watchlist, leader counts
uv run python -m scanner.run --market us --scan leaders                  # full leaders table
uv run python -m scanner.run --market us --no-update --as-of 2026-10-30   # cached data, past session
uv run python -m scanner.run --market us --symbols NVDA,PLTR             # quick dev run
uv run python -m scanner.run --market us --alert --dry-run               # print Telegram alerts, save nothing
uv run --env-file .env python -m scanner.run --market us --alert         # send alerts, save state/us.json
uv run --env-file .env python -m scanner.notify --chat-id                # find TELEGRAM_CHAT_ID
uv run --env-file .env python -m scanner.notify --test                   # send a test message
```

Other CLI flags: `--scan all|leaders|watchlist|breakouts`, `--window 1m|3m|6m`, `--mode fixed|rank`, `--refresh` (re-run today's incremental update), `--limit N`, `--csv PATH`, `-v`.

## Architecture

Data flows: universe loader → `PriceProvider` → parquet cache (long format) → `Panels` (wide) → indicators → scans → CLI.

- **Two price shapes** (`scanner/data/base.py`). *Long* rows (`symbol, date, open, high, low, close, volume`) are what providers return and the cache stores. *Panels* are wide date × symbol DataFrames, one per field; indicators and scans use these so every symbol is computed at once. `Panels.from_long` builds the trading calendar from dates where ≥50% of the median symbol count has a bar, so a stray holiday bar from one ticker can't shift every "n bars ago". Missing bars stay NaN on purpose (no forward-fill).
- **Scans are market-agnostic** (`scanner/scans.py`). Every rule is computed as a date × symbol frame (`liquidity_frames` → `leader_frames` → `setup_frames`); results are read off the *last row*. `run_scans` builds the frames once on a short tail of the panels (`_rows_needed`) and returns leaders, watchlist and breakouts together, and the CLI uses only `run_scans`. Rolling means over different tail lengths differ by ~1e-15, so compare tables with a tolerance in tests. `panels.as_of(date)` is how point-in-time / no-lookahead evaluation works, and the Phase 5 backtest should reuse it. Use `traded_value` / `min_avg_traded_value` (dollar volume in the US, turnover in India), not market-specific names.
- **Watchlist / breakout rules** (`_watch_state`): a name is on the watchlist at day *t* if its setup qualified within the last `watchlist_memory_sessions` (5), it hasn't closed above that day's trigger since, and the *structural* rules (Stage 1 liquidity on *t* itself via `setup.require_liquid_today`, leader in the last 20 sessions, close > SMA50, SMA10 > SMA20, within 15% of the 20-day high) still hold at *t*. The noisy rules (higher lows, tight range, volume dry-up) may lapse within the window. **A breakout on *t* requires being on the watchlist at *t−1*.** The trigger comes from the setup day, the watchlist stop is the latest low, and the breakout stop is the breakout day's low. Average volume always means the 20 sessions *before* the bar (`indicators.prior_mean`). `tests/synth.py` holds the hand-checked "textbook" setup used by the setup and breakout tests.
- **Indicators** (`scanner/indicators.py`) accept a Series or a panel and require full windows (`min_periods = n`).
- **Market adapters** (`scanner/data/us_loader.py`) supply `load_universe(cfg)` and `make_provider(cfg)`. `scanner/run.py` maps `--market` to an adapter via `MARKETS`; India plugs in there.
- **yfinance** lives only in `scanner/data/yfinance_provider.py`, behind the `PriceProvider` protocol, so a paid provider can replace it. It calls `yf.download(auto_adjust=False)` and uses OHLCV as returned: split-adjusted, NOT dividend-adjusted (deliberate choice; `Adj Close` is ignored). A batch where every ticker fails is treated as rate limiting and retried with backoff; partial failures get one leftover pass.
- **Cache** (`scanner/data/cache.py`, files under `data/cache/<market>/`, gitignored): `prices.parquet`, `universe.csv`, `meta.json`. Updates re-download a 150-bar overlap, which covers the 126-bar 6M window, so a split Yahoo records late is still caught. If an overlapping close differs by >0.5% (a split re-adjusts all of Yahoo's history), that symbol's full history is re-downloaded. `meta.json["checked"]` records per symbol the last cutoff it was updated against; per-symbol tracking keeps a `--symbols` run from making a full run skip other names. Writes are atomic (temp file + `os.replace`) and the first download checkpoints every 10 batches.
- **One-day-move flag**: leaders whose window contains a one-day move of at least `flag_day_move_multiple` (3×) are marked ⚠. These are real squeezes or reverse splits Yahoo hasn't adjusted yet; the first full run found CHOW (~10× overnight) and VCIG (~20×), with no split recorded on Yahoo.
- **Partial bars**: `last_complete_session_date` drops today's bar until market close + `close_buffer_minutes` in the exchange time zone from config.
- **Config** (`scanner/config.py`): strict pydantic models (`extra="forbid"`), so a misspelled YAML key is an error, not a silent default.
- **US universe filter**: Nasdaq Trader `nasdaqlisted`/`otherlisted`, keeping NASDAQ/NYSE/NYSE American and dropping ETFs, test issues and the regex patterns in `us.yaml`. The name regexes were tuned against the real files; `tests/fixtures/*_sample.txt` pins real false-positive cases (e.g. "Preferred Bank", ADRs with "right to receive", MLP "Common Units"). Re-check against the live files when changing patterns.
- **Alerts** (`scanner/state.py`, `scanner/notify.py`): `state/<market>.json` (committed, one file per market) records the last *sent* run. "New" means on today's watchlist but not on the last sent one, plus today's "ok" breakouts. The same market date skips silently (holidays, reruns), an older date is refused, and fewer than `alerts.min_active_ratio` × last run's active symbols triggers a warning with no state change. The first run records a baseline. State is saved only after Telegram delivery succeeds, never on `--dry-run`. A real `--alert` run refuses `--symbols/--limit/--as-of`. The bot token is part of every Bot API URL, so `TelegramNotifier.scrub` removes it from all error text; keep it that way.
- **GitHub Actions** (`.github/workflows/scan-us.yml`): cron 22:00 UTC Mon–Fri. The price cache lives in the Actions cache (new key per run, restore newest); `state/us.json` is committed back by github-actions[bot]. Manual dispatch defaults to dry run. Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. The repo is public: never commit `.env`, tokens or cache files.

