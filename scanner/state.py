"""Alert state: what was last SENT for a market, and what is new in today's scan.

One JSON file per market, state/<market>.json, committed to the repo by the GitHub Actions run:

    {
      "market": "us",
      "as_of": "2026-11-12",              # market date of the last run whose alerts were delivered
      "active_count": 5910,               # symbols with a bar that day (data-health baseline)
      "watchlist": {"XERS": {"trigger": 16.88, "first_seen": "2026-11-12"}, ...},
      "breakouts": ["SGMT"],                 # "ok" breakouts (stop within one ADR)
      "wide_breakouts": [],                  # breakouts whose stop is wider than one ADR
      "updated_at": "2026-11-13T22:07:41+00:00"
    }

Rules (see `diff_results`):
* same market date as the state     -> skip silently (market holiday, or a rerun of the same day)
* older market date than the state  -> refuse (never move state backwards)
* fewer than min_active_ratio x the last run's symbols have a bar -> unhealthy: warn, keep old state
* new watchlist names = on today's watchlist, not on the last SENT watchlist. A name that drops off
  and later returns is new again.
* breakouts = today's breakouts, split into "ok" and "too wide" (stop wider than one ADR; SPEC
  rejects those). Each breakout belongs to one market date, so all are new once the date has moved
  on. The "too wide" ones are listed in their own section when alerts.list_wide_breakouts is on.
* no state file yet + baseline_first_run -> record today's watchlist without listing it.
The caller saves `next_state` only after the messages were delivered.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from scanner.config import PROJECT_ROOT, AlertsConfig
from scanner.scans import ScanResults

STATE_DIR = PROJECT_ROOT / "state"

Status = Literal["send", "skip_same_day", "refuse_older", "unhealthy"]


def state_path(market: str, directory: Path = STATE_DIR) -> Path:
    return Path(directory) / f"{market}.json"


@dataclass
class MarketState:
    market: str
    as_of: date
    active_count: int
    watchlist: dict[str, dict] = field(default_factory=dict)
    breakouts: list[str] = field(default_factory=list)
    wide_breakouts: list[str] = field(default_factory=list)
    updated_at: str = ""

    @classmethod
    def load(cls, path: Path) -> MarketState | None:
        if not Path(path).exists():
            return None
        raw = json.loads(Path(path).read_text())
        return cls(
            market=raw["market"],
            as_of=date.fromisoformat(raw["as_of"]),
            active_count=int(raw["active_count"]),
            watchlist=dict(raw.get("watchlist", {})),
            breakouts=list(raw.get("breakouts", [])),
            wide_breakouts=list(raw.get("wide_breakouts", [])),
            updated_at=raw.get("updated_at", ""),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["as_of"] = self.as_of.isoformat()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)


@dataclass
class Diff:
    status: Status
    as_of: date
    reason: str  # human-readable explanation for non-"send" statuses
    first_run: bool = False
    new_watchlist: list[str] = field(default_factory=list)  # in watchlist-table order
    breakouts: list[str] = field(default_factory=list)  # "ok" breakouts, in breakout-table order
    wide_breakouts: list[str] = field(default_factory=list)  # stop wider than one ADR
    rejected_breakouts: int = 0  # = len(wide_breakouts)
    watchlist_size: int = 0
    next_state: MarketState | None = None
    # Filled by the caller: breakouts on sessions missed since the last sent run, oldest first,
    # and how many older missed sessions were not checked (alerts.catch_up_max_sessions).
    catch_up: list = field(default_factory=list)  # [(session Timestamp, BreakoutResult)]
    catch_up_skipped: int = 0


def diff_results(
    prev: MarketState | None,
    results: ScanResults,
    market: str,
    acfg: AlertsConfig,
    now: datetime | None = None,
) -> Diff:
    as_of = results.leaders.as_of.date()
    active = results.leaders.active_count
    if prev is not None:
        if as_of == prev.as_of:
            return Diff("skip_same_day", as_of, f"no new session since {prev.as_of} (holiday or rerun)")
        if as_of < prev.as_of:
            return Diff("refuse_older", as_of, f"scan date {as_of} is older than the last sent run {prev.as_of}")
        if active < acfg.min_active_ratio * prev.active_count:
            return Diff(
                "unhealthy",
                as_of,
                f"only {active:,} symbols have a bar on {as_of}, vs {prev.active_count:,} on {prev.as_of} "
                f"(below {acfg.min_active_ratio:.0%}); data download probably incomplete",
            )

    wl = results.watchlist.table
    bo = results.breakouts.table
    ok_breakouts = [s for s in bo.index if bo.loc[s, "status"] == "ok"] if len(bo) else []
    wide_breakouts = [s for s in bo.index if bo.loc[s, "status"] != "ok"] if len(bo) else []
    first_run = prev is None
    prev_wl = prev.watchlist if prev else {}
    if first_run and acfg.baseline_first_run:
        new_watchlist: list[str] = []
    else:
        new_watchlist = [s for s in wl.index if s not in prev_wl]

    next_state = MarketState(
        market=market,
        as_of=as_of,
        active_count=int(active),
        watchlist={
            s: {
                "trigger": round(float(wl.loc[s, "trigger"]), 4),
                "first_seen": prev_wl.get(s, {}).get("first_seen", as_of.isoformat()),
            }
            for s in wl.index
        },
        breakouts=ok_breakouts,
        wide_breakouts=wide_breakouts,
        updated_at=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    )
    return Diff(
        status="send",
        as_of=as_of,
        reason="",
        first_run=first_run,
        new_watchlist=new_watchlist,
        breakouts=ok_breakouts,
        wide_breakouts=wide_breakouts,
        rejected_breakouts=len(wide_breakouts),
        watchlist_size=len(wl),
        next_state=next_state,
    )
