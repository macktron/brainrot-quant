"""Structured run history for P&L tracking and dashboard consumption.

Appends one JSON line per live run to ``history/runs_live.jsonl`` and one line
per paper run to ``history/runs_paper.jsonl``.

Older mock/scratch data may exist in ``history/runs.jsonl``, but it is not
read or written by this module.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HISTORY_DIR = Path(__file__).resolve().parent.parent / "history"
HISTORY_DIR.mkdir(exist_ok=True)
LIVE_FILE = HISTORY_DIR / "runs_live.jsonl"
PAPER_FILE = HISTORY_DIR / "runs_paper.jsonl"


def record_trade(
    *,
    side: str,
    follower_question: str,
    follower_condition_id: str,
    follower_slug: str,
    token_id: str,
    leader_question: str,
    leader_outcome: str,
    confidence: float,
    rationale: str,
    bet_size_usdc: float,
    order_id: str | None,
    executed: bool,
    error: str | None,
    category: str = "",
) -> dict[str, Any]:
    """Build a compact trade record dict."""
    return {
        "side": side,
        "follower": follower_question,
        "follower_cid": follower_condition_id,
        "follower_slug": follower_slug,
        "token_id": token_id,
        "leader": leader_question,
        "leader_outcome": leader_outcome,
        "confidence": round(confidence, 3),
        "rationale": rationale[:300],
        "bet_usdc": round(bet_size_usdc, 2),
        "order_id": order_id,
        "executed": executed,
        "error": error,
        "category": category,
        # Filled in later by reconciliation / on-chain lookup
        "outcome": None,
        "pnl_usdc": None,
        "resolved_at": None,
    }


def save_run(
    *,
    mode: str,
    balance_before: float | None,
    balance_after: float | None,
    markets_scanned: int,
    clusters: int,
    relations_discovered: int,
    trades: list[dict[str, Any]],
) -> Path:
    """Append a single run record to the appropriate history file.

    - LIVE runs → ``history/runs_live.jsonl``
    - PAPER runs → ``history/runs_paper.jsonl``
    """
    now = datetime.now(timezone.utc)

    executed = [t for t in trades if t.get("executed")]
    failed = [t for t in trades if not t.get("executed") and t.get("error") and t["error"] != "Paper trade"]
    total_deployed = sum(t.get("bet_usdc", 0) for t in executed)

    record = {
        "ts": now.isoformat(),
        "mode": mode,
        "bal_before": round(balance_before, 2) if balance_before is not None else None,
        "bal_after": round(balance_after, 2) if balance_after is not None else None,
        "markets": markets_scanned,
        "clusters": clusters,
        "relations": relations_discovered,
        "n_executed": len(executed),
        "n_failed": len(failed),
        "deployed_usdc": round(total_deployed, 2),
        "trades": trades,
    }

    target = LIVE_FILE if mode == "live" else PAPER_FILE
    with open(target, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info("Run record saved to %s (%d trades)", target, len(trades))
    return target
