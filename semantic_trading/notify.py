"""Discord webhook notifications for trade signals."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from semantic_trading.config import DISCORD_WEBHOOK_URL

logger = logging.getLogger(__name__)


def send_trade_notification(
    *,
    side: str,
    follower_question: str,
    leader_question: str,
    leader_outcome: str,
    confidence: float,
    rationale: str,
    amount_usdc: float | None = None,
    order_id: str | None = None,
    error: str | None = None,
) -> bool:
    """Send a trade signal notification to Discord. Returns True on success."""
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set, skipping notification")
        return False

    status_emoji = "🟢" if order_id else ("🔴" if error else "🟡")
    status_text = (
        f"Executed (order: `{order_id}`)"
        if order_id
        else (f"FAILED: {error}" if error else "Signal only (dry run)")
    )

    embed = {
        "title": f"{status_emoji} Trade Signal: BUY **{side}**",
        "color": 0x00CC66 if order_id else (0xFF4444 if error else 0xFFAA00),
        "fields": [
            {"name": "Follower Market", "value": follower_question[:200], "inline": False},
            {"name": "Leader Market", "value": leader_question[:200], "inline": False},
            {"name": "Leader Outcome", "value": leader_outcome, "inline": True},
            {"name": "Confidence", "value": f"{confidence:.0%}", "inline": True},
            {"name": "Status", "value": status_text, "inline": False},
            {"name": "Rationale", "value": rationale[:500], "inline": False},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if amount_usdc is not None:
        embed["fields"].insert(4, {
            "name": "Size",
            "value": f"${amount_usdc:.2f} USDC",
            "inline": True,
        })

    payload = {
        "username": "Semantic Trader",
        "embeds": [embed],
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(DISCORD_WEBHOOK_URL, json=payload)
            resp.raise_for_status()
        logger.info("Discord notification sent")
        return True
    except Exception as e:
        logger.error("Failed to send Discord notification: %s", e)
        return False


def send_summary_notification(
    *,
    markets_fetched: int,
    relations_discovered: int,
    trades_executed: int,
    trades_failed: int,
    dry_run: bool = False,
) -> bool:
    """Send a daily run summary to Discord."""
    if not DISCORD_WEBHOOK_URL:
        return False

    mode = "DRY RUN" if dry_run else "LIVE"
    embed = {
        "title": f"📊 Daily Run Summary ({mode})",
        "color": 0x5865F2,
        "fields": [
            {"name": "Markets Fetched", "value": str(markets_fetched), "inline": True},
            {"name": "Relations Discovered", "value": str(relations_discovered), "inline": True},
            {"name": "Trades Executed", "value": str(trades_executed), "inline": True},
            {"name": "Trades Failed", "value": str(trades_failed), "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    payload = {"username": "Semantic Trader", "embeds": [embed]}

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(DISCORD_WEBHOOK_URL, json=payload)
            resp.raise_for_status()
        return True
    except Exception:
        return False
