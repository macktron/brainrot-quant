"""Discord webhook notifications for trade signals, alerts, and summaries."""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone

import httpx

from semantic_trading.config import DISCORD_NOTIFY_EXECUTED_TRADES, DISCORD_WEBHOOK_URL

logger = logging.getLogger(__name__)


def _send_discord(payload: dict) -> bool:
    """Send a payload to the Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set, skipping notification")
        return False
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(DISCORD_WEBHOOK_URL, json=payload)
            resp.raise_for_status()
        return True
    except Exception as e:
        logger.error("Failed to send Discord notification: %s", e)
        return False


def send_trade_notification(
    *,
    side: str,
    follower_question: str,
    leader_question: str,
    leader_outcome: str,
    confidence: float,
    rationale: str,
    amount_usdc: float | None = None,
    balance_after: float | None = None,
    order_id: str | None = None,
    error: str | None = None,
) -> bool:
    """Send a Discord embed for a successfully executed live order (filled trades only)."""
    if not DISCORD_NOTIFY_EXECUTED_TRADES:
        return False
    status_emoji = "🟢" if order_id else ("🔴" if error else "🟡")
    status_text = (
        f"Filled (order: `{order_id}`)"
        if order_id
        else (f"FAILED: {error}" if error else "Signal only (paper trade)")
    )

    embed = {
        "title": f"{status_emoji} Trade filled: BUY **{side}**",
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
            "name": "Bet Size",
            "value": f"${amount_usdc:.2f} USDC",
            "inline": True,
        })

    if balance_after is not None:
        embed["fields"].insert(5, {
            "name": "Balance After",
            "value": f"${balance_after:.2f} USDC",
            "inline": True,
        })

    return _send_discord({"username": "Semantic Trader", "embeds": [embed]})


def send_summary_notification(
    *,
    markets_fetched: int,
    relations_discovered: int,
    trades_executed: int,
    trades_failed: int,
    balance_usdc: float | None = None,
    total_deployed: float = 0.0,
    dry_run: bool = False,
    run_digest: str | None = None,
    trades_taken_lines: list[str] | None = None,
) -> bool:
    """Send a daily run summary to Discord."""
    mode = "PAPER TRADE" if dry_run else "LIVE"
    embed = {
        "title": f"📊 Daily Run Summary ({mode})",
        "color": 0x5865F2,
        "fields": [
            {"name": "Markets Scanned", "value": str(markets_fetched), "inline": True},
            {"name": "Relations Found", "value": str(relations_discovered), "inline": True},
            {"name": "Trades Executed", "value": str(trades_executed), "inline": True},
            {"name": "Trades Failed", "value": str(trades_failed), "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if balance_usdc is not None:
        embed["fields"].append({
            "name": "Current Balance",
            "value": f"${balance_usdc:.2f} USDC",
            "inline": True,
        })

    if total_deployed > 0:
        embed["fields"].append({
            "name": "Deployed This Run",
            "value": f"${total_deployed:.2f} USDC",
            "inline": True,
        })

    if trades_taken_lines:
        body = "\n".join(trades_taken_lines[:12])
        if len(trades_taken_lines) > 12:
            body += f"\n… +{len(trades_taken_lines) - 12} more"
        embed["fields"].append({
            "name": "Trades taken",
            "value": body[:1020] or "(none)",
            "inline": False,
        })

    if run_digest:
        embed["fields"].append({
            "name": "Why (this run)",
            "value": run_digest[:1020],
            "inline": False,
        })

    return _send_discord({"username": "Semantic Trader", "embeds": [embed]})


def send_balance_alert(
    *,
    balance_usdc: float,
    is_bankrupt: bool,
) -> bool:
    """Send a low-balance or bankruptcy alert to Discord."""
    if is_bankrupt:
        embed = {
            "title": "🚨 BANKRUPT — Trading Halted",
            "description": (
                f"Balance is **${balance_usdc:.2f} USDC** — below the $1.00 minimum.\n\n"
                "**All trading has been stopped.** Deposit more USDC to your "
                "Polymarket account to resume."
            ),
            "color": 0xFF0000,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        embed = {
            "title": "⚠️ Low Balance Warning",
            "description": (
                f"Balance is **${balance_usdc:.2f} USDC** — getting low.\n\n"
                "Trades will continue but sizes will be small. "
                "Consider depositing more USDC."
            ),
            "color": 0xFF8800,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return _send_discord({
        "username": "Semantic Trader",
        "content": "@everyone" if is_bankrupt else "",
        "embeds": [embed],
    })


def send_error_alert(*, error: Exception, context: str = "pipeline") -> bool:
    """Send an unhandled error alert to Discord."""
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    tb_text = "".join(tb)[-1500:]  # last 1500 chars of traceback

    embed = {
        "title": f"💥 Pipeline Crash: {context}",
        "description": f"```\n{tb_text}\n```",
        "color": 0xFF0000,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return _send_discord({
        "username": "Semantic Trader",
        "content": "@everyone",
        "embeds": [embed],
    })
