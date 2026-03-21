"""Trade execution on Polymarket via py-clob-client with dynamic position sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from semantic_trading.config import (
    BANKRUPT_THRESHOLD,
    BET_FRACTION,
    CLOB_API_BASE,
    LOW_BALANCE_THRESHOLD,
    MAX_TRADES_PER_RUN,
    MIN_BET_USDC,
    POLYMARKET_PRIVATE_KEY,
)

logger = logging.getLogger(__name__)

CHAIN_ID = 137  # Polygon


@dataclass
class TradeExecution:
    success: bool
    order_id: str | None = None
    error: str | None = None
    amount_usdc: float | None = None


@dataclass
class BalanceInfo:
    balance_usdc: float
    is_bankrupt: bool
    is_low: bool


def _get_clob_client():
    """Lazily import and initialize the CLOB client."""
    from py_clob_client.client import ClobClient

    client = ClobClient(CLOB_API_BASE, key=POLYMARKET_PRIVATE_KEY, chain_id=CHAIN_ID)
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    return client


def fetch_balance() -> BalanceInfo:
    """Fetch current USDC balance from Polymarket."""
    if not POLYMARKET_PRIVATE_KEY:
        return BalanceInfo(balance_usdc=0.0, is_bankrupt=True, is_low=True)

    try:
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

        client = _get_clob_client()
        result = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )

        balance_wei = int(result.get("balance", "0"))
        balance_usdc = balance_wei / 1e6

        logger.info("Polymarket balance: $%.2f USDC", balance_usdc)

        return BalanceInfo(
            balance_usdc=balance_usdc,
            is_bankrupt=balance_usdc < BANKRUPT_THRESHOLD,
            is_low=balance_usdc < LOW_BALANCE_THRESHOLD,
        )

    except Exception as e:
        logger.error("Failed to fetch balance: %s", e)
        return BalanceInfo(balance_usdc=0.0, is_bankrupt=True, is_low=True)


def compute_trade_size(
    balance_usdc: float,
    confidence: float,
    trades_remaining: int,
) -> float:
    """
    Compute USDC bet size based on current balance, confidence, and remaining trade slots.

    Uses a confidence-scaled fraction of the balance allocated to this trade.
    Higher confidence → larger fraction (up to BET_FRACTION).
    """
    if balance_usdc < MIN_BET_USDC:
        return 0.0

    # Scale bet fraction by confidence: 0.80 conf → 80% of BET_FRACTION, 1.0 → 100%
    confidence_scalar = min(confidence / 0.90, 1.25)  # boost high-confidence trades
    fraction = BET_FRACTION * confidence_scalar

    # Divide remaining capital across remaining trade slots
    per_trade = balance_usdc * fraction
    if trades_remaining > 1:
        max_per_trade = balance_usdc / trades_remaining
        per_trade = min(per_trade, max_per_trade)

    # Never bet more than 30% of balance on a single trade regardless
    per_trade = min(per_trade, balance_usdc * 0.30)
    # Enforce minimum — if we can't meet it, don't trade
    if per_trade < MIN_BET_USDC:
        return 0.0

    return round(per_trade, 2)


def execute_trade(
    *,
    token_id: str,
    amount_usdc: float,
) -> TradeExecution:
    """Execute a market buy on Polymarket via FOK (fill-or-kill)."""
    if not POLYMARKET_PRIVATE_KEY:
        return TradeExecution(success=False, error="POLYMARKET_PRIVATE_KEY not set")

    if amount_usdc < MIN_BET_USDC:
        return TradeExecution(success=False, error=f"Below minimum bet: ${amount_usdc:.2f}")

    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        client = _get_clob_client()

        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount_usdc,
            side=BUY,
        )

        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)

        order_id = resp.get("orderID", resp.get("id", "unknown"))
        logger.info("Trade executed: order_id=%s, amount=$%.2f", order_id, amount_usdc)

        return TradeExecution(
            success=True,
            order_id=order_id,
            amount_usdc=amount_usdc,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error("Trade execution failed: %s", error_msg)
        return TradeExecution(success=False, error=error_msg)
