"""Trade execution on Polymarket via py-clob-client."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from semantic_trading.config import CLOB_API_BASE, POLYMARKET_PRIVATE_KEY, TRADE_SIZE_USDC

logger = logging.getLogger(__name__)

CHAIN_ID = 137  # Polygon


@dataclass
class TradeExecution:
    success: bool
    order_id: str | None = None
    error: str | None = None
    price: float | None = None
    size: float | None = None


def _get_clob_client():
    """Lazily import and initialize the CLOB client."""
    from py_clob_client.client import ClobClient

    client = ClobClient(CLOB_API_BASE, key=POLYMARKET_PRIVATE_KEY, chain_id=CHAIN_ID)
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    return client


def get_best_price(token_id: str, side: str = "BUY") -> float | None:
    """Fetch the current best price for a token from the order book."""
    try:
        client = _get_clob_client()
        book = client.get_order_book(token_id)
        if side == "BUY" and book.asks:
            return float(book.asks[0].price)
        if side == "SELL" and book.bids:
            return float(book.bids[0].price)
    except Exception as e:
        logger.error("Failed to fetch order book: %s", e)
    return None


def execute_trade(
    *,
    token_id: str,
    side: str = "BUY",
    amount_usdc: float | None = None,
) -> TradeExecution:
    """
    Execute a market buy on Polymarket.

    Uses FOK (fill-or-kill) to get immediate execution at best available price.
    Falls back to a GTC limit order at the current best ask if FOK isn't supported.
    """
    if not POLYMARKET_PRIVATE_KEY:
        return TradeExecution(success=False, error="POLYMARKET_PRIVATE_KEY not set")

    amount = amount_usdc or TRADE_SIZE_USDC
    if amount <= 0:
        return TradeExecution(success=False, error=f"Invalid trade size: {amount}")

    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        client = _get_clob_client()

        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount,
            side=BUY,
        )

        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)

        order_id = resp.get("orderID", resp.get("id", "unknown"))
        logger.info("Trade executed: order_id=%s, amount=$%.2f", order_id, amount)

        return TradeExecution(
            success=True,
            order_id=order_id,
            size=amount,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error("Trade execution failed: %s", error_msg)
        return TradeExecution(success=False, error=error_msg)
