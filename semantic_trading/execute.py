"""Trade execution on Polymarket via py-clob-client with dynamic position sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from semantic_trading.config import (
    BANKRUPT_THRESHOLD,
    BET_FRACTION,
    CLOB_API_BASE,
    LOW_BALANCE_THRESHOLD,
    MAX_EXPOSURE_PER_MARKET,
    MAX_TRADES_PER_RUN,
    MIN_BET_USDC,
    POLY_API_KEY,
    POLY_API_SECRET,
    POLY_FUNDER,
    POLY_PASSPHRASE,
    POLYMARKET_PRIVATE_KEY,
    SKIP_EXPOSED_MARKETS,
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


@dataclass
class Position:
    """Represents an existing position on a market."""
    token_id: str
    condition_id: str
    size: float  # number of shares
    avg_price: float  # average entry price (0-1)
    side: str  # YES or NO
    cost_basis: float = 0.0  # approximate USDC invested

    def __post_init__(self):
        self.cost_basis = self.size * self.avg_price


@dataclass
class ExposureInfo:
    """Tracks exposure across all positions."""
    positions: list[Position] = field(default_factory=list)
    exposure_by_condition: dict[str, float] = field(default_factory=dict)
    exposure_by_token: dict[str, float] = field(default_factory=dict)
    total_exposure: float = 0.0

    def get_market_exposure(self, condition_id: str) -> float:
        """Get total USDC exposure for a specific market."""
        return self.exposure_by_condition.get(condition_id, 0.0)

    def is_market_exposed(self, condition_id: str) -> bool:
        """Check if we have any exposure to this market."""
        return condition_id in self.exposure_by_condition

    def get_remaining_capacity(self, condition_id: str, balance: float) -> float:
        """Get remaining capacity to add exposure to a market.

        Returns the max additional USDC we can deploy on this market
        without exceeding MAX_EXPOSURE_PER_MARKET.
        """
        current = self.get_market_exposure(condition_id)
        max_allowed = balance * MAX_EXPOSURE_PER_MARKET
        return max(0.0, max_allowed - current)


def _get_clob_client():
    """Lazily import and initialize the CLOB client with L2 auth."""
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    if POLY_API_KEY and POLY_API_SECRET and POLY_PASSPHRASE:
        # Ensure base64 padding — Polymarket may return secrets without trailing '='
        secret = POLY_API_SECRET
        if len(secret) % 4:
            secret += "=" * (4 - len(secret) % 4)
        creds = ApiCreds(
            api_key=POLY_API_KEY,
            api_secret=secret,
            api_passphrase=POLY_PASSPHRASE,
        )
        client = ClobClient(
            CLOB_API_BASE,
            key=POLYMARKET_PRIVATE_KEY,
            chain_id=CHAIN_ID,
            creds=creds,
            signature_type=1,
            funder=POLY_FUNDER or None,
        )
    else:
        client = ClobClient(CLOB_API_BASE, key=POLYMARKET_PRIVATE_KEY, chain_id=CHAIN_ID)
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

    return client


def fetch_positions() -> ExposureInfo:
    """Fetch all current positions from Polymarket and calculate exposure.

    Returns an ExposureInfo object with:
    - List of all positions
    - Exposure by condition_id (market)
    - Exposure by token_id
    - Total exposure across all positions
    """
    if not POLYMARKET_PRIVATE_KEY:
        logger.warning("No private key set, cannot fetch positions")
        return ExposureInfo()

    try:
        client = _get_clob_client()
        raw_positions = client.get_positions()

        if not raw_positions:
            logger.info("No existing positions found")
            return ExposureInfo()

        positions: list[Position] = []
        exposure_by_condition: dict[str, float] = {}
        exposure_by_token: dict[str, float] = {}
        total_exposure = 0.0

        for pos in raw_positions:
            try:
                asset = pos.get("asset", {})
                token_id = asset.get("token_id", "")
                condition_id = asset.get("condition_id", "")
                size = float(pos.get("size", 0))
                avg_price = float(pos.get("avgPrice", 0))
                side = pos.get("side", "").upper()

                if size <= 0:
                    continue

                position = Position(
                    token_id=token_id,
                    condition_id=condition_id,
                    size=size,
                    avg_price=avg_price,
                    side=side,
                )
                positions.append(position)

                exposure_by_condition[condition_id] = (
                    exposure_by_condition.get(condition_id, 0.0) + position.cost_basis
                )
                exposure_by_token[token_id] = (
                    exposure_by_token.get(token_id, 0.0) + position.cost_basis
                )
                total_exposure += position.cost_basis

            except (KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to parse position: %s — %s", pos, e)
                continue

        logger.info(
            "Fetched %d positions, total exposure: $%.2f across %d markets",
            len(positions), total_exposure, len(exposure_by_condition)
        )

        return ExposureInfo(
            positions=positions,
            exposure_by_condition=exposure_by_condition,
            exposure_by_token=exposure_by_token,
            total_exposure=total_exposure,
        )

    except Exception as e:
        logger.error("Failed to fetch positions: %s", e)
        return ExposureInfo()


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
