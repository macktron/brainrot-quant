"""Market exposure tracking to prevent duplicate positions.

Tracks existing positions from:
1. Trade history (runs_live.jsonl) - what we've already traded
2. Polymarket API - actual on-chain positions (optional)

Provides filtering to skip markets we're already exposed to.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from semantic_trading.config import (
    CLOB_API_BASE,
    MAX_EXPOSURE_PER_MARKET_USDC,
    MAX_EXPOSURE_PER_CATEGORY_USDC,
    MAX_TOTAL_EXPOSURE_USDC,
    POLY_API_KEY,
    POLY_API_SECRET,
    POLY_FUNDER,
    POLY_PASSPHRASE,
    POLYMARKET_PRIVATE_KEY,
    SKIP_EXISTING_POSITIONS,
)

logger = logging.getLogger(__name__)

HISTORY_DIR = Path(__file__).resolve().parent.parent / "history"
LIVE_FILE = HISTORY_DIR / "runs_live.jsonl"


@dataclass
class Position:
    """An existing position in a market."""
    condition_id: str
    token_id: str
    side: str  # "YES" or "NO"
    size_usdc: float
    question: str = ""
    category: str = ""
    entry_time: datetime | None = None


@dataclass
class ExposureInfo:
    """Current exposure state across all positions."""
    positions: dict[str, Position] = field(default_factory=dict)  # condition_id -> Position
    total_exposure_usdc: float = 0.0
    exposure_by_category: dict[str, float] = field(default_factory=dict)
    exposure_by_market: dict[str, float] = field(default_factory=dict)

    def has_position(self, condition_id: str) -> bool:
        """Check if we already have a position in this market."""
        return condition_id in self.positions

    def get_position(self, condition_id: str) -> Position | None:
        """Get existing position for a market, if any."""
        return self.positions.get(condition_id)

    def market_exposure(self, condition_id: str) -> float:
        """Get current exposure in USDC for a specific market."""
        return self.exposure_by_market.get(condition_id, 0.0)

    def category_exposure(self, category: str) -> float:
        """Get current exposure in USDC for a specific category."""
        return self.exposure_by_category.get(category, 0.0)

    def can_trade_market(
        self,
        condition_id: str,
        proposed_size_usdc: float,
        category: str = "",
    ) -> tuple[bool, str]:
        """
        Check if we can place a trade on this market.

        Returns (can_trade, reason) tuple.
        """
        if SKIP_EXISTING_POSITIONS and self.has_position(condition_id):
            return False, f"Already have position in this market"

        current_market_exposure = self.market_exposure(condition_id)
        new_market_exposure = current_market_exposure + proposed_size_usdc

        if MAX_EXPOSURE_PER_MARKET_USDC > 0 and new_market_exposure > MAX_EXPOSURE_PER_MARKET_USDC:
            return False, f"Would exceed per-market limit (${new_market_exposure:.2f} > ${MAX_EXPOSURE_PER_MARKET_USDC:.2f})"

        if category and MAX_EXPOSURE_PER_CATEGORY_USDC > 0:
            current_cat_exposure = self.category_exposure(category)
            new_cat_exposure = current_cat_exposure + proposed_size_usdc
            if new_cat_exposure > MAX_EXPOSURE_PER_CATEGORY_USDC:
                return False, f"Would exceed {category} category limit (${new_cat_exposure:.2f} > ${MAX_EXPOSURE_PER_CATEGORY_USDC:.2f})"

        if MAX_TOTAL_EXPOSURE_USDC > 0:
            new_total = self.total_exposure_usdc + proposed_size_usdc
            if new_total > MAX_TOTAL_EXPOSURE_USDC:
                return False, f"Would exceed total exposure limit (${new_total:.2f} > ${MAX_TOTAL_EXPOSURE_USDC:.2f})"

        return True, "OK"

    def add_position(
        self,
        condition_id: str,
        token_id: str,
        side: str,
        size_usdc: float,
        question: str = "",
        category: str = "",
    ) -> None:
        """Track a new position (call after executing a trade)."""
        self.positions[condition_id] = Position(
            condition_id=condition_id,
            token_id=token_id,
            side=side,
            size_usdc=size_usdc,
            question=question,
            category=category,
            entry_time=datetime.now(timezone.utc),
        )
        self.exposure_by_market[condition_id] = self.exposure_by_market.get(condition_id, 0.0) + size_usdc
        if category:
            self.exposure_by_category[category] = self.exposure_by_category.get(category, 0.0) + size_usdc
        self.total_exposure_usdc += size_usdc


def load_exposure_from_history() -> ExposureInfo:
    """
    Load exposure state from trade history.

    Reads runs_live.jsonl to find all executed trades that haven't resolved yet.
    This gives us a conservative view of current exposure.
    """
    exposure = ExposureInfo()

    if not LIVE_FILE.exists():
        logger.info("No trade history found at %s", LIVE_FILE)
        return exposure

    try:
        with open(LIVE_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    run = json.loads(line)
                except json.JSONDecodeError:
                    continue

                trades = run.get("trades", [])
                for trade in trades:
                    if not trade.get("executed"):
                        continue

                    condition_id = trade.get("follower_cid", "")
                    if not condition_id:
                        continue

                    if trade.get("outcome") is not None:
                        continue

                    size_usdc = trade.get("bet_usdc", 0.0)
                    if size_usdc <= 0:
                        continue

                    exposure.add_position(
                        condition_id=condition_id,
                        token_id=trade.get("token_id", ""),
                        side=trade.get("side", ""),
                        size_usdc=size_usdc,
                        question=trade.get("follower", ""),
                        category=trade.get("category", ""),
                    )

        logger.info(
            "Loaded exposure from history: %d positions, $%.2f total",
            len(exposure.positions),
            exposure.total_exposure_usdc,
        )

    except Exception as e:
        logger.error("Failed to load exposure from history: %s", e)

    return exposure


def fetch_positions_from_polymarket() -> dict[str, Position]:
    """
    Fetch actual open positions from Polymarket CLOB API.

    Returns a dict of condition_id -> Position for all markets where we hold tokens.
    """
    positions: dict[str, Position] = {}

    if not POLYMARKET_PRIVATE_KEY:
        logger.warning("No POLYMARKET_PRIVATE_KEY, cannot fetch positions")
        return positions

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        if POLY_API_KEY and POLY_API_SECRET and POLY_PASSPHRASE:
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
                chain_id=137,
                creds=creds,
                signature_type=1,
                funder=POLY_FUNDER or None,
            )
        else:
            client = ClobClient(CLOB_API_BASE, key=POLYMARKET_PRIVATE_KEY, chain_id=137)
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)

        open_positions = client.get_positions()

        if not open_positions:
            logger.info("No open positions found on Polymarket")
            return positions

        for pos in open_positions:
            condition_id = pos.get("conditionId") or pos.get("condition_id", "")
            if not condition_id:
                continue

            size = float(pos.get("size", 0))
            if size <= 0:
                continue

            avg_price = float(pos.get("avgPrice", 0) or pos.get("average_price", 0.5))
            size_usdc = size * avg_price

            outcome = pos.get("outcome", "").upper()
            side = "YES" if outcome == "YES" else "NO" if outcome == "NO" else ""

            positions[condition_id] = Position(
                condition_id=condition_id,
                token_id=pos.get("tokenId") or pos.get("token_id", ""),
                side=side,
                size_usdc=size_usdc,
                question=pos.get("question", ""),
            )

        logger.info("Fetched %d open positions from Polymarket", len(positions))

    except ImportError:
        logger.warning("py-clob-client not available, skipping Polymarket position fetch")
    except Exception as e:
        logger.error("Failed to fetch positions from Polymarket: %s", e)

    return positions


def load_full_exposure(include_api_positions: bool = True) -> ExposureInfo:
    """
    Load complete exposure state from both history and Polymarket API.

    Args:
        include_api_positions: If True, also fetch positions from Polymarket API.
                               This is more accurate but requires API credentials.
    """
    exposure = load_exposure_from_history()

    if include_api_positions:
        api_positions = fetch_positions_from_polymarket()

        for condition_id, pos in api_positions.items():
            if condition_id not in exposure.positions:
                exposure.add_position(
                    condition_id=pos.condition_id,
                    token_id=pos.token_id,
                    side=pos.side,
                    size_usdc=pos.size_usdc,
                    question=pos.question,
                    category=pos.category,
                )
            else:
                if pos.size_usdc > exposure.positions[condition_id].size_usdc:
                    diff = pos.size_usdc - exposure.positions[condition_id].size_usdc
                    exposure.exposure_by_market[condition_id] = pos.size_usdc
                    exposure.total_exposure_usdc += diff
                    exposure.positions[condition_id].size_usdc = pos.size_usdc

    logger.info(
        "Total exposure: %d positions, $%.2f across %d categories",
        len(exposure.positions),
        exposure.total_exposure_usdc,
        len(exposure.exposure_by_category),
    )

    return exposure


def filter_markets_by_exposure(
    condition_ids: set[str],
    exposure: ExposureInfo,
) -> set[str]:
    """
    Filter out markets we're already exposed to.

    Args:
        condition_ids: Set of market condition_ids to filter
        exposure: Current exposure state

    Returns:
        Set of condition_ids we can still trade (no existing position)
    """
    if not SKIP_EXISTING_POSITIONS:
        return condition_ids

    filtered = {cid for cid in condition_ids if not exposure.has_position(cid)}
    removed = len(condition_ids) - len(filtered)

    if removed > 0:
        logger.info("Filtered out %d markets with existing positions", removed)

    return filtered
