"""Pydantic types mirroring the paper's ATypes + internal data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paper ATypes (Section 3.1)
# ---------------------------------------------------------------------------

class SingleMarket(BaseModel):
    """A single binary prediction market."""
    question: str
    market_start_time: str
    market_end_time: str


class MarketRelation(BaseModel):
    """A discovered relationship between two markets."""
    question_i: str
    question_j: str
    is_same_outcome: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class MarketRelationList(BaseModel):
    """Collection of discovered market relationships within a cluster."""
    relations: list[MarketRelation]


class ClusterLabel(BaseModel):
    """Category label assigned to a cluster."""
    category: str


# ---------------------------------------------------------------------------
# Internal data models
# ---------------------------------------------------------------------------

class MarketOutcome(str, Enum):
    YES = "Yes"
    NO = "No"


class ResolvedMarket(BaseModel):
    """Full resolved market record used for backtesting."""
    condition_id: str
    question: str
    market_slug: str = ""
    outcome: str  # "Yes" or "No"
    market_start_time: datetime
    market_end_time: datetime
    resolved_on: Optional[datetime] = None
    tokens: list[dict] = Field(default_factory=list)
    last_trade_price: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None

    @property
    def yes_token_id(self) -> Optional[str]:
        for t in self.tokens:
            if t.get("outcome", "").lower() == "yes":
                return t.get("token_id")
        return None

    @property
    def no_token_id(self) -> Optional[str]:
        for t in self.tokens:
            if t.get("outcome", "").lower() == "no":
                return t.get("token_id")
        return None


class PricePoint(BaseModel):
    """Single price observation."""
    timestamp: datetime
    price: float


class ClusterAssignment(BaseModel):
    """Market assigned to a cluster with an index."""
    cluster_id: int
    market: ResolvedMarket


class TradeResult(BaseModel):
    """Result of a single leader-follower trade."""
    leader_question: str
    follower_question: str
    is_same_outcome: bool
    confidence_score: float
    leader_outcome: str
    follower_outcome: str
    side: str  # "YES" or "NO"
    entry_price: float
    pnl: float
    correct: bool


class BacktestReport(BaseModel):
    """Summary of a full backtest run."""
    total_trades: int
    correct_trades: int
    accuracy: float
    total_invested: float
    total_pnl: float
    roi: float
    trades: list[TradeResult]
