from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
POLYMARKET_PRIVATE_KEY: str = os.environ.get("POLYMARKET_PRIVATE_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

CLUSTER_CATEGORIES = [
    "politics",
    "geopolitics",
    "elections",
    "economy",
    "finance",
    "earnings",
    "crypto",
    "tech",
    "sports",
    "culture",
    "other",
]

CONFIDENCE_THRESHOLD = 0.5
ENTRY_PRICE_CUTOFF = 0.1
TERMINAL_PRICE_CUTOFF = 0.1
MIN_MARKET_DURATION_DAYS = 7
