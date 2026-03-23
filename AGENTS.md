# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Semantic Trading is a stateless, single-shot prediction-market trading bot for Polymarket. It fetches markets, clusters them using OpenAI embeddings, labels clusters via GPT-4o, discovers leader-follower causal relationships, and executes trades when leaders resolve. See `README.md` for full strategy details.

### Running the application

- **Backtest (historical data):** `python run_backtest.py --max-markets 200`
- **Paper trading (live markets, no real money):** `python run_live.py --paper`
- **Live trading (real money, requires Polymarket credentials):** `python run_live.py --live`

Paper mode (`--paper`) is the safe default and does not require Polymarket private keys. Both backtest and paper modes require `OPENAI_API_KEY` (for embeddings and LLM calls) and outbound internet access (for Polymarket Gamma API, which is public/no-auth).

### Required environment variables

All secrets are injected automatically in Cloud Agent VMs. For local dev, copy `.env.example` to `.env` and fill in:
- `OPENAI_API_KEY` — required for all modes
- `POLYMARKET_PRIVATE_KEY`, `POLY_API_KEY`, `POLY_API_SECRET`, `POLY_PASSPHRASE`, `POLY_FUNDER` — required only for live trading
- `DISCORD_WEBHOOK_URL` — optional, for notifications

### Key caveats

- There are no automated tests in this repository. Validation is done by running the pipelines (`run_backtest.py`, `run_live.py --paper`).
- There is no linter or type-checker configured. Use `python -m py_compile <file>` for basic syntax checks.
- The project has no database, no Docker, no Makefile. Dependencies are managed entirely through `pyproject.toml` and installed via `pip install -e .`.
- The `data/` directory is created at runtime and is gitignored. Cached market data and signals are stored there.
- Run history is appended to `history/runs_paper.jsonl` (paper) or `history/runs_live.jsonl` (live).
- Sports clusters are automatically excluded from the discovery pipeline.
