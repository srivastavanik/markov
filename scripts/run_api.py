#!/usr/bin/env python3
"""Run MARKOV API service."""
from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    # Configure root logger so markov.* messages appear in the terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Ensure markov loggers propagate at DEBUG+ so LLM errors are visible
    logging.getLogger("markov").setLevel(logging.DEBUG)
    # Suppress noisy HTTP access logs (dashboard polls /api/games every few seconds)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    host = os.getenv("MARKOV_API_HOST", "0.0.0.0")
    port = int(os.getenv("MARKOV_API_PORT", "8000"))
    uvicorn.run("markov.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

