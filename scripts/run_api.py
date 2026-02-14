#!/usr/bin/env python3
"""Run MARKOV API service."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("MARKOV_API_HOST", "0.0.0.0")
    port = int(os.getenv("MARKOV_API_PORT", "8000"))
    uvicorn.run("markov.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

