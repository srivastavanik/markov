#!/usr/bin/env python3
"""
Run a single game with random actions. Verifies the engine works end-to-end.

Usage:
    python -m scripts.run [--seed SEED]
"""
import argparse
import random
import sys

from markov.config import load_game_config
from markov.orchestrator import random_action_provider, run_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Markov game with random actions")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-round output")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[seed={args.seed}]")

    config = load_game_config()
    state = run_game(config=config, action_provider=random_action_provider, verbose=not args.quiet)

    if args.quiet:
        if state.winner:
            print(f"Winner: {state.winner.name} ({state.winner.family}) after {state.round_num} rounds")
        else:
            print(f"No winner after {state.round_num} rounds, {state.living_count} alive")

    sys.exit(0)


if __name__ == "__main__":
    main()
