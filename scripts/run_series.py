#!/usr/bin/env python3
"""
Run controlled experiment series.

Usage:
    python -m scripts.run_series --type standard --games 15
    python -m scripts.run_series --type single_provider --provider anthropic --games 5
    python -m scripts.run_series --type shuffled --games 10
    python -m scripts.run_series --type no_family --games 10
    python -m scripts.run_series --type flat_hierarchy --games 10
    python -m scripts.run_series --type all --games 3  # run all 5+ series types
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from markov.attribution import build_attribution_report
from markov.series import run_series


PROVIDERS = ["anthropic", "openai", "google", "xai"]


async def _run_single(args: argparse.Namespace) -> dict:
    """Run a single series."""
    result = await run_series(
        series_type=args.type,
        num_games=args.games,
        provider=args.provider,
        verbose=args.verbose,
    )
    print(f"\nSeries complete. {result.get('num_games', '?')} games.")
    win_rates = result.get("win_rate_by_provider", {})
    if win_rates:
        print("Win rates:")
        for prov, rate in sorted(win_rates.items()):
            print(f"  {prov}: {rate:.1%}")
    return result


async def _run_all(args: argparse.Namespace) -> None:
    """Run all series types for full attribution analysis."""
    all_results: dict[str, dict] = {}

    # Series A: Standard
    print("\n" + "=" * 60)
    print("SERIES A: Standard")
    print("=" * 60)
    all_results["standard"] = await run_series(
        "standard", num_games=args.games, verbose=args.verbose,
    )

    # Series B: Single-Provider (one per provider)
    for prov in PROVIDERS:
        print("\n" + "=" * 60)
        print(f"SERIES B: Single-Provider ({prov})")
        print("=" * 60)
        result = await run_series(
            "single_provider", num_games=args.games, provider=prov, verbose=args.verbose,
        )
        # Merge single-provider results under one key
        if "single_provider" not in all_results:
            all_results["single_provider"] = result
        else:
            # Merge per_provider data
            existing = all_results["single_provider"]
            for p, data in result.get("per_provider", {}).items():
                existing.setdefault("per_provider", {})[p] = data

    # Series C: Shuffled Families
    print("\n" + "=" * 60)
    print("SERIES C: Shuffled Families")
    print("=" * 60)
    all_results["shuffled"] = await run_series(
        "shuffled", num_games=args.games, verbose=args.verbose,
    )

    # Series D: No Family
    print("\n" + "=" * 60)
    print("SERIES D: No Family")
    print("=" * 60)
    all_results["no_family"] = await run_series(
        "no_family", num_games=args.games, verbose=args.verbose,
    )

    # Series E: Flat Hierarchy
    print("\n" + "=" * 60)
    print("SERIES E: Flat Hierarchy")
    print("=" * 60)
    all_results["flat_hierarchy"] = await run_series(
        "flat_hierarchy", num_games=args.games, verbose=args.verbose,
    )

    # Series F: Flat Temperature
    print("\n" + "=" * 60)
    print("SERIES F: Flat Temperature (0.7 across all tiers)")
    print("=" * 60)
    all_results["flat_temperature"] = await run_series(
        "flat_temperature", num_games=args.games, verbose=args.verbose,
    )

    # Attribution analysis
    print("\n" + "=" * 60)
    print("ATTRIBUTION ANALYSIS")
    print("=" * 60)
    report = build_attribution_report(all_results)
    print(report)

    # Save attribution report
    output_dir = Path("data/series")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "attribution_report.md", "w") as f:
        f.write(report)
    with open(output_dir / "all_series_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAttribution report saved to data/series/attribution_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Markov experiment series")
    parser.add_argument(
        "--type",
        choices=["standard", "single_provider", "shuffled", "no_family", "flat_hierarchy", "flat_temperature", "all"],
        default="standard",
        help="Series type to run",
    )
    parser.add_argument("--games", type=int, default=15, help="Number of games per series")
    parser.add_argument("--provider", type=str, default=None, help="Provider for single_provider series")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    if args.type == "all":
        asyncio.run(_run_all(args))
    else:
        if args.type == "single_provider" and not args.provider:
            print("Error: --provider required for single_provider series")
            sys.exit(1)
        asyncio.run(_run_single(args))


if __name__ == "__main__":
    main()
