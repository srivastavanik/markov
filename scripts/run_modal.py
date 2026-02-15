#!/usr/bin/env python3
"""
Run experiment series on Modal serverless compute.

Games within a series run in parallel, drastically reducing wall-clock time.
A 15-game series that takes ~5-8 hours sequentially completes in ~20-30 min.

Setup (one-time):
    pip install modal
    modal token new
    modal secret create markov-api-keys \
        ANTHROPIC_API_KEY=sk-ant-... \
        OPENAI_API_KEY=sk-... \
        GOOGLE_API_KEY=... \
        XAI_API_KEY=...

Usage:
    python -m scripts.run_modal --type standard --games 15
    python -m scripts.run_modal --type single_provider --provider anthropic --games 5
    python -m scripts.run_modal --type all --games 10
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from markov.attribution import build_attribution_report
from markov.metrics import SeriesMetrics
from markov.modal_app import run_game_remote
from markov.persistence import (
    persist_game_from_result,
    persist_series,
    save_agent_traces_to_disk,
    upload_agent_traces_to_s3,
    upload_game_to_s3,
    upload_series_index_to_s3,
)
from markov.series import (
    CONFIG_BUILDERS,
    build_single_provider_config,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = ["anthropic", "openai", "google", "xai"]

logger = logging.getLogger("markov.modal_runner")


def _save_game_result(game_dir: Path, result: dict) -> None:
    """Save a remote game result to disk in the standard format."""
    game_dir.mkdir(parents=True, exist_ok=True)

    with open(game_dir / "game.json", "w") as f:
        json.dump(result["game_data"], f, indent=2, default=str)

    with open(game_dir / "analysis.json", "w") as f:
        json.dump(result["analysis"], f, indent=2, default=str)

    if result.get("metrics"):
        with open(game_dir / "metrics.json", "w") as f:
            json.dump(result["metrics"], f, indent=2, default=str)

    with open(game_dir / "highlights.json", "w") as f:
        json.dump(result["highlights"], f, indent=2, default=str)

    with open(game_dir / "transcript.md", "w") as f:
        f.write(result["transcript"])

    if result.get("agents"):
        with open(game_dir / "agents.json", "w") as f:
            json.dump(result["agents"], f, indent=2, default=str)


def run_series_modal(
    series_type: str,
    num_games: int = 15,
    provider: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run a series with games fanned out in parallel on Modal."""
    # Generate config locally (where config.yaml exists)
    if series_type == "single_provider":
        if provider is None:
            raise ValueError("provider is required for single_provider series")
        config = build_single_provider_config(provider)
    else:
        builder = CONFIG_BUILDERS.get(series_type)
        if builder is None:
            raise ValueError(f"Unknown series type: {series_type}")
        config = builder()

    config_json = config.model_dump_json()

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{series_type}_{provider}" if provider else series_type
    series_id = f"{label}_{timestamp}"
    series_dir = _PROJECT_ROOT / "data" / "series" / series_id
    series_dir.mkdir(parents=True, exist_ok=True)

    # Save series config
    with open(series_dir / "series_config.json", "w") as f:
        json.dump({
            "series_type": series_type,
            "provider": provider,
            "num_games": num_games,
            "config": config.model_dump(),
            "execution": "modal",
        }, f, indent=2, default=str)

    # Fan out all games in parallel via Modal .map()
    game_ids = [f"game_{i + 1:03d}" for i in range(num_games)]
    config_jsons = [config_json] * num_games

    print(f"Launching {num_games} games in parallel on Modal...")

    series_metrics = SeriesMetrics()
    total_cost: dict[str, float] = {}
    game_num = 0
    all_results: list[dict] = []

    for result in run_game_remote.map(config_jsons, game_ids):
        game_num += 1
        gid = result["game_id"]
        winner = result["result"].get("winner_name", "none")
        rounds = result["result"].get("total_rounds", "?")
        print(f"  [{game_num}/{num_games}] {gid} complete. Winner: {winner}. Rounds: {rounds}")

        # Save to disk
        _save_game_result(series_dir / gid, result)

        # Extract per-agent traces (for agent review)
        save_agent_traces_to_disk(result, series_dir / gid)

        # Persist to Supabase (fire-and-forget)
        persist_game_from_result(result, series_id=series_id, series_type=series_type)

        # Upload to S3 (fire-and-forget)
        upload_game_to_s3(result, series_id=series_id)
        upload_agent_traces_to_s3(result, series_id=series_id)

        # Keep lightweight copy for series index (drop heavy game_data)
        all_results.append({
            k: v for k, v in result.items() if k != "game_data"
        })

        # Accumulate metrics
        if result.get("metrics"):
            series_metrics.add_game(result["metrics"])

        # Accumulate costs
        for k, v in result.get("cost", {}).items():
            if isinstance(v, (int, float)):
                total_cost[k] = total_cost.get(k, 0) + v

    # Compute and save aggregate
    aggregate = series_metrics.compute()
    aggregate["series_id"] = series_id
    aggregate["series_type"] = series_type
    aggregate["provider"] = provider
    aggregate["total_cost"] = total_cost

    with open(series_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    # Write summary report
    report = _build_series_report(series_id, series_type, num_games, aggregate)
    with open(series_dir / "report.md", "w") as f:
        f.write(report)

    # Persist series aggregate + S3 index
    persist_series(series_id, series_type, num_games, aggregate)
    upload_series_index_to_s3(series_id, series_type, all_results, aggregate)

    print(f"\nSeries {series_id} complete. Results saved to {series_dir}")
    return aggregate


def _build_series_report(
    series_id: str,
    series_type: str,
    num_games: int,
    aggregate: dict,
) -> str:
    """Generate a Markdown summary report for the series."""
    lines: list[str] = []
    lines.append(f"# Series Report: {series_id}")
    lines.append("")
    lines.append(f"**Type:** {series_type}")
    lines.append(f"**Games:** {num_games}")
    lines.append(f"**Execution:** Modal (parallel)")
    lines.append("")

    win_rates = aggregate.get("win_rate_by_provider", {})
    if win_rates:
        lines.append("## Win Rates by Provider")
        lines.append("")
        for prov, rate in sorted(win_rates.items()):
            lines.append(f"- **{prov}**: {rate:.1%}")
        lines.append("")

    win_by_tier = aggregate.get("win_rate_by_tier", {})
    if win_by_tier:
        lines.append("## Win Rates by Tier")
        lines.append("")
        tier_labels = {1: "Boss", 2: "Lieutenant", 3: "Soldier"}
        for tier, rate in sorted(win_by_tier.items(), key=lambda x: int(x[0])):
            label = tier_labels.get(int(tier), f"Tier {tier}")
            lines.append(f"- **{label}**: {rate:.1%}")
        lines.append("")

    per_provider = aggregate.get("per_provider", {})
    if per_provider:
        lines.append("## Provider Behavioral Summary")
        lines.append("")
        lines.append("| Provider | Deception | Malice | Unprompted Malice | Safety | Guilt | Avg Survival |")
        lines.append("|---|---|---|---|---|---|---|")
        for prov, data in sorted(per_provider.items()):
            lines.append(
                f"| {prov} "
                f"| {data.get('avg_deception_delta', 0):.3f} "
                f"| {data.get('avg_malice_rate', 0):.3f} "
                f"| {data.get('avg_unprompted_malice_rate', 0):.3f} "
                f"| {data.get('avg_safety_artifact_rate', 0):.3f} "
                f"| {data.get('avg_guilt_rate', 0):.3f} "
                f"| {data.get('avg_survival_rounds', 0):.1f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _run_all(args: argparse.Namespace) -> None:
    """Run all series types for full attribution analysis."""
    all_results: dict[str, dict] = {}

    # Series A: Standard
    print(f"\n{'=' * 60}")
    print("SERIES A: Standard")
    print(f"{'=' * 60}")
    all_results["standard"] = run_series_modal(
        "standard", num_games=args.games, verbose=args.verbose,
    )

    # Series B: Single-Provider (one per provider)
    for prov in PROVIDERS:
        print(f"\n{'=' * 60}")
        print(f"SERIES B: Single-Provider ({prov})")
        print(f"{'=' * 60}")
        result = run_series_modal(
            "single_provider", num_games=args.games, provider=prov, verbose=args.verbose,
        )
        if "single_provider" not in all_results:
            all_results["single_provider"] = result
        else:
            existing = all_results["single_provider"]
            for p, data in result.get("per_provider", {}).items():
                existing.setdefault("per_provider", {})[p] = data

    # Series C: Shuffled Families
    print(f"\n{'=' * 60}")
    print("SERIES C: Shuffled Families")
    print(f"{'=' * 60}")
    all_results["shuffled"] = run_series_modal(
        "shuffled", num_games=args.games, verbose=args.verbose,
    )

    # Series D: No Family
    print(f"\n{'=' * 60}")
    print("SERIES D: No Family")
    print(f"{'=' * 60}")
    all_results["no_family"] = run_series_modal(
        "no_family", num_games=args.games, verbose=args.verbose,
    )

    # Series E: Flat Hierarchy
    print(f"\n{'=' * 60}")
    print("SERIES E: Flat Hierarchy")
    print(f"{'=' * 60}")
    all_results["flat_hierarchy"] = run_series_modal(
        "flat_hierarchy", num_games=args.games, verbose=args.verbose,
    )

    # Series F: Flat Temperature
    print(f"\n{'=' * 60}")
    print("SERIES F: Flat Temperature (0.7 across all tiers)")
    print(f"{'=' * 60}")
    all_results["flat_temperature"] = run_series_modal(
        "flat_temperature", num_games=args.games, verbose=args.verbose,
    )

    # Attribution analysis
    print(f"\n{'=' * 60}")
    print("ATTRIBUTION ANALYSIS")
    print(f"{'=' * 60}")
    report = build_attribution_report(all_results)
    print(report)

    output_dir = _PROJECT_ROOT / "data" / "series"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "attribution_report.md", "w") as f:
        f.write(report)
    with open(output_dir / "all_series_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAttribution report saved to data/series/attribution_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Markov experiment series on Modal")
    parser.add_argument(
        "--type",
        choices=[
            "standard", "single_provider", "shuffled",
            "no_family", "flat_hierarchy", "flat_temperature", "all",
        ],
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
        _run_all(args)
    else:
        if args.type == "single_provider" and not args.provider:
            print("Error: --provider required for single_provider series")
            sys.exit(1)
        result = run_series_modal(args.type, args.games, args.provider, args.verbose)
        win_rates = result.get("win_rate_by_provider", {})
        if win_rates:
            print("Win rates:")
            for prov, rate in sorted(win_rates.items()):
                print(f"  {prov}: {rate:.1%}")


if __name__ == "__main__":
    main()
