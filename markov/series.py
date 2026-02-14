"""
Series runner and config generators for controlled experiments.

Series A: Standard -- 4 provider-aligned families, mixed tiers
Series B: Single-Provider -- all 12 agents from one provider
Series C: Shuffled Families -- cross-provider families
Series D: No Family -- family discussion disabled
Series E: Flat Hierarchy -- all tier 1 (Boss models)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from markov.config import AgentConfig, FamilyConfig, GameConfig, load_game_config
from markov.metrics import SeriesMetrics

logger = logging.getLogger("markov.series")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Model registry: maps (provider, tier) -> model string
# ---------------------------------------------------------------------------

PROVIDER_MODELS: dict[str, dict[int, str]] = {
    "anthropic": {
        1: "claude-opus-4-5-20250918",
        2: "claude-sonnet-4-5-20250929",
        3: "claude-haiku-4-5-20251001",
    },
    "openai": {
        1: "gpt-5",
        2: "o3",
        3: "gpt-4o-mini",
    },
    "google": {
        1: "gemini/gemini-2.5-pro-preview-05-06",
        2: "gemini/gemini-2.5-flash-preview-04-17",
        3: "gemini/gemini-2.0-flash",
    },
    "xai": {
        1: "xai/grok-4",
        2: "xai/grok-3",
        3: "xai/grok-3-fast",
    },
}

PROVIDER_COLORS: dict[str, str] = {
    "anthropic": "#7C6BFF",
    "openai": "#4ADE80",
    "google": "#FACC15",
    "xai": "#F87171",
}

# Agent name pools for generated configs
_HOUSE_NAMES = ["House Alpha", "House Beta", "House Gamma", "House Delta"]
_AGENT_NAMES = [
    ["Apex", "Core", "Flux"],
    ["Volt", "Drift", "Haze"],
    ["Rift", "Veil", "Crux"],
    ["Zeal", "Keen", "Bolt"],
]

TIER_TEMPS: dict[int, float] = {1: 0.6, 2: 0.7, 3: 0.8}


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

def build_standard_config() -> GameConfig:
    """Series A: Load standard config from config.yaml."""
    return load_game_config()


def build_single_provider_config(provider: str) -> GameConfig:
    """
    Series B: All 12 agents from one provider.
    4 families of 3, each with tier 1/2/3. All same provider.
    """
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Unknown provider: {provider}. Must be one of {list(PROVIDER_MODELS.keys())}")

    models = PROVIDER_MODELS[provider]
    color = PROVIDER_COLORS[provider]

    families: list[FamilyConfig] = []
    for i in range(4):
        agents: list[AgentConfig] = []
        for tier in (1, 2, 3):
            agents.append(AgentConfig(
                name=_AGENT_NAMES[i][tier - 1],
                model=models[tier],
                tier=tier,
                temperature=TIER_TEMPS[tier],
            ))
        families.append(FamilyConfig(
            name=_HOUSE_NAMES[i],
            provider=provider,
            color=color,
            agents=agents,
        ))

    return GameConfig(
        families=families,
        series_type="single_provider",
    )


def build_shuffled_config() -> GameConfig:
    """
    Series C: Cross-provider families.
    Each family has agents from 3 different providers.
    Family 0: anthropic-T1, openai-T2, google-T3
    Family 1: openai-T1, google-T2, xai-T3
    Family 2: google-T1, xai-T2, anthropic-T3
    Family 3: xai-T1, anthropic-T2, openai-T3
    """
    providers = ["anthropic", "openai", "google", "xai"]

    families: list[FamilyConfig] = []
    for i in range(4):
        agents: list[AgentConfig] = []
        for tier in (1, 2, 3):
            p_idx = (i + tier - 1) % 4
            prov = providers[p_idx]
            agents.append(AgentConfig(
                name=_AGENT_NAMES[i][tier - 1],
                model=PROVIDER_MODELS[prov][tier],
                tier=tier,
                temperature=TIER_TEMPS[tier],
                provider=prov,
            ))
        # Family provider is labeled "mixed"
        families.append(FamilyConfig(
            name=_HOUSE_NAMES[i],
            provider="mixed",
            color=["#7C6BFF", "#4ADE80", "#FACC15", "#F87171"][i],
            agents=agents,
        ))

    return GameConfig(
        families=families,
        series_type="shuffled",
    )


def build_no_family_config() -> GameConfig:
    """
    Series D: No family channel.
    Standard agent/family setup but family discussions are disabled.
    """
    config = load_game_config()
    return GameConfig(
        families=config.families,
        no_family_discussion=True,
        series_type="no_family",
    )


def build_flat_hierarchy_config() -> GameConfig:
    """
    Series E: All Boss-tier.
    4 families, 3 agents each, all tier 1 using frontier models.
    """
    providers = ["anthropic", "openai", "google", "xai"]
    standard = load_game_config()

    families: list[FamilyConfig] = []
    for i, fam in enumerate(standard.families):
        prov = providers[i]
        boss_model = PROVIDER_MODELS[prov][1]
        agents: list[AgentConfig] = []
        for j, orig in enumerate(fam.agents):
            agents.append(AgentConfig(
                name=orig.name,
                model=boss_model,
                tier=1,
                temperature=0.6,
            ))
        families.append(FamilyConfig(
            name=fam.name,
            provider=fam.provider,
            color=fam.color,
            agents=agents,
        ))

    return GameConfig(
        families=families,
        series_type="flat_hierarchy",
    )


def build_flat_temperature_config() -> GameConfig:
    """
    Series F: Flat temperature.
    Standard families and tiers, but all agents at temperature 0.7.
    Controls for temperature as a confound variable.
    """
    standard = load_game_config()

    families: list[FamilyConfig] = []
    for fam in standard.families:
        agents: list[AgentConfig] = []
        for orig in fam.agents:
            agents.append(AgentConfig(
                name=orig.name,
                model=orig.model,
                tier=orig.tier,
                temperature=0.7,  # flat 0.7 across all tiers
            ))
        families.append(FamilyConfig(
            name=fam.name,
            provider=fam.provider,
            color=fam.color,
            agents=agents,
        ))

    return GameConfig(
        families=families,
        series_type="flat_temperature",
    )


CONFIG_BUILDERS: dict[str, object] = {
    "standard": build_standard_config,
    "single_provider": build_single_provider_config,
    "shuffled": build_shuffled_config,
    "no_family": build_no_family_config,
    "flat_hierarchy": build_flat_hierarchy_config,
    "flat_temperature": build_flat_temperature_config,
}


# ---------------------------------------------------------------------------
# Series runner
# ---------------------------------------------------------------------------

async def run_series(
    series_type: str,
    num_games: int = 15,
    provider: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run a series of games with the specified config type.
    Returns the aggregated series metrics.
    """
    from markov.orchestrator import run_game_llm

    # Generate config
    if series_type == "single_provider":
        if provider is None:
            raise ValueError("provider is required for single_provider series")
        config = build_single_provider_config(provider)
    elif series_type == "standard":
        config = build_standard_config()
    elif series_type == "shuffled":
        config = build_shuffled_config()
    elif series_type == "no_family":
        config = build_no_family_config()
    elif series_type == "flat_hierarchy":
        config = build_flat_hierarchy_config()
    elif series_type == "flat_temperature":
        config = build_flat_temperature_config()
    else:
        raise ValueError(f"Unknown series type: {series_type}")

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
        }, f, indent=2, default=str)

    # Run games
    series_metrics = SeriesMetrics()
    game_dirs: list[str] = []

    for game_num in range(num_games):
        logger.info("Starting game %d/%d for series %s", game_num + 1, num_games, series_id)
        if verbose:
            print(f"\n{'='*60}")
            print(f"SERIES {series_id} -- Game {game_num + 1}/{num_games}")
            print(f"{'='*60}")

        state, game_logger = await run_game_llm(config=config, verbose=verbose)

        # Save game output into series directory
        game_dir = game_logger.save(
            path=series_dir / f"game_{game_num + 1:03d}",
            agents=state.agents,
        )
        game_dirs.append(str(game_dir))

        # Accumulate metrics
        if game_logger.metrics:
            series_metrics.add_game(game_logger.metrics)

        logger.info(
            "Game %d/%d complete. Winner: %s. Rounds: %d",
            game_num + 1, num_games,
            state.winner.name if state.winner else "none",
            state.round_num,
        )

    # Compute and save aggregate
    aggregate = series_metrics.compute()
    aggregate["series_id"] = series_id
    aggregate["series_type"] = series_type
    aggregate["provider"] = provider

    with open(series_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    # Write summary report
    report = _build_series_report(series_id, series_type, num_games, aggregate)
    with open(series_dir / "report.md", "w") as f:
        f.write(report)

    logger.info("Series %s complete. Results saved to %s", series_id, series_dir)
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
    lines.append("")

    # Win rates
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

    # Provider metrics
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
