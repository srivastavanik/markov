"""Tests for series config generators, attribution, and no_family_discussion flag."""
import pytest

from markov.attribution import attribute_behavior, build_attribution_report
from markov.config import GameConfig
from markov.series import (
    build_flat_hierarchy_config,
    build_flat_temperature_config,
    build_no_family_config,
    build_shuffled_config,
    build_single_provider_config,
    build_standard_config,
)


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

class TestStandardConfig:
    def test_loads_4_families(self):
        config = build_standard_config()
        assert len(config.families) == 4

    def test_12_agents(self):
        config = build_standard_config()
        total = sum(len(f.agents) for f in config.families)
        assert total == 12


class TestSingleProviderConfig:
    def test_all_anthropic(self):
        config = build_single_provider_config("anthropic")
        assert len(config.families) == 4
        for fam in config.families:
            assert fam.provider == "anthropic"
            for agent in fam.agents:
                assert "claude" in agent.model.lower() or "anthropic" in agent.provider.lower()

    def test_all_openai(self):
        config = build_single_provider_config("openai")
        for fam in config.families:
            assert fam.provider == "openai"

    def test_all_google(self):
        config = build_single_provider_config("google")
        for fam in config.families:
            assert fam.provider == "google"
            for agent in fam.agents:
                assert "gemini" in agent.model.lower()

    def test_all_xai(self):
        config = build_single_provider_config("xai")
        for fam in config.families:
            assert fam.provider == "xai"
            for agent in fam.agents:
                assert "grok" in agent.model.lower()

    def test_has_all_3_tiers(self):
        config = build_single_provider_config("anthropic")
        for fam in config.families:
            tiers = {a.tier for a in fam.agents}
            assert tiers == {1, 2, 3}

    def test_12_agents(self):
        config = build_single_provider_config("anthropic")
        total = sum(len(f.agents) for f in config.families)
        assert total == 12

    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            build_single_provider_config("invalid")

    def test_series_type_set(self):
        config = build_single_provider_config("anthropic")
        assert config.series_type == "single_provider"


class TestShuffledConfig:
    def test_4_families(self):
        config = build_shuffled_config()
        assert len(config.families) == 4

    def test_cross_provider(self):
        """Each family should have agents from different providers."""
        config = build_shuffled_config()
        for fam in config.families:
            providers = {a.provider for a in fam.agents}
            # Each family has agents from at least 2 different providers
            assert len(providers) >= 2, f"Family {fam.name} has only {providers}"

    def test_all_tiers_present(self):
        config = build_shuffled_config()
        for fam in config.families:
            tiers = {a.tier for a in fam.agents}
            assert tiers == {1, 2, 3}

    def test_series_type_set(self):
        config = build_shuffled_config()
        assert config.series_type == "shuffled"


class TestNoFamilyConfig:
    def test_family_discussion_disabled(self):
        config = build_no_family_config()
        assert config.no_family_discussion is True

    def test_still_has_4_families(self):
        config = build_no_family_config()
        assert len(config.families) == 4

    def test_series_type_set(self):
        config = build_no_family_config()
        assert config.series_type == "no_family"


class TestFlatHierarchyConfig:
    def test_all_tier_1(self):
        config = build_flat_hierarchy_config()
        for fam in config.families:
            for agent in fam.agents:
                assert agent.tier == 1, f"Agent {agent.name} has tier {agent.tier}, expected 1"

    def test_uses_boss_models(self):
        config = build_flat_hierarchy_config()
        # All agents in a family should use the same (boss) model
        for fam in config.families:
            models = {a.model for a in fam.agents}
            assert len(models) == 1, f"Family {fam.name} has mixed models: {models}"

    def test_series_type_set(self):
        config = build_flat_hierarchy_config()
        assert config.series_type == "flat_hierarchy"


class TestFlatTemperatureConfig:
    def test_all_temperature_0_7(self):
        config = build_flat_temperature_config()
        for fam in config.families:
            for agent in fam.agents:
                assert agent.temperature == 0.7, f"Agent {agent.name} has temp {agent.temperature}, expected 0.7"

    def test_preserves_tiers(self):
        config = build_flat_temperature_config()
        for fam in config.families:
            tiers = {a.tier for a in fam.agents}
            assert tiers == {1, 2, 3}

    def test_series_type_set(self):
        config = build_flat_temperature_config()
        assert config.series_type == "flat_temperature"


class TestNoFamilyDiscussionFlag:
    def test_default_is_false(self):
        config = build_standard_config()
        assert config.no_family_discussion is False

    def test_can_be_set_true(self):
        config = build_no_family_config()
        assert config.no_family_discussion is True


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

class TestAttribution:
    def _mock_series(self, present_series: list[str], metric: str = "avg_deception_delta", value: float = 0.5) -> dict[str, dict]:
        all_types = ["standard", "single_provider", "shuffled", "no_family", "flat_hierarchy"]
        results = {}
        for st in all_types:
            if st in present_series:
                results[st] = {"per_provider": {"test": {metric: value}}}
            else:
                results[st] = {"per_provider": {"test": {metric: 0.0}}}
        return results

    def test_universal(self):
        series = self._mock_series(
            ["standard", "single_provider", "shuffled", "no_family", "flat_hierarchy"]
        )
        result = attribute_behavior("avg_deception_delta", 0.3, series)
        assert result["attribution"] == "universal"

    def test_provider_intrinsic(self):
        series = self._mock_series(["standard", "single_provider"])
        result = attribute_behavior("avg_deception_delta", 0.3, series)
        assert result["attribution"] == "provider_intrinsic"

    def test_family_conditioned(self):
        series = self._mock_series(["standard", "single_provider", "shuffled", "flat_hierarchy"])
        result = attribute_behavior("avg_deception_delta", 0.3, series)
        assert result["attribution"] == "family_conditioned"

    def test_tier_driven(self):
        series = self._mock_series(["standard", "single_provider", "shuffled", "no_family"])
        result = attribute_behavior("avg_deception_delta", 0.3, series)
        assert result["attribution"] == "tier_driven"


class TestAttributionReport:
    def test_generates_markdown(self):
        all_series = {
            "standard": {"per_provider": {"anthropic": {"avg_deception_delta": 0.4, "avg_malice_rate": 0.2}}, "win_rate_by_provider": {"anthropic": 0.5}},
            "shuffled": {"per_provider": {"anthropic": {"avg_deception_delta": 0.3, "avg_malice_rate": 0.15}}, "win_rate_by_provider": {"anthropic": 0.4}},
        }
        report = build_attribution_report(all_series)
        assert "# Attribution Analysis Report" in report
        assert "Provider Comparison" in report
