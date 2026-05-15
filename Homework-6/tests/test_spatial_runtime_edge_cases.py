import pytest

from src.spatial_runtime import (
    compute_frame_latency,
    compute_weighted_quality,
    condition_matches,
    enforce_latency_budget,
    evaluate_uniform_feasible_strategy,
    plan_frame,
)


def sample_policy(budget_ms: float = 70) -> dict:
    return {
        "frame_grid": {"rows": 1, "cols": 2},
        "latency_budget_ms": budget_ms,
        "mode_order": ["skip", "fast", "balanced", "accurate"],
        "modes": {
            "skip": {"latency_ms": 0, "quality": 0.00},
            "fast": {"latency_ms": 10, "quality": 0.60},
            "balanced": {"latency_ms": 25, "quality": 0.80},
            "accurate": {"latency_ms": 60, "quality": 0.92},
        },
        "rules": [
            {
                "name": "important_block_use_accurate",
                "condition": "importance >= 0.80",
                "mode": "accurate",
            },
            {
                "name": "motion_hotspot_use_balanced",
                "condition": "motion >= 0.70 and importance >= 0.40",
                "mode": "balanced",
            },
            {
                "name": "medium_block_use_fast",
                "condition": "importance >= 0.20",
                "mode": "fast",
            },
            {
                "name": "default_skip",
                "condition": "true",
                "mode": "skip",
            },
        ],
    }


def test_spatial_condition_supports_and_or() -> None:
    block = {"importance": 0.50, "motion": 0.75}

    assert condition_matches("motion >= 0.70 and importance >= 0.40", block) is True
    assert condition_matches("importance >= 0.90 or motion >= 0.70", block) is True
    assert condition_matches("importance >= 0.90 and motion >= 0.70", block) is False


def test_spatial_condition_invalid_format_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid condition format"):
        condition_matches("importance >=", {"importance": 0.5})


def test_weighted_quality_returns_zero_when_importance_sum_is_zero() -> None:
    policy = sample_policy()
    decisions = [
        {"importance": 0.0, "final_mode": "accurate"},
        {"importance": 0.0, "final_mode": "fast"},
    ]

    assert compute_weighted_quality(policy, decisions) == 0.0


def test_enforce_latency_budget_downgrades_lower_importance_block_first() -> None:
    policy = sample_policy(budget_ms=70)
    decisions = [
        {
            "block_id": "r0c0",
            "importance": 0.95,
            "final_mode": "accurate",
            "downgraded_by_budget": False,
        },
        {
            "block_id": "r0c1",
            "importance": 0.10,
            "final_mode": "accurate",
            "downgraded_by_budget": False,
        },
    ]

    final_decisions = enforce_latency_budget(policy, decisions)
    decisions_by_block = {d["block_id"]: d for d in final_decisions}

    assert compute_frame_latency(policy, final_decisions) <= policy["latency_budget_ms"]
    assert decisions_by_block["r0c0"]["final_mode"] == "accurate"
    assert decisions_by_block["r0c1"]["final_mode"] == "fast"
    assert decisions_by_block["r0c1"]["downgraded_by_budget"] is True


def test_uniform_feasible_chooses_skip_when_budget_is_too_small() -> None:
    policy = sample_policy(budget_ms=5)
    frame = {
        "frame_id": 1,
        "blocks": [
            {"block_id": "r0c0", "row": 0, "col": 0, "importance": 0.5, "motion": 0.5},
            {"block_id": "r0c1", "row": 0, "col": 1, "importance": 0.5, "motion": 0.5},
        ],
    }

    result = evaluate_uniform_feasible_strategy(policy, [frame])

    assert result["strategy"] == "Uniform feasible (skip)"
    assert result["avg_latency_ms_per_frame"] == 0
    assert result["budget_violations"] == 0


def test_plan_frame_records_budget_violation_when_no_downgrade_can_help() -> None:
    policy = sample_policy(budget_ms=-1)
    frame = {
        "frame_id": 1,
        "blocks": [
            {"block_id": "r0c0", "row": 0, "col": 0, "importance": 0.01, "motion": 0.01},
            {"block_id": "r0c1", "row": 0, "col": 1, "importance": 0.01, "motion": 0.01},
        ],
    }

    plan = plan_frame(policy, frame)

    assert plan["final_latency_ms"] == 0
    assert plan["budget_violation"] is True
