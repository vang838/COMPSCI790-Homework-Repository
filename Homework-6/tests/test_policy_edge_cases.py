import pytest

from src.run_policy import condition_matches, evaluate_strategy, select_mode


def sample_policy() -> dict:
    return {
        "modes": {
            "fast": {"latency_ms": 20, "quality": 0.70},
            "balanced": {"latency_ms": 45, "quality": 0.84},
            "accurate": {"latency_ms": 90, "quality": 0.92},
        },
        "rules": [
            {
                "name": "high_motion_use_accurate",
                "condition": "motion > 0.70",
                "mode": "accurate",
            },
            {
                "name": "queue_pressure_use_fast",
                "condition": "queue > 8",
                "mode": "fast",
            },
            {
                "name": "default_balanced",
                "condition": "true",
                "mode": "balanced",
            },
        ],
    }


@pytest.mark.parametrize(
    ("condition", "frame", "expected"),
    [
        ("true", {}, True),
        ("motion >= 0.70", {"motion": 0.70}, True),
        ("motion <= 0.70", {"motion": 0.70}, True),
        ("queue == 8", {"queue": 8}, True),
        ("queue < 8", {"queue": 7}, True),
        ("queue > 8", {"queue": 8}, False),
    ],
)
def test_condition_comparison_operators(condition: str, frame: dict, expected: bool) -> None:
    assert condition_matches(condition, frame) is expected


def test_invalid_condition_format_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid condition format"):
        condition_matches("motion >", {"motion": 0.50})


def test_unsupported_operator_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported operator"):
        condition_matches("motion != 0.50", {"motion": 0.50})


def test_select_mode_raises_when_no_rule_matches() -> None:
    policy = {
        "modes": sample_policy()["modes"],
        "rules": [
            {
                "name": "only_high_motion",
                "condition": "motion > 0.90",
                "mode": "accurate",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="No rule matched"):
        select_mode(policy, {"frame_id": 1, "motion": 0.10, "queue": 1})


def test_fixed_mode_strategy_ignores_adaptive_rules() -> None:
    policy = sample_policy()
    frames = [
        {"frame_id": 1, "motion": 0.95, "queue": 12},
        {"frame_id": 2, "motion": 0.10, "queue": 1},
    ]

    avg_latency, avg_quality = evaluate_strategy(
        policy=policy,
        frames=frames,
        fixed_mode="fast",
    )

    assert avg_latency == 20.0
    assert avg_quality == 0.70
