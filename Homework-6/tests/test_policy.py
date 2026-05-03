import pytest

from src.run_policy import condition_matches, select_mode, evaluate_strategy


def sample_policy() -> dict:
    return {
        "modes": {
            "fast": {
                "latency_ms": 20,
                "quality": 0.70,
            },
            "balanced": {
                "latency_ms": 45,
                "quality": 0.84,
            },
            "accurate": {
                "latency_ms": 90,
                "quality": 0.92,
            },
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


def test_condition_matches_high_motion() -> None:
    frame = {"frame_id": 1, "motion": 0.85, "queue": 2}

    assert condition_matches("motion > 0.70", frame) is True


def test_condition_does_not_match_low_motion() -> None:
    frame = {"frame_id": 1, "motion": 0.20, "queue": 2}

    assert condition_matches("motion > 0.70", frame) is False


def test_selects_accurate_for_high_motion() -> None:
    policy = sample_policy()
    frame = {"frame_id": 1, "motion": 0.85, "queue": 2}

    mode, rule_name = select_mode(policy, frame)

    assert mode == "accurate"
    assert rule_name == "high_motion_use_accurate"


def test_selects_fast_for_queue_pressure() -> None:
    policy = sample_policy()
    frame = {"frame_id": 1, "motion": 0.20, "queue": 10}

    mode, rule_name = select_mode(policy, frame)

    assert mode == "fast"
    assert rule_name == "queue_pressure_use_fast"


def test_selects_balanced_by_default() -> None:
    policy = sample_policy()
    frame = {"frame_id": 1, "motion": 0.20, "queue": 2}

    mode, rule_name = select_mode(policy, frame)

    assert mode == "balanced"
    assert rule_name == "default_balanced"


def test_rule_priority_uses_first_matching_rule() -> None:
    policy = sample_policy()
    frame = {"frame_id": 1, "motion": 0.90, "queue": 12}

    mode, rule_name = select_mode(policy, frame)

    assert mode == "accurate"
    assert rule_name == "high_motion_use_accurate"


def test_invalid_frame_field_raises_error() -> None:
    frame = {"frame_id": 1, "motion": 0.50, "queue": 2}

    with pytest.raises(ValueError):
        condition_matches("confidence > 0.50", frame)


def test_adaptive_strategy_metrics() -> None:
    policy = sample_policy()
    frames = [
        {"frame_id": 1, "motion": 0.15, "queue": 2},
        {"frame_id": 2, "motion": 0.82, "queue": 3},
        {"frame_id": 3, "motion": 0.30, "queue": 10},
        {"frame_id": 4, "motion": 0.74, "queue": 4},
        {"frame_id": 5, "motion": 0.22, "queue": 6},
    ]

    avg_latency, avg_quality = evaluate_strategy(
        policy=policy,
        frames=frames,
        fixed_mode=None,
    )

    assert avg_latency == 58.0
    assert round(avg_quality, 2) == 0.84