from src.spatial_runtime import (
    evaluate_greedy_budget_strategy,
    evaluate_random_budget_strategy,
    evaluate_topk_accurate_strategy,
    evaluate_uniform_feasible_strategy,
)


def sample_policy() -> dict:
    return {
        "frame_grid": {
            "rows": 1,
            "cols": 3,
        },
        "latency_budget_ms": 80,
        "mode_order": ["skip", "fast", "balanced", "accurate"],
        "modes": {
            "skip": {"latency_ms": 0, "quality": 0.00},
            "fast": {"latency_ms": 10, "quality": 0.60},
            "balanced": {"latency_ms": 25, "quality": 0.80},
            "accurate": {"latency_ms": 60, "quality": 0.92},
        },
        "rules": [],
    }


def sample_frames() -> list[dict]:
    return [
        {
            "frame_id": 1,
            "blocks": [
                {
                    "block_id": "r0c0",
                    "row": 0,
                    "col": 0,
                    "importance": 0.95,
                    "motion": 0.80,
                },
                {
                    "block_id": "r0c1",
                    "row": 0,
                    "col": 1,
                    "importance": 0.50,
                    "motion": 0.40,
                },
                {
                    "block_id": "r0c2",
                    "row": 0,
                    "col": 2,
                    "importance": 0.10,
                    "motion": 0.10,
                },
            ],
        }
    ]


def test_uniform_feasible_stays_under_budget() -> None:
    policy = sample_policy()
    result = evaluate_uniform_feasible_strategy(policy, sample_frames())

    assert result["budget_violations"] == 0


def test_topk_accurate_stays_under_budget() -> None:
    policy = sample_policy()
    result = evaluate_topk_accurate_strategy(policy, sample_frames())

    assert result["budget_violations"] == 0
    assert result["avg_latency_ms_per_frame"] <= policy["latency_budget_ms"]


def test_greedy_budget_stays_under_budget() -> None:
    policy = sample_policy()
    result = evaluate_greedy_budget_strategy(policy, sample_frames())

    assert result["budget_violations"] == 0
    assert result["avg_latency_ms_per_frame"] <= policy["latency_budget_ms"]


def test_random_budgeted_stays_under_budget() -> None:
    policy = sample_policy()
    result = evaluate_random_budget_strategy(policy, sample_frames(), seed=790)

    assert result["budget_violations"] == 0
    assert result["avg_latency_ms_per_frame"] <= policy["latency_budget_ms"]