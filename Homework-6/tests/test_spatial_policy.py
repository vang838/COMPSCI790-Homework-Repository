from src.spatial_runtime import (
    condition_matches,
    evaluate_spatial_dsl,
    evaluate_uniform_strategy,
    plan_frame,
    select_mode,
)


def sample_policy() -> dict:
    return {
        "frame_grid": {
            "rows": 1,
            "cols": 2,
        },
        "latency_budget_ms": 80,
        "mode_order": ["skip", "fast", "balanced", "accurate"],
        "modes": {
            "skip": {
                "latency_ms": 0,
                "quality": 0.00,
            },
            "fast": {
                "latency_ms": 10,
                "quality": 0.60,
            },
            "balanced": {
                "latency_ms": 25,
                "quality": 0.80,
            },
            "accurate": {
                "latency_ms": 60,
                "quality": 0.92,
            },
        },
        "rules": [
            {
                "name": "important_block_use_accurate",
                "condition": "importance >= 0.80",
                "mode": "accurate",
            },
            {
                "name": "motion_block_use_balanced",
                "condition": "motion >= 0.70",
                "mode": "balanced",
            },
            {
                "name": "medium_block_use_balanced",
                "condition": "importance >= 0.50",
                "mode": "balanced",
            },
            {
                "name": "low_block_use_fast",
                "condition": "importance >= 0.20",
                "mode": "fast",
            },
            {
                "name": "unimportant_block_skip",
                "condition": "true",
                "mode": "skip",
            },
        ],
    }


def test_condition_matches_importance_threshold() -> None:
    block = {
        "importance": 0.85,
        "motion": 0.20,
    }

    assert condition_matches("importance >= 0.80", block) is True


def test_selects_accurate_for_important_block() -> None:
    policy = sample_policy()

    block = {
        "block_id": "r0c0",
        "importance": 0.91,
        "motion": 0.10,
    }

    mode, rule_name = select_mode(policy, block)

    assert mode == "accurate"
    assert rule_name == "important_block_use_accurate"


def test_selects_skip_for_unimportant_block() -> None:
    policy = sample_policy()

    block = {
        "block_id": "r0c0",
        "importance": 0.05,
        "motion": 0.05,
    }

    mode, rule_name = select_mode(policy, block)

    assert mode == "skip"
    assert rule_name == "unimportant_block_skip"


def test_budget_preserves_more_important_block() -> None:
    policy = sample_policy()

    frame = {
        "frame_id": 1,
        "blocks": [
            {
                "block_id": "r0c0",
                "row": 0,
                "col": 0,
                "importance": 0.95,
                "motion": 0.10,
            },
            {
                "block_id": "r0c1",
                "row": 0,
                "col": 1,
                "importance": 0.85,
                "motion": 0.10,
            },
        ],
    }

    plan = plan_frame(policy, frame)

    decisions_by_block = {
        decision["block_id"]: decision
        for decision in plan["decisions"]
    }

    assert plan["final_latency_ms"] <= policy["latency_budget_ms"]
    assert decisions_by_block["r0c0"]["final_mode"] == "accurate"
    assert decisions_by_block["r0c1"]["final_mode"] == "fast"
    assert decisions_by_block["r0c1"]["downgraded_by_budget"] is True


def test_spatial_dsl_evaluation_has_no_budget_violation() -> None:
    policy = sample_policy()

    frames = [
        {
            "frame_id": 1,
            "blocks": [
                {
                    "block_id": "r0c0",
                    "row": 0,
                    "col": 0,
                    "importance": 0.95,
                    "motion": 0.10,
                },
                {
                    "block_id": "r0c1",
                    "row": 0,
                    "col": 1,
                    "importance": 0.85,
                    "motion": 0.10,
                },
            ],
        }
    ]

    result = evaluate_spatial_dsl(policy, frames)

    assert result["avg_latency_ms_per_frame"] <= policy["latency_budget_ms"]
    assert result["budget_violations"] == 0


def test_uniform_accurate_violates_budget() -> None:
    policy = sample_policy()

    frames = [
        {
            "frame_id": 1,
            "blocks": [
                {
                    "block_id": "r0c0",
                    "row": 0,
                    "col": 0,
                    "importance": 0.95,
                    "motion": 0.10,
                },
                {
                    "block_id": "r0c1",
                    "row": 0,
                    "col": 1,
                    "importance": 0.85,
                    "motion": 0.10,
                },
            ],
        }
    ]

    result = evaluate_uniform_strategy(policy, frames, "accurate")

    assert result["budget_violations"] == 1