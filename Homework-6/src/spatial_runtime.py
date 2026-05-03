import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
POLICY_PATH = BASE_DIR / "policies" / "spatial_policy.json"


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Load the spatial adaptive inference policy."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _eval_single_condition(expression: str, block: dict[str, Any]) -> bool:
    """
    Evaluate one simple condition against a block.

    Supported examples:
    - importance >= 0.80
    - motion >= 0.70
    - importance < 0.20
    """
    parts = expression.strip().split()

    if len(parts) != 3:
        raise ValueError(f"Invalid condition format: {expression}")

    field, operator, raw_value = parts

    if field not in block:
        raise ValueError(f"Unknown block field in condition: {field}")

    left_value = float(block[field])
    right_value = float(raw_value)

    if operator == ">":
        return left_value > right_value
    if operator == ">=":
        return left_value >= right_value
    if operator == "<":
        return left_value < right_value
    if operator == "<=":
        return left_value <= right_value
    if operator == "==":
        return left_value == right_value

    raise ValueError(f"Unsupported operator in condition: {operator}")


def condition_matches(condition: str, block: dict[str, Any]) -> bool:
    """
    Evaluate a DSL condition against one image block.

    Supports:
    - true
    - single comparisons
    - simple and/or expressions
    """
    condition = condition.strip().lower()

    if condition == "true":
        return True

    if " or " in condition:
        return any(condition_matches(part, block) for part in condition.split(" or "))

    if " and " in condition:
        return all(condition_matches(part, block) for part in condition.split(" and "))

    return _eval_single_condition(condition, block)


def select_mode(policy: dict[str, Any], block: dict[str, Any]) -> tuple[str, str]:
    """
    Apply the DSL rules in order and return the selected mode and rule name.
    """
    for rule in policy["rules"]:
        if condition_matches(rule["condition"], block):
            return str(rule["mode"]), str(rule["name"])

    raise RuntimeError(f"No rule matched block {block.get('block_id', '<unknown>')}")


def mode_latency(policy: dict[str, Any], mode: str) -> float:
    return float(policy["modes"][mode]["latency_ms"])


def mode_quality(policy: dict[str, Any], mode: str) -> float:
    return float(policy["modes"][mode]["quality"])


def compute_frame_latency(policy: dict[str, Any], decisions: list[dict[str, Any]]) -> float:
    return sum(mode_latency(policy, str(decision["final_mode"])) for decision in decisions)


def compute_weighted_quality(policy: dict[str, Any], decisions: list[dict[str, Any]]) -> float:
    """
    Compute importance-weighted quality.

    This gives more weight to important blocks, which better matches the idea that
    quality matters more in regions likely to contain useful objects or motion.
    """
    total_importance = sum(float(decision["importance"]) for decision in decisions)

    if total_importance == 0:
        return 0.0

    weighted_total = 0.0

    for decision in decisions:
        importance = float(decision["importance"])
        quality = mode_quality(policy, str(decision["final_mode"]))
        weighted_total += importance * quality

    return weighted_total / total_importance


def make_initial_decisions(policy: dict[str, Any], frame: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Assign an initial mode to every block using the DSL rules.
    """
    decisions: list[dict[str, Any]] = []

    for block in frame["blocks"]:
        mode, rule_name = select_mode(policy, block)

        decisions.append(
            {
                "frame_id": frame["frame_id"],
                "block_id": block["block_id"],
                "row": block["row"],
                "col": block["col"],
                "importance": block["importance"],
                "motion": block["motion"],
                "initial_mode": mode,
                "final_mode": mode,
                "rule_applied": rule_name,
                "downgraded_by_budget": False,
            }
        )

    return decisions


def downgrade_least_important_block(
    policy: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> bool:
    """
    Downgrade the least important block by one mode level.

    Example:
    accurate -> balanced -> fast -> skip

    Returns True if a downgrade happened.
    Returns False if no further downgrade is possible.
    """
    mode_order = policy["mode_order"]
    mode_rank = {mode: index for index, mode in enumerate(mode_order)}

    candidates = [
        decision
        for decision in decisions
        if mode_rank[str(decision["final_mode"])] > 0
    ]

    if not candidates:
        return False

    candidate = min(
        candidates,
        key=lambda decision: (
            float(decision["importance"]),
            -mode_latency(policy, str(decision["final_mode"])),
        ),
    )

    current_mode = str(candidate["final_mode"])
    next_mode = mode_order[mode_rank[current_mode] - 1]

    candidate["final_mode"] = next_mode
    candidate["downgraded_by_budget"] = True

    return True


def enforce_latency_budget(
    policy: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Enforce the per-frame latency budget.

    If the initial plan is too expensive, the runtime repeatedly downgrades
    the least important blocks until the plan fits inside the budget.
    """
    budget = float(policy["latency_budget_ms"])

    while compute_frame_latency(policy, decisions) > budget:
        changed = downgrade_least_important_block(policy, decisions)

        if not changed:
            break

    return decisions


def plan_frame(policy: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    """
    Build a Remix-style execution plan for one frame.

    The plan contains one inference decision per image block.
    """
    decisions = make_initial_decisions(policy, frame)

    initial_latency = sum(
        mode_latency(policy, str(decision["initial_mode"]))
        for decision in decisions
    )

    decisions = enforce_latency_budget(policy, decisions)

    final_latency = compute_frame_latency(policy, decisions)
    weighted_quality = compute_weighted_quality(policy, decisions)
    budget = float(policy["latency_budget_ms"])

    return {
        "frame_id": frame["frame_id"],
        "initial_latency_ms": initial_latency,
        "final_latency_ms": final_latency,
        "weighted_quality": weighted_quality,
        "budget_ms": budget,
        "budget_violation": final_latency > budget,
        "decisions": decisions,
    }


def evaluate_spatial_dsl(policy: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Evaluate the spatial adaptive DSL over many frames.
    """
    plans = [plan_frame(policy, frame) for frame in frames]

    avg_latency = sum(float(plan["final_latency_ms"]) for plan in plans) / len(plans)
    avg_quality = sum(float(plan["weighted_quality"]) for plan in plans) / len(plans)
    budget_violations = sum(1 for plan in plans if bool(plan["budget_violation"]))

    return {
        "strategy": "Spatial DSL",
        "avg_latency_ms_per_frame": avg_latency,
        "avg_weighted_quality": avg_quality,
        "budget_violations": budget_violations,
        "plans": plans,
    }


def evaluate_uniform_strategy(
    policy: dict[str, Any],
    frames: list[dict[str, Any]],
    fixed_mode: str,
) -> dict[str, Any]:
    """
    Evaluate a baseline where every block uses the same mode.

    Examples:
    - Uniform fast
    - Uniform balanced
    - Uniform accurate
    """
    budget = float(policy["latency_budget_ms"])

    frame_latencies: list[float] = []
    frame_qualities: list[float] = []
    budget_violations = 0

    for frame in frames:
        decisions = []

        for block in frame["blocks"]:
            decisions.append(
                {
                    "importance": block["importance"],
                    "final_mode": fixed_mode,
                }
            )

        latency = len(frame["blocks"]) * mode_latency(policy, fixed_mode)
        quality = compute_weighted_quality(policy, decisions)

        frame_latencies.append(latency)
        frame_qualities.append(quality)

        if latency > budget:
            budget_violations += 1

    return {
        "strategy": f"Uniform {fixed_mode}",
        "avg_latency_ms_per_frame": sum(frame_latencies) / len(frame_latencies),
        "avg_weighted_quality": sum(frame_qualities) / len(frame_qualities),
        "budget_violations": budget_violations,
    }

def clone_block_decision(block: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "importance": block["importance"],
        "final_mode": mode,
    }


def evaluate_uniform_feasible_strategy(
    policy: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Select the highest-quality uniform mode that fits the latency budget.

    This is a fairer uniform baseline than uniform balanced or uniform accurate,
    because it only chooses a uniform strategy that satisfies the budget.
    """
    budget = float(policy["latency_budget_ms"])
    mode_order = list(policy["mode_order"])

    feasible_modes: list[str] = []

    for mode in mode_order:
        per_frame_latency = 9 * mode_latency(policy, mode)

        if per_frame_latency <= budget:
            feasible_modes.append(mode)

    if not feasible_modes:
        selected_mode = "skip"
    else:
        selected_mode = feasible_modes[-1]

    result = evaluate_uniform_strategy(policy, frames, selected_mode)
    result["strategy"] = f"Uniform feasible ({selected_mode})"

    return result


def evaluate_topk_accurate_strategy(
    policy: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Budget-aware spatial baseline.

    Starts every block in fast mode, then upgrades the most important blocks
    to accurate mode while staying within the budget.
    """
    budget = float(policy["latency_budget_ms"])

    frame_latencies: list[float] = []
    frame_qualities: list[float] = []
    budget_violations = 0

    for frame in frames:
        blocks = list(frame["blocks"])

        decisions = [
            clone_block_decision(block, "fast")
            for block in blocks
        ]

        current_latency = len(blocks) * mode_latency(policy, "fast")
        upgrade_cost = mode_latency(policy, "accurate") - mode_latency(policy, "fast")

        sorted_indices = sorted(
            range(len(blocks)),
            key=lambda index: float(blocks[index]["importance"]),
            reverse=True,
        )

        for index in sorted_indices:
            if current_latency + upgrade_cost <= budget:
                decisions[index]["final_mode"] = "accurate"
                current_latency += upgrade_cost

        quality = compute_weighted_quality(policy, decisions)

        frame_latencies.append(current_latency)
        frame_qualities.append(quality)

        if current_latency > budget:
            budget_violations += 1

    return {
        "strategy": "Top-K accurate",
        "avg_latency_ms_per_frame": sum(frame_latencies) / len(frame_latencies),
        "avg_weighted_quality": sum(frame_qualities) / len(frame_qualities),
        "budget_violations": budget_violations,
    }


def evaluate_greedy_budget_strategy(
    policy: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Strong budget-aware baseline.

    Starts all blocks at skip, then repeatedly upgrades the block that gives
    the highest importance-weighted benefit per additional latency cost.
    """
    budget = float(policy["latency_budget_ms"])
    mode_order = list(policy["mode_order"])
    mode_rank = {mode: index for index, mode in enumerate(mode_order)}

    frame_latencies: list[float] = []
    frame_qualities: list[float] = []
    budget_violations = 0

    for frame in frames:
        blocks = list(frame["blocks"])

        decisions = [
            clone_block_decision(block, "skip")
            for block in blocks
        ]

        current_latency = 0.0

        while True:
            best_choice: tuple[float, int, str, float] | None = None

            for index, block in enumerate(blocks):
                current_mode = str(decisions[index]["final_mode"])
                current_rank = mode_rank[current_mode]

                if current_rank == len(mode_order) - 1:
                    continue

                next_mode = mode_order[current_rank + 1]

                additional_latency = (
                    mode_latency(policy, next_mode)
                    - mode_latency(policy, current_mode)
                )

                if additional_latency <= 0:
                    continue

                if current_latency + additional_latency > budget:
                    continue

                quality_gain = (
                    mode_quality(policy, next_mode)
                    - mode_quality(policy, current_mode)
                )

                importance = float(block["importance"])
                score = importance * quality_gain / additional_latency

                if best_choice is None or score > best_choice[0]:
                    best_choice = (score, index, next_mode, additional_latency)

            if best_choice is None:
                break

            _, index, next_mode, additional_latency = best_choice
            decisions[index]["final_mode"] = next_mode
            current_latency += additional_latency

        quality = compute_weighted_quality(policy, decisions)

        frame_latencies.append(current_latency)
        frame_qualities.append(quality)

        if current_latency > budget:
            budget_violations += 1

    return {
        "strategy": "Greedy budget",
        "avg_latency_ms_per_frame": sum(frame_latencies) / len(frame_latencies),
        "avg_weighted_quality": sum(frame_qualities) / len(frame_qualities),
        "budget_violations": budget_violations,
    }


def evaluate_random_budget_strategy(
    policy: dict[str, Any],
    frames: list[dict[str, Any]],
    seed: int = 790,
) -> dict[str, Any]:
    """
    Random budget-aware baseline.

    Randomly upgrades blocks while staying under budget.
    This controls for the possibility that any non-uniform allocation works.
    """
    import random

    random.seed(seed)

    budget = float(policy["latency_budget_ms"])
    mode_order = list(policy["mode_order"])
    mode_rank = {mode: index for index, mode in enumerate(mode_order)}

    frame_latencies: list[float] = []
    frame_qualities: list[float] = []
    budget_violations = 0

    for frame in frames:
        blocks = list(frame["blocks"])

        decisions = [
            clone_block_decision(block, "skip")
            for block in blocks
        ]

        current_latency = 0.0
        block_indices = list(range(len(blocks)))
        random.shuffle(block_indices)

        changed = True

        while changed:
            changed = False
            random.shuffle(block_indices)

            for index in block_indices:
                current_mode = str(decisions[index]["final_mode"])
                current_rank = mode_rank[current_mode]

                if current_rank == len(mode_order) - 1:
                    continue

                next_mode = mode_order[current_rank + 1]

                additional_latency = (
                    mode_latency(policy, next_mode)
                    - mode_latency(policy, current_mode)
                )

                if current_latency + additional_latency <= budget:
                    decisions[index]["final_mode"] = next_mode
                    current_latency += additional_latency
                    changed = True

        quality = compute_weighted_quality(policy, decisions)

        frame_latencies.append(current_latency)
        frame_qualities.append(quality)

        if current_latency > budget:
            budget_violations += 1

    return {
        "strategy": "Random budgeted",
        "avg_latency_ms_per_frame": sum(frame_latencies) / len(frame_latencies),
        "avg_weighted_quality": sum(frame_qualities) / len(frame_qualities),
        "budget_violations": budget_violations,
    }