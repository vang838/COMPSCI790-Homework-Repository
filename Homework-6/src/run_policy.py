import argparse
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = BASE_DIR / "policies" / "policy.json"


def load_policy(path: Path) -> dict[str, Any]:
    """Load the adaptive inference policy from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def condition_matches(condition: str, frame: dict[str, Any]) -> bool:
    """
    Evaluate a simple DSL condition against one frame.

    Supported examples:
    - "motion > 0.70"
    - "queue > 8"
    - "true"
    """
    condition = condition.strip().lower()

    if condition == "true":
        return True

    parts = condition.split()

    if len(parts) != 3:
        raise ValueError(f"Invalid condition format: {condition}")

    field, operator, raw_value = parts

    if field not in frame:
        raise ValueError(f"Unknown frame field in condition: {field}")

    left_value = float(frame[field])
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


def select_mode(policy: dict[str, Any], frame: dict[str, Any]) -> tuple[str, str]:
    """
    Select an inference mode by applying the policy rules in order.
    Returns the selected mode and the rule name.
    """
    for rule in policy["rules"]:
        if condition_matches(rule["condition"], frame):
            return rule["mode"], rule["name"]

    raise RuntimeError("No rule matched this frame.")


def evaluate_strategy(
    policy: dict[str, Any],
    frames: list[dict[str, Any]],
    fixed_mode: str | None = None,
) -> tuple[float, float]:
    """Evaluate either a fixed mode or the adaptive DSL policy."""
    total_latency = 0.0
    total_quality = 0.0

    for frame in frames:
        if fixed_mode is None:
            mode, _ = select_mode(policy, frame)
        else:
            mode = fixed_mode

        mode_info = policy["modes"][mode]
        total_latency += float(mode_info["latency_ms"])
        total_quality += float(mode_info["quality"])

    frame_count = len(frames)
    avg_latency = total_latency / frame_count
    avg_quality = total_quality / frame_count

    return avg_latency, avg_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the JSON policy interpreter on sample frame metadata."
    )

    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Path to the JSON policy file. Default: policies/policy.json",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = args.policy

    if not policy_path.is_absolute():
        policy_path = BASE_DIR / policy_path

    policy = load_policy(policy_path)

    frames: list[dict[str, Any]] = [
        {"frame_id": 1, "motion": 0.15, "queue": 2},
        {"frame_id": 2, "motion": 0.82, "queue": 3},
        {"frame_id": 3, "motion": 0.30, "queue": 10},
        {"frame_id": 4, "motion": 0.74, "queue": 4},
        {"frame_id": 5, "motion": 0.22, "queue": 6},
    ]

    total_latency = 0.0
    total_quality = 0.0

    print(f"Policy file: {policy_path}")
    print()
    print("Frame | Motion | Queue | Selected Mode | Rule Applied             | Latency(ms) | Quality")
    print("-" * 91)

    for frame in frames:
        mode, rule_name = select_mode(policy, frame)
        mode_info = policy["modes"][mode]

        latency = float(mode_info["latency_ms"])
        quality = float(mode_info["quality"])

        total_latency += latency
        total_quality += quality

        print(
            f"{frame['frame_id']:>5} | "
            f"{frame['motion']:.2f}   | "
            f"{frame['queue']:>5} | "
            f"{mode:<13} | "
            f"{rule_name:<25} | "
            f"{latency:>10.0f} | "
            f"{quality:.2f}"
        )

    frame_count = len(frames)
    avg_latency = total_latency / frame_count
    avg_quality = total_quality / frame_count

    print()
    print("Summary")
    print("-" * 30)
    print(f"Average latency: {avg_latency:.2f} ms/frame")
    print(f"Average quality score: {avg_quality:.2f}")

    print()
    print("Baseline Comparison")
    print("-" * 65)
    print("Strategy       | Avg Latency(ms/frame) | Avg Quality")
    print("-" * 65)

    strategies = [
        ("Fixed fast", "fast"),
        ("Fixed balanced", "balanced"),
        ("Fixed accurate", "accurate"),
        ("Adaptive DSL", None),
    ]

    for strategy_name, fixed_mode in strategies:
        avg_latency, avg_quality = evaluate_strategy(
            policy=policy,
            frames=frames,
            fixed_mode=fixed_mode,
        )

        print(
            f"{strategy_name:<14} | "
            f"{avg_latency:>21.2f} | "
            f"{avg_quality:.2f}"
        )


if __name__ == "__main__":
    main()