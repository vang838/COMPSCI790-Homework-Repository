import csv
import random
from pathlib import Path
from typing import Any

from src.run_policy import POLICY_PATH, evaluate_strategy, load_policy, select_mode

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "experiment"
FRAME_RESULTS_PATH = OUTPUT_DIR / "frame_decisions.csv"
SUMMARY_RESULTS_PATH = OUTPUT_DIR / "experiment_summary.csv"


def generate_frames(frame_count: int, seed: int = 790) -> list[dict[str, Any]]:
    random.seed(seed)

    frames: list[dict[str, Any]] = []

    for frame_id in range(1, frame_count + 1):
        frame = {
            "frame_id": frame_id,
            "motion": round(random.uniform(0.0, 1.0), 2),
            "queue": random.randint(0, 12),
        }
        frames.append(frame)

    return frames


def write_frame_decisions(policy: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with FRAME_RESULTS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "frame_id",
                "motion",
                "queue",
                "selected_mode",
                "rule_applied",
                "latency_ms",
                "quality",
            ]
        )

        for frame in frames:
            mode, rule_name = select_mode(policy, frame)
            mode_info = policy["modes"][mode]

            writer.writerow(
                [
                    frame["frame_id"],
                    frame["motion"],
                    frame["queue"],
                    mode,
                    rule_name,
                    mode_info["latency_ms"],
                    mode_info["quality"],
                ]
            )


def write_summary(policy: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    strategies = [
        ("Fixed fast", "fast"),
        ("Fixed balanced", "balanced"),
        ("Fixed accurate", "accurate"),
        ("Adaptive DSL", None),
    ]

    with SUMMARY_RESULTS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["strategy", "avg_latency_ms_per_frame", "avg_quality"])

        for strategy_name, fixed_mode in strategies:
            avg_latency, avg_quality = evaluate_strategy(
                policy=policy,
                frames=frames,
                fixed_mode=fixed_mode,
            )

            writer.writerow(
                [
                    strategy_name,
                    f"{avg_latency:.2f}",
                    f"{avg_quality:.2f}",
                ]
            )
def print_mode_distribution(policy: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    counts = {
        "fast": 0,
        "balanced": 0,
        "accurate": 0,
    }

    for frame in frames:
        mode, _ = select_mode(policy, frame)
        counts[mode] += 1

    print()
    print("Mode Distribution")
    print("-" * 30)

    for mode, count in counts.items():
        percentage = count / len(frames) * 100
        print(f"{mode:<10}: {count:>3} frames ({percentage:.1f}%)")


def main() -> None:
    policy = load_policy(POLICY_PATH)
    frames = generate_frames(frame_count=100)

    write_frame_decisions(policy, frames)
    write_summary(policy, frames)
    print_mode_distribution(policy, frames)
    
    print("Synthetic experiment complete.")
    print(f"Frames evaluated: {len(frames)}")
    print(f"Frame decisions written to: {FRAME_RESULTS_PATH}")
    print(f"Summary written to: {SUMMARY_RESULTS_PATH}")


if __name__ == "__main__":
    main()