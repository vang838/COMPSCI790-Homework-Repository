import csv
import random
from pathlib import Path
from typing import Any

from src.spatial_runtime import (
    POLICY_PATH,
    evaluate_greedy_budget_strategy,
    evaluate_random_budget_strategy,
    evaluate_spatial_dsl,
    evaluate_topk_accurate_strategy,
    evaluate_uniform_feasible_strategy,
    evaluate_uniform_strategy,
    load_policy,
)


RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "spatial"

FRAME_DECISIONS_PATH = OUTPUT_DIR / "spatial_frame_decisions.csv"
SUMMARY_PATH = OUTPUT_DIR / "spatial_experiment_summary.csv"
MODE_DISTRIBUTION_PATH = OUTPUT_DIR / "spatial_mode_distribution.csv"
MULTI_SEED_SUMMARY_PATH = OUTPUT_DIR / "spatial_multiseed_summary.csv"
BUDGET_SWEEP_PATH = OUTPUT_DIR / "spatial_budget_sweep.csv"


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def generate_frames(
    frame_count: int,
    rows: int,
    cols: int,
    seed: int = 790,
) -> list[dict[str, Any]]:
    """
    Generate synthetic frames with spatial hotspots.

    Each frame has one hotspot region. Blocks closer to the hotspot receive
    higher importance and motion scores. This simulates the idea that only
    certain parts of the frame deserve expensive inference.
    """
    random.seed(seed)

    frames: list[dict[str, Any]] = []

    for frame_id in range(1, frame_count + 1):
        hot_row = random.randrange(rows)
        hot_col = random.randrange(cols)

        blocks: list[dict[str, Any]] = []

        for row in range(rows):
            for col in range(cols):
                distance = abs(row - hot_row) + abs(col - hot_col)

                hotspot_boost = max(0.0, 0.75 - 0.25 * distance)

                base_importance = random.uniform(0.00, 0.30)
                importance = round(clamp(base_importance + hotspot_boost), 2)

                base_motion = random.uniform(0.00, 0.35)
                motion = round(clamp(base_motion + hotspot_boost * 0.70), 2)

                blocks.append(
                    {
                        "block_id": f"r{row}c{col}",
                        "row": row,
                        "col": col,
                        "importance": importance,
                        "motion": motion,
                    }
                )

        frames.append(
            {
                "frame_id": frame_id,
                "hotspot": f"r{hot_row}c{hot_col}",
                "blocks": blocks,
            }
        )

    return frames


def write_frame_decisions(spatial_result: dict[str, Any]) -> None:
    """
    Save every block-level decision made by the spatial DSL.
    """
    with FRAME_DECISIONS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "frame_id",
                "block_id",
                "row",
                "col",
                "importance",
                "motion",
                "initial_mode",
                "final_mode",
                "rule_applied",
                "downgraded_by_budget",
                "final_latency_ms",
            ]
        )

        for plan in spatial_result["plans"]:
            for decision in plan["decisions"]:
                writer.writerow(
                    [
                        decision["frame_id"],
                        decision["block_id"],
                        decision["row"],
                        decision["col"],
                        decision["importance"],
                        decision["motion"],
                        decision["initial_mode"],
                        decision["final_mode"],
                        decision["rule_applied"],
                        decision["downgraded_by_budget"],
                        plan["final_latency_ms"],
                    ]
                )


def write_summary(results: list[dict[str, Any]], budget_ms: float) -> None:
    """
    Save baseline and spatial DSL comparison results.
    """
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "strategy",
                "avg_latency_ms_per_frame",
                "avg_weighted_quality",
                "latency_budget_ms",
                "budget_violations",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["strategy"],
                    f"{float(result['avg_latency_ms_per_frame']):.2f}",
                    f"{float(result['avg_weighted_quality']):.2f}",
                    f"{budget_ms:.2f}",
                    result["budget_violations"],
                ]
            )
            


def write_mode_distribution(spatial_result: dict[str, Any]) -> None:
    """
    Save how often each mode was selected across all blocks.
    """
    counts: dict[str, int] = {}

    for plan in spatial_result["plans"]:
        for decision in plan["decisions"]:
            mode = str(decision["final_mode"])
            counts[mode] = counts.get(mode, 0) + 1

    total_blocks = sum(counts.values())

    with MODE_DISTRIBUTION_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["mode", "block_count", "percentage"])

        for mode in sorted(counts):
            percentage = counts[mode] / total_blocks * 100
            writer.writerow([mode, counts[mode], f"{percentage:.2f}"])


def print_summary(results: list[dict[str, Any]], budget_ms: float) -> None:
    print("Spatial Remix-style experiment complete.")
    print(f"Latency budget: {budget_ms:.2f} ms/frame")
    print()
    print("Strategy           | Avg Latency | Avg Weighted Quality | Budget Violations")
    print("-" * 75)

    for result in results:
        print(
            f"{result['strategy']:<18} | "
            f"{float(result['avg_latency_ms_per_frame']):>11.2f} | "
            f"{float(result['avg_weighted_quality']):>20.2f} | "
            f"{int(result['budget_violations']):>17}"
        )

    print()
    print(f"Frame decisions written to: {FRAME_DECISIONS_PATH}")
    print(f"Summary written to: {SUMMARY_PATH}")
    print(f"Mode distribution written to: {MODE_DISTRIBUTION_PATH}")
    print(f"Multi-seed summary written to: {MULTI_SEED_SUMMARY_PATH}")
    print(f"Budget sweep written to: {BUDGET_SWEEP_PATH}")

def write_multiseed_summary(policy: dict[str, Any]) -> None:
    seeds = [100, 250, 500, 790, 1000]
    budget_ms = float(policy["latency_budget_ms"])

    rows = int(policy["frame_grid"]["rows"])
    cols = int(policy["frame_grid"]["cols"])

    with MULTI_SEED_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "seed",
                "strategy",
                "avg_latency_ms_per_frame",
                "avg_weighted_quality",
                "latency_budget_ms",
                "budget_violations",
            ]
        )

        for seed in seeds:
            frames = generate_frames(
                frame_count=100,
                rows=rows,
                cols=cols,
                seed=seed,
            )

            spatial_result = evaluate_spatial_dsl(policy, frames)

            results = [
                evaluate_uniform_feasible_strategy(policy, frames),
                evaluate_topk_accurate_strategy(policy, frames),
                evaluate_greedy_budget_strategy(policy, frames),
                evaluate_random_budget_strategy(policy, frames, seed=seed),
                spatial_result,
            ]

            for result in results:
                writer.writerow(
                    [
                        seed,
                        result["strategy"],
                        f"{float(result['avg_latency_ms_per_frame']):.2f}",
                        f"{float(result['avg_weighted_quality']):.2f}",
                        f"{budget_ms:.2f}",
                        result["budget_violations"],
                    ]
                )

def write_budget_sweep(policy: dict[str, Any]) -> None:
    budgets = [100, 130, 160, 200]
    seeds = [100, 250, 500, 790, 1000]

    rows = int(policy["frame_grid"]["rows"])
    cols = int(policy["frame_grid"]["cols"])

    with BUDGET_SWEEP_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "budget_ms",
                "seed",
                "strategy",
                "avg_latency_ms_per_frame",
                "avg_weighted_quality",
                "budget_violations",
            ]
        )

        for budget in budgets:
            policy_for_budget = dict(policy)
            policy_for_budget["latency_budget_ms"] = budget

            for seed in seeds:
                frames = generate_frames(
                    frame_count=100,
                    rows=rows,
                    cols=cols,
                    seed=seed,
                )

                spatial_result = evaluate_spatial_dsl(policy_for_budget, frames)

                results = [
                    evaluate_uniform_feasible_strategy(policy_for_budget, frames),
                    evaluate_topk_accurate_strategy(policy_for_budget, frames),
                    evaluate_greedy_budget_strategy(policy_for_budget, frames),
                    evaluate_random_budget_strategy(policy_for_budget, frames, seed=seed),
                    spatial_result,
                ]

                for result in results:
                    writer.writerow(
                        [
                            budget,
                            seed,
                            result["strategy"],
                            f"{float(result['avg_latency_ms_per_frame']):.2f}",
                            f"{float(result['avg_weighted_quality']):.2f}",
                            result["budget_violations"],
                        ]
                    )


def main() -> None:
    policy = load_policy(POLICY_PATH)

    rows = int(policy["frame_grid"]["rows"])
    cols = int(policy["frame_grid"]["cols"])
    budget_ms = float(policy["latency_budget_ms"])

    frames = generate_frames(frame_count=100, rows=rows, cols=cols)

    spatial_result = evaluate_spatial_dsl(policy, frames)

    results = [
        evaluate_uniform_strategy(policy, frames, "fast"),
        evaluate_uniform_strategy(policy, frames, "balanced"),
        evaluate_uniform_strategy(policy, frames, "accurate"),
        evaluate_uniform_feasible_strategy(policy, frames),
        evaluate_topk_accurate_strategy(policy, frames),
        evaluate_greedy_budget_strategy(policy, frames),
        evaluate_random_budget_strategy(policy, frames, seed=790),
        spatial_result,
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_frame_decisions(spatial_result)
    write_summary(results, budget_ms)
    write_mode_distribution(spatial_result)
    print_summary(results, budget_ms)
    write_multiseed_summary(policy)
    write_budget_sweep(policy)


if __name__ == "__main__":
    main()