import argparse
import csv
import time
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.spatial_runtime import load_policy, select_mode
from src.yolo_spatial_experiment import (
    BASE_DIR,
    RESULTS_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_IMAGE_PATH,
    DEFAULT_POLICY_PATH,
    ensure_default_image_exists,
    get_device,
    cuda_sync,
    load_frames,
    split_frame_into_blocks,
    predict_timed,
    extract_detections,
    compute_detection_recovery,
    compute_reference_coverage,
    mean_confidence,
)


def infer_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
        return output_dir if output_dir.is_absolute() else BASE_DIR / output_dir

    source_stem = Path(args.source).stem.lower()
    policy_stem = Path(args.policy).stem.lower()

    if "dashcam" in source_stem:
        if args.use_feedback and "feedback_strict" in policy_stem:
            return RESULTS_DIR / "dashcam_feedback_strict"

        if args.use_feedback and "feedback" in policy_stem:
            return RESULTS_DIR / "dashcam_feedback"

        if "feedback_strict" in policy_stem or "strict" in policy_stem:
            return RESULTS_DIR / "dashcam_strict_no_feedback"

        return RESULTS_DIR / "dashcam_original"

    if "parking" in source_stem or "sample2" in source_stem:
        if "multi_res" in policy_stem:
            return RESULTS_DIR / "yolo_tiled_sample2_multi_res"

        if "960" in policy_stem:
            return RESULTS_DIR / "yolo_tiled_sample2_960"

        if "640" in policy_stem:
            return RESULTS_DIR / "yolo_tiled_sample2_640"

        if "feedback" in policy_stem:
            return RESULTS_DIR / "yolo_tiled_sample2_feedback"

        return RESULTS_DIR / "yolo_tiled_sample2"

    return RESULTS_DIR / "yolo_tiled_sample1"


def write_summary(summary: dict[str, Any], summary_path: Path) -> None:
    with summary_path.open("w", encoding="utf-8", newline="") as file:


def write_spatial_decisions(rows: list[dict[str, Any]], decisions_path: Path) -> None:
    with decisions_path.open("w", encoding="utf-8", newline="") as file:


def write_detections(rows: list[dict[str, Any]]) -> None:
    with DETECTIONS_PATH.open("w", encoding="utf-8", newline="") as file:


def run_uniform_tiled(
    model: YOLO,
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    rows: int,
    cols: int,
    device: int | str,
    imgsz: int,
    conf: float,
    frame_id: int,
) -> tuple[float, list[dict[str, Any]]]:
    start = time.perf_counter()

    blocks = split_frame_into_blocks(
        frame=frame,
        previous_gray=previous_gray,
        rows=rows,
        cols=cols,
    )

    crops = [block["crop"] for block in blocks]

    results, _ = predict_timed(
        model=model,
        source=crops,
        device=device,
        imgsz=imgsz,
        conf=conf,
    )

    cuda_sync(device)
    end = time.perf_counter()

    detections = extract_detections(
        results=results,
        frame_id=frame_id,
        strategy="uniform_tiled_all_blocks",
        block_meta=blocks,
    )

    latency_ms = (end - start) * 1000
    return latency_ms, detections


def run_spatial_dsl_tiled(
    model: YOLO,
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    policy: dict[str, Any],
    rows: int,
    cols: int,
    device: int | str,
    imgsz: int,
    conf: float,
    frame_id: int,
    feedback_scores: dict[str, float] | None = None,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], ]:
    start = time.perf_counter()

    blocks = split_frame_into_blocks(
        frame=frame,
        previous_gray=previous_gray,
        rows=rows,
        cols=cols,
    )
    
    if feedback_scores is not None:
        attach_feedback_to_blocks(blocks, feedback_scores)
    else:
        for block in blocks:
            block["feedback_score"] = 0.0

    selected_blocks: list[dict[str, Any]] = []
    selected_blocks_by_imgsz: dict[int, list[dict[str, Any]]] = {}
    decision_rows: list[dict[str, Any]] = []

    for block in blocks:
        mode, rule_name = select_mode(policy, block)
        block["selected_mode"] = mode
        block["rule_applied"] = rule_name

        decision_rows.append(
            {
                "frame_id": frame_id,
                "block_id": block["block_id"],
                "row": block["row"],
                "col": block["col"],
                "importance": block["importance"],
                "motion": block["motion"],
                "edge": block["edge"],
                "selected_mode": mode,
                "rule_applied": rule_name,
                "x1": block["x1"],
                "y1": block["y1"],
                "x2": block["x2"],
                "y2": block["y2"],
                "feedback_score": block["feedback_score"],
            }
        )

        if mode == "skip":
            continue

        mode_config = policy["modes"][mode]
        block_imgsz = int(mode_config["imgsz"])

        if block_imgsz <= 0:
            continue

        selected_blocks.append(block)

        if block_imgsz not in selected_blocks_by_imgsz:
            selected_blocks_by_imgsz[block_imgsz] = []

        selected_blocks_by_imgsz[block_imgsz].append(block)

    detections: list[dict[str, Any]] = []

    for block_imgsz, blocks_for_size in selected_blocks_by_imgsz.items():
        crops = [block["crop"] for block in blocks_for_size]

        results, _ = predict_timed(
            model=model,
            source=crops,
            device=device,
            imgsz=block_imgsz,
            conf=conf,
        )

        detections.extend(
            extract_detections(
                results=results,
                frame_id=frame_id,
                strategy=f"spatial_dsl_selected_blocks_{block_imgsz}",
                block_meta=blocks_for_size,
            )
        )

    cuda_sync(device)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    return latency_ms, detections, selected_blocks, decision_rows

def make_block_id(row: int, col: int) -> str:
    return f"r{row}c{col}"


def initialize_feedback_scores(rows: int, cols: int) -> dict[str, float]:
    return {
        make_block_id(row, col): 0.0
        for row in range(rows)
        for col in range(cols)
    }


def decay_feedback_scores(
    feedback_scores: dict[str, float],
    decay: float = 0.70,
) -> None:
    for block_id in feedback_scores:
        feedback_scores[block_id] *= decay

        if feedback_scores[block_id] < 0.05:
            feedback_scores[block_id] = 0.0


def detection_center(detection: dict[str, Any]) -> tuple[float, float]:
    center_x = (float(detection["x1"]) + float(detection["x2"])) / 2.0
    center_y = (float(detection["y1"]) + float(detection["y2"])) / 2.0
    return center_x, center_y


def block_id_for_point(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
    rows: int,
    cols: int,
) -> str:
    col = min(cols - 1, max(0, int(x / frame_width * cols)))
    row = min(rows - 1, max(0, int(y / frame_height * rows)))
    return make_block_id(row, col)


def update_feedback_from_detections(
    feedback_scores: dict[str, float],
    detections: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    rows: int,
    cols: int,
    boost: float = 1.0,
) -> None:
    for detection in detections:
        center_x, center_y = detection_center(detection)

        block_id = block_id_for_point(
            x=center_x,
            y=center_y,
            frame_width=frame_width,
            frame_height=frame_height,
            rows=rows,
            cols=cols,
        )

        feedback_scores[block_id] = max(feedback_scores[block_id], boost)


def attach_feedback_to_blocks(
    blocks: list[dict[str, Any]],
    feedback_scores: dict[str, float],
) -> None:
    for block in blocks:
        block_id = str(block["block_id"])
        block["feedback_score"] = round(feedback_scores.get(block_id, 0.0), 4)

def run_experiment(args: argparse.Namespace) -> None:
    output_dir = infer_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "yolo_tiled_summary.csv"
    decisions_path = output_dir / "yolo_tiled_spatial_decisions.csv"
    detections_path = output_dir / "yolo_tiled_detections.csv"

    if args.source == "bus.jpg":
        ensure_default_image_exists()

    source_path = BASE_DIR / args.source

    policy = load_policy(BASE_DIR / args.policy)
    rows = int(policy["frame_grid"]["rows"])
    cols = int(policy["frame_grid"]["cols"])
    feedback_scores = initialize_feedback_scores(rows, cols)

    detect_imgsz_values = sorted(
        {
            int(mode_config["imgsz"])
            for mode_name, mode_config in policy["modes"].items()
            if mode_name != "skip" and int(mode_config["imgsz"]) > 0
        }
    )

    tile_imgsz = detect_imgsz_values[0] if detect_imgsz_values else 0

    device = get_device()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    model = YOLO(str(DEFAULT_MODEL_PATH))
    frames = load_frames(source_path, max_frames=args.frames)

    # Warm-up on first frame.
    for _ in range(args.warmup):
        model.predict(
            source=frames[0],
            device=device,
            imgsz=args.reference_imgsz,
            conf=args.conf,
            verbose=False,
        )

    previous_gray: np.ndarray | None = None

    full_latencies: list[float] = []
    tiled_latencies: list[float] = []
    spatial_latencies: list[float] = []

    full_detection_counts: list[int] = []
    tiled_detection_counts: list[int] = []
    spatial_detection_counts: list[int] = []

    full_confidences: list[float] = []
    tiled_confidences: list[float] = []
    spatial_confidences: list[float] = []

    tiled_recovery_scores: list[float] = []
    spatial_recovery_scores: list[float] = []
    spatial_coverage_scores: list[float] = []
    selected_block_counts: list[int] = []

    all_decision_rows: list[dict[str, Any]] = []
    all_detection_rows: list[dict[str, Any]] = []

    for frame_id, frame in enumerate(frames, start=1):
        # Full-frame YOLO reference.
        full_results, full_latency = predict_timed(
            model=model,
            source=frame,
            device=device,
            imgsz=args.reference_imgsz,
            conf=args.conf,
        )

        full_detections = extract_detections(
            results=full_results,
            frame_id=frame_id,
            strategy="full_frame_reference",
        )

        # Uniform tiled baseline: run YOLO on all 9 blocks.
        tiled_latency, tiled_detections = run_uniform_tiled(
            model=model,
            frame=frame,
            previous_gray=previous_gray,
            rows=rows,
            cols=cols,
            device=device,
            imgsz=tile_imgsz,
            conf=args.conf,
            frame_id=frame_id,
        )

        # Spatial DSL tiled: run YOLO only on selected blocks.
        spatial_latency, spatial_detections, selected_blocks, decision_rows = run_spatial_dsl_tiled(
            model=model,
            frame=frame,
            previous_gray=previous_gray,
            policy=policy,
            rows=rows,
            cols=cols,
            device=device,
            imgsz=tile_imgsz,
            conf=args.conf,
            frame_id=frame_id,
            feedback_scores=feedback_scores if args.use_feedback else None,
        )
        
        if args.use_feedback:
            decay_feedback_scores(feedback_scores, decay=args.feedback_decay)

            frame_height, frame_width = frame.shape[:2]

            update_feedback_from_detections(
                feedback_scores=feedback_scores,
                detections=spatial_detections,
                frame_width=frame_width,
                frame_height=frame_height,
                rows=rows,
                cols=cols,
            )

        tiled_recovery = compute_detection_recovery(
            reference_detections=full_detections,
            spatial_detections=tiled_detections,
            iou_threshold=args.iou_threshold,
        )

        spatial_recovery = compute_detection_recovery(
            reference_detections=full_detections,
            spatial_detections=spatial_detections,
            iou_threshold=args.iou_threshold,
        )

        spatial_coverage = compute_reference_coverage(
            reference_detections=full_detections,
            selected_blocks=selected_blocks,
        )

        full_latencies.append(full_latency)
        tiled_latencies.append(tiled_latency)
        spatial_latencies.append(spatial_latency)

        full_detection_counts.append(len(full_detections))
        tiled_detection_counts.append(len(tiled_detections))
        spatial_detection_counts.append(len(spatial_detections))

        full_confidences.append(mean_confidence(full_detections))
        tiled_confidences.append(mean_confidence(tiled_detections))
        spatial_confidences.append(mean_confidence(spatial_detections))

        tiled_recovery_scores.append(tiled_recovery)
        spatial_recovery_scores.append(spatial_recovery)
        spatial_coverage_scores.append(spatial_coverage)
        selected_block_counts.append(len(selected_blocks))

        all_decision_rows.extend(decision_rows)
        all_detection_rows.extend(full_detections)
        all_detection_rows.extend(tiled_detections)
        all_detection_rows.extend(spatial_detections)

        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    full_avg_latency = mean(full_latencies)
    tiled_avg_latency = mean(tiled_latencies)
    spatial_avg_latency = mean(spatial_latencies)

    spatial_vs_tiled_speedup = (
        (tiled_avg_latency - spatial_avg_latency) / tiled_avg_latency * 100
        if tiled_avg_latency > 0
        else 0.0
    )

    spatial_vs_full_speedup = (
        (full_avg_latency - spatial_avg_latency) / full_avg_latency * 100
        if full_avg_latency > 0
        else 0.0
    )

    summary = {
        "source": args.source,
        "frames": len(frames),
        "device": device_name,
        "grid": f"{rows}x{cols}",
        "reference_imgsz": args.reference_imgsz,
        "tile_imgsz": "/".join(str(value) for value in detect_imgsz_values),
        "full_frame_avg_latency_ms": f"{full_avg_latency:.2f}",
        "uniform_tiled_avg_latency_ms": f"{tiled_avg_latency:.2f}",
        "spatial_dsl_tiled_avg_latency_ms": f"{spatial_avg_latency:.2f}",
        "spatial_vs_uniform_tiled_speedup_percent": f"{spatial_vs_tiled_speedup:.2f}",
        "spatial_vs_full_frame_speedup_percent": f"{spatial_vs_full_speedup:.2f}",
        "full_frame_avg_detections": f"{mean(full_detection_counts):.2f}",
        "uniform_tiled_avg_detections": f"{mean(tiled_detection_counts):.2f}",
        "spatial_dsl_avg_detections": f"{mean(spatial_detection_counts):.2f}",
        "full_frame_mean_confidence": f"{mean(full_confidences):.4f}",
        "uniform_tiled_mean_confidence": f"{mean(tiled_confidences):.4f}",
        "spatial_dsl_mean_confidence": f"{mean(spatial_confidences):.4f}",
        "uniform_tiled_detection_recovery": f"{mean(tiled_recovery_scores):.4f}",
        "spatial_dsl_detection_recovery": f"{mean(spatial_recovery_scores):.4f}",
        "spatial_reference_coverage": f"{mean(spatial_coverage_scores):.4f}",
        "spatial_avg_selected_blocks": f"{mean(selected_block_counts):.2f}",
    }

    write_summary(summary, summary_path)
    write_spatial_decisions(all_decision_rows, decisions_path)
    write_detections(all_detection_rows, detections_path)

    print("YOLO tiled Spatial DSL experiment complete.")
    print(f"Source: {args.source}")
    print(f"Frames evaluated: {len(frames)}")
    print(f"Device: {device_name}")
    print(f"Grid: {rows}x{cols}")
    print(f"Tile image sizes: {detect_imgsz_values}")
    print()

    print("Strategy                       | Avg Latency(ms/frame) | Avg Detections | Mean Confidence")
    print("-" * 95)

    print(
        f"{'Full-frame YOLO reference':<30} | "
        f"{full_avg_latency:>21.2f} | "
        f"{mean(full_detection_counts):>14.2f} | "
        f"{mean(full_confidences):.4f}"
    )

    print(
        f"{'Uniform tiled YOLO':<30} | "
        f"{tiled_avg_latency:>21.2f} | "
        f"{mean(tiled_detection_counts):>14.2f} | "
        f"{mean(tiled_confidences):.4f}"
    )

    print(
        f"{'Spatial DSL tiled YOLO':<30} | "
        f"{spatial_avg_latency:>21.2f} | "
        f"{mean(spatial_detection_counts):>14.2f} | "
        f"{mean(spatial_confidences):.4f}"
    )

    print()
    print(f"Spatial vs uniform tiled speedup: {spatial_vs_tiled_speedup:.2f}%")
    print(f"Spatial vs full-frame speedup:    {spatial_vs_full_speedup:.2f}%")
    print(f"Average selected blocks/frame:    {mean(selected_block_counts):.2f} of {rows * cols}")
    print(f"Uniform tiled detection recovery: {mean(tiled_recovery_scores):.4f}")
    print(f"Spatial DSL detection recovery:   {mean(spatial_recovery_scores):.4f}")
    print(f"Spatial reference coverage:       {mean(spatial_coverage_scores):.4f}")
    print()
    print(f"Summary written to: {summary_path}")
    print(f"Spatial decisions written to: {decisions_path}")
    print(f"Detections written to: {detections_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare uniform tiled YOLO against Spatial DSL tiled YOLO."
    )

    parser.add_argument(
        "--source",
        default="bus.jpg",
        help="Image or video path relative to Homework-6.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of frames to evaluate.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )

    parser.add_argument(
        "--reference-imgsz",
        type=int,
        default=640,
        help="Image size for full-frame YOLO reference.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warm-up predictions before measuring.",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.30,
        help="IoU threshold for matching tiled detections to reference detections.",
    )
    
    parser.add_argument(
        "--policy",
        default="policies/yolo_spatial_policy.json",
        help="Policy path relative to Homework-6.",
    )
    
    parser.add_argument(
        "--use-feedback",
        action="store_true",
        help="Enable previous-detection feedback for tile selection.",
    )

    parser.add_argument(
        "--feedback-decay",
        type=float,
        default=0.70,
        help="Decay factor for feedback scores between frames.",
    )
    
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory relative to Homework-6. If omitted, a directory is inferred from source and policy.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()