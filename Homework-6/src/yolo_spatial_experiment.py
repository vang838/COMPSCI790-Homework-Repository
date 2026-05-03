import argparse
import csv
import time
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.spatial_runtime import load_policy, select_mode


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

DEFAULT_MODEL_PATH = BASE_DIR / "yolo11n.pt"
DEFAULT_IMAGE_PATH = BASE_DIR / "bus.jpg"
DEFAULT_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
DEFAULT_POLICY_PATH = BASE_DIR / "policies" / "yolo_spatial_policy.json"

SUMMARY_CSV_PATH = RESULTS_DIR / "yolo_spatial_summary.csv"
FRAME_DECISIONS_CSV_PATH = RESULTS_DIR / "yolo_spatial_frame_decisions.csv"
DETECTIONS_CSV_PATH = RESULTS_DIR / "yolo_spatial_detections.csv"


def ensure_default_image_exists() -> None:
    if DEFAULT_IMAGE_PATH.exists():
        return

    print(f"Downloading default image to {DEFAULT_IMAGE_PATH}")
    urllib.request.urlretrieve(DEFAULT_IMAGE_URL, DEFAULT_IMAGE_PATH)


def get_device() -> int | str:
    return 0 if torch.cuda.is_available() else "cpu"


def cuda_sync(device: int | str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_frames(source_path: Path, max_frames: int) -> list[np.ndarray]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if source_path.suffix.lower() in image_extensions:
        image = cv2.imread(str(source_path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {source_path}")

        return [image.copy() for _ in range(max_frames)]

    capture = cv2.VideoCapture(str(source_path))

    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {source_path}")

    frames: list[np.ndarray] = []

    while len(frames) < max_frames:
        ok, frame = capture.read()

        if not ok:
            break

        frames.append(frame)

    capture.release()

    if not frames:
        raise RuntimeError(f"No frames were read from source: {source_path}")

    return frames


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return [0.0 for _ in values]

    return [
        (value - min_value) / (max_value - min_value)
        for value in values
    ]


def split_frame_into_blocks(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    rows: int,
    cols: int,
) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edge_scores: list[float] = []
    motion_scores: list[float] = []
    blocks: list[dict[str, Any]] = []

    for row in range(rows):
        for col in range(cols):
            x1 = int(col * width / cols)
            y1 = int(row * height / rows)
            x2 = int((col + 1) * width / cols)
            y2 = int((row + 1) * height / rows)

            crop = frame[y1:y2, x1:x2]
            gray_crop = gray[y1:y2, x1:x2]

            edges = cv2.Canny(gray_crop, threshold1=80, threshold2=160)
            edge_score = float(np.mean(edges) / 255.0)

            if previous_gray is None:
                motion_score = 0.0
            else:
                previous_crop = previous_gray[y1:y2, x1:x2]
                diff = cv2.absdiff(gray_crop, previous_crop)
                motion_score = float(np.mean(diff) / 255.0)

            edge_scores.append(edge_score)
            motion_scores.append(motion_score)

            blocks.append(
                {
                    "block_id": f"r{row}c{col}",
                    "row": row,
                    "col": col,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop": crop,
                    "raw_edge": edge_score,
                    "raw_motion": motion_score,
                }
            )

    normalized_edges = normalize(edge_scores)
    normalized_motion = normalize(motion_scores)

    for index, block in enumerate(blocks):
        edge = normalized_edges[index]
        motion = normalized_motion[index]

        importance = 0.70 * edge + 0.30 * motion

        block["edge"] = round(edge, 4)
        block["motion"] = round(motion, 4)
        block["importance"] = round(importance, 4)

    return blocks


def predict_timed(
    model: YOLO,
    source: Any,
    device: int | str,
    imgsz: int,
    conf: float,
) -> tuple[list[Any], float]:
    cuda_sync(device)
    start = time.perf_counter()

    results = model.predict(
        source=source,
        device=device,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
    )

    cuda_sync(device)
    end = time.perf_counter()

    return list(results), (end - start) * 1000


def extract_detections(
    results: list[Any],
    frame_id: int,
    strategy: str,
    block_meta: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []

    for result_index, result in enumerate(results):
        if block_meta is None:
            offset_x = 0
            offset_y = 0
            block_id = "full"
        else:
            meta = block_meta[result_index]
            offset_x = int(meta["x1"])
            offset_y = int(meta["y1"])
            block_id = str(meta["block_id"])

        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            class_id = int(box.cls[0].detach().cpu().item())
            confidence = float(box.conf[0].detach().cpu().item())

            detections.append(
                {
                    "frame_id": frame_id,
                    "strategy": strategy,
                    "block_id": block_id,
                    "class_id": class_id,
                    "confidence": confidence,
                    "x1": x1 + offset_x,
                    "y1": y1 + offset_y,
                    "x2": x2 + offset_x,
                    "y2": y2 + offset_y,
                }
            )

    return detections


def bbox_center(detection: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(detection["x1"]) + float(detection["x2"])) / 2.0,
        (float(detection["y1"]) + float(detection["y2"])) / 2.0,
    )


def point_inside_block(x: float, y: float, block: dict[str, Any]) -> bool:
    return (
        float(block["x1"]) <= x < float(block["x2"])
        and float(block["y1"]) <= y < float(block["y2"])
    )


def compute_reference_coverage(
    reference_detections: list[dict[str, Any]],
    selected_blocks: list[dict[str, Any]],
) -> float:
    if not reference_detections:
        return 1.0

    covered = 0

    for detection in reference_detections:
        center_x, center_y = bbox_center(detection)

        if any(point_inside_block(center_x, center_y, block) for block in selected_blocks):
            covered += 1

    return covered / len(reference_detections)


def compute_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
    bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_detection_recovery(
    reference_detections: list[dict[str, Any]],
    spatial_detections: list[dict[str, Any]],
    iou_threshold: float = 0.30,
) -> float:
    if not reference_detections:
        return 1.0

    matched_spatial_indices: set[int] = set()
    recovered = 0

    for reference in reference_detections:
        best_index = None
        best_iou = 0.0

        for index, spatial in enumerate(spatial_detections):
            if index in matched_spatial_indices:
                continue

            if int(reference["class_id"]) != int(spatial["class_id"]):
                continue

            iou = compute_iou(reference, spatial)

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched_spatial_indices.add(best_index)
            recovered += 1

    return recovered / len(reference_detections)


def mean_confidence(detections: list[dict[str, Any]]) -> float:
    if not detections:
        return 0.0

    return mean(float(detection["confidence"]) for detection in detections)


def write_frame_decisions(rows: list[dict[str, Any]]) -> None:
    with FRAME_DECISIONS_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "frame_id",
            "block_id",
            "row",
            "col",
            "importance",
            "motion",
            "edge",
            "selected_mode",
            "rule_applied",
            "x1",
            "y1",
            "x2",
            "y2",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_detections(rows: list[dict[str, Any]]) -> None:
    with DETECTIONS_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "frame_id",
            "strategy",
            "block_id",
            "class_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "frame_id": row["frame_id"],
                    "strategy": row["strategy"],
                    "block_id": row["block_id"],
                    "class_id": row["class_id"],
                    "confidence": f"{float(row['confidence']):.4f}",
                    "x1": f"{float(row['x1']):.2f}",
                    "y1": f"{float(row['y1']):.2f}",
                    "x2": f"{float(row['x2']):.2f}",
                    "y2": f"{float(row['y2']):.2f}",
                }
            )


def write_summary(summary: dict[str, Any]) -> None:
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "frames",
            "source",
            "device",
            "full_frame_avg_latency_ms",
            "spatial_dsl_avg_latency_ms",
            "latency_speedup_percent",
            "full_frame_avg_detections",
            "spatial_dsl_avg_detections",
            "spatial_avg_selected_blocks",
            "spatial_reference_coverage",
            "spatial_detection_recovery",
            "full_frame_mean_confidence",
            "spatial_mean_confidence",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def run_experiment(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.source == "bus.jpg":
        ensure_default_image_exists()

    source_path = BASE_DIR / args.source
    policy = load_policy(DEFAULT_POLICY_PATH)

    rows = int(policy["frame_grid"]["rows"])
    cols = int(policy["frame_grid"]["cols"])

    device = get_device()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    model = YOLO(str(DEFAULT_MODEL_PATH))

    frames = load_frames(source_path, max_frames=args.frames)

    # Warm up GPU/model.
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
    spatial_latencies: list[float] = []
    full_detection_counts: list[int] = []
    spatial_detection_counts: list[int] = []
    selected_block_counts: list[int] = []
    coverage_scores: list[float] = []
    recovery_scores: list[float] = []
    full_confidences: list[float] = []
    spatial_confidences: list[float] = []

    frame_decision_rows: list[dict[str, Any]] = []
    all_detection_rows: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(frames, start=1):
        # Full-frame reference.
        full_results, full_latency = predict_timed(
            model=model,
            source=frame,
            device=device,
            imgsz=args.reference_imgsz,
            conf=args.conf,
        )

        full_detections = extract_detections(
            results=full_results,
            frame_id=frame_index,
            strategy="full_frame_reference",
        )

        # Spatial DSL decision.
        spatial_start = time.perf_counter()

        blocks = split_frame_into_blocks(
            frame=frame,
            previous_gray=previous_gray,
            rows=rows,
            cols=cols,
        )

        selected_blocks: list[dict[str, Any]] = []

        for block in blocks:
            mode, rule_name = select_mode(policy, block)
            block["selected_mode"] = mode
            block["rule_applied"] = rule_name

            frame_decision_rows.append(
                {
                    "frame_id": frame_index,
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
                }
            )

            if mode != "skip":
                selected_blocks.append(block)

        spatial_detections: list[dict[str, Any]] = []

        if selected_blocks:
            crops = [block["crop"] for block in selected_blocks]
            spatial_results, _ = predict_timed(
                model=model,
                source=crops,
                device=device,
                imgsz=int(policy["modes"]["detect"]["imgsz"]),
                conf=args.conf,
            )

            spatial_detections = extract_detections(
                results=spatial_results,
                frame_id=frame_index,
                strategy="spatial_dsl",
                block_meta=selected_blocks,
            )

        cuda_sync(device)
        spatial_end = time.perf_counter()
        spatial_latency = (spatial_end - spatial_start) * 1000

        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        coverage = compute_reference_coverage(
            reference_detections=full_detections,
            selected_blocks=selected_blocks,
        )

        recovery = compute_detection_recovery(
            reference_detections=full_detections,
            spatial_detections=spatial_detections,
            iou_threshold=args.iou_threshold,
        )

        full_latencies.append(full_latency)
        spatial_latencies.append(spatial_latency)
        full_detection_counts.append(len(full_detections))
        spatial_detection_counts.append(len(spatial_detections))
        selected_block_counts.append(len(selected_blocks))
        coverage_scores.append(coverage)
        recovery_scores.append(recovery)
        full_confidences.append(mean_confidence(full_detections))
        spatial_confidences.append(mean_confidence(spatial_detections))

        all_detection_rows.extend(full_detections)
        all_detection_rows.extend(spatial_detections)

    full_avg_latency = mean(full_latencies)
    spatial_avg_latency = mean(spatial_latencies)

    speedup_percent = (
        (full_avg_latency - spatial_avg_latency) / full_avg_latency * 100.0
        if full_avg_latency > 0
        else 0.0
    )

    summary = {
        "frames": len(frames),
        "source": args.source,
        "device": device_name,
        "full_frame_avg_latency_ms": f"{full_avg_latency:.2f}",
        "spatial_dsl_avg_latency_ms": f"{spatial_avg_latency:.2f}",
        "latency_speedup_percent": f"{speedup_percent:.2f}",
        "full_frame_avg_detections": f"{mean(full_detection_counts):.2f}",
        "spatial_dsl_avg_detections": f"{mean(spatial_detection_counts):.2f}",
        "spatial_avg_selected_blocks": f"{mean(selected_block_counts):.2f}",
        "spatial_reference_coverage": f"{mean(coverage_scores):.4f}",
        "spatial_detection_recovery": f"{mean(recovery_scores):.4f}",
        "full_frame_mean_confidence": f"{mean(full_confidences):.4f}",
        "spatial_mean_confidence": f"{mean(spatial_confidences):.4f}",
    }

    write_summary(summary)
    write_frame_decisions(frame_decision_rows)
    write_detections(all_detection_rows)

    print("YOLO Spatial DSL experiment complete.")
    print(f"Source: {args.source}")
    print(f"Frames evaluated: {len(frames)}")
    print(f"Device: {device_name}")
    print()
    print("Strategy              | Avg Latency(ms/frame) | Avg Detections | Mean Confidence")
    print("-" * 82)
    print(
        f"{'Full-frame YOLO':<21} | "
        f"{full_avg_latency:>21.2f} | "
        f"{mean(full_detection_counts):>14.2f} | "
        f"{mean(full_confidences):.4f}"
    )
    print(
        f"{'Spatial DSL + YOLO':<21} | "
        f"{spatial_avg_latency:>21.2f} | "
        f"{mean(spatial_detection_counts):>14.2f} | "
        f"{mean(spatial_confidences):.4f}"
    )
    print()
    print(f"Latency speedup: {speedup_percent:.2f}%")
    print(f"Average selected blocks/frame: {mean(selected_block_counts):.2f} of {rows * cols}")
    print(f"Reference detection coverage: {mean(coverage_scores):.4f}")
    print(f"Spatial detection recovery:   {mean(recovery_scores):.4f}")
    print()
    print(f"Summary written to: {SUMMARY_CSV_PATH}")
    print(f"Frame decisions written to: {FRAME_DECISIONS_CSV_PATH}")
    print(f"Detections written to: {DETECTIONS_CSV_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal real YOLO + Spatial DSL experiment."
    )

    parser.add_argument(
        "--source",
        default="bus.jpg",
        help="Image or video path relative to Homework-6. Default: bus.jpg",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of frames to evaluate. For images, the image is repeated.",
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
        help="IoU threshold for matching spatial detections to reference detections.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()