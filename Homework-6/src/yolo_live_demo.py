import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.spatial_runtime import load_policy, select_mode
from src.yolo_spatial_experiment import split_frame_into_blocks, predict_timed


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "yolo11n.pt"


def get_device() -> int | str:
    return 0 if torch.cuda.is_available() else "cpu"


def cuda_sync(device: int | str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_block_overlay(
    frame: np.ndarray,
    block: dict[str, Any],
    selected: bool,
) -> None:
    x1 = int(block["x1"])
    y1 = int(block["y1"])
    x2 = int(block["x2"])
    y2 = int(block["y2"])

    mode = str(block["selected_mode"])
    importance = float(block["importance"])
    motion = float(block["motion"])

    color = (0, 255, 255) if selected else (255, 255, 255)
    thickness = 4 if selected else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"{block['block_id']} | {mode} | imp={importance:.2f} mot={motion:.2f}"
    draw_label(frame, label, x1 + 15, y1 + 35, color)


def draw_detection(
    frame: np.ndarray,
    box: Any,
    block: dict[str, Any],
    model: YOLO,
) -> None:
    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()

    x1 += int(block["x1"])
    x2 += int(block["x1"])
    y1 += int(block["y1"])
    y2 += int(block["y1"])

    class_id = int(box.cls[0].detach().cpu().item())
    confidence = float(box.conf[0].detach().cpu().item())

    class_name = model.names.get(class_id, str(class_id))

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        3,
    )

    draw_label(
        frame,
        f"{class_name} {confidence:.2f}",
        int(x1),
        max(30, int(y1) - 10),
        (0, 255, 0),
    )


def process_frame(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    model: YOLO,
    policy: dict[str, Any],
    device: int | str,
    conf: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start = time.perf_counter()

    rows = int(policy["frame_grid"]["rows"])
    cols = int(policy["frame_grid"]["cols"])

    annotated = frame.copy()

    blocks = split_frame_into_blocks(
        frame=frame,
        previous_gray=previous_gray,
        rows=rows,
        cols=cols,
    )

    selected_blocks: list[dict[str, Any]] = []
    blocks_by_imgsz: dict[int, list[dict[str, Any]]] = {}

    for block in blocks:
        mode, rule_name = select_mode(policy, block)

        block["selected_mode"] = mode
        block["rule_applied"] = rule_name

        if mode == "skip":
            draw_block_overlay(annotated, block, selected=False)
            continue

        imgsz = int(policy["modes"][mode]["imgsz"])

        if imgsz <= 0:
            draw_block_overlay(annotated, block, selected=False)
            continue

        selected_blocks.append(block)

        if imgsz not in blocks_by_imgsz:
            blocks_by_imgsz[imgsz] = []

        blocks_by_imgsz[imgsz].append(block)

        draw_block_overlay(annotated, block, selected=True)

    detection_count = 0

    for imgsz, blocks_for_size in blocks_by_imgsz.items():
        crops = [block["crop"] for block in blocks_for_size]

        results, _ = predict_timed(
            model=model,
            source=crops,
            device=device,
            imgsz=imgsz,
            conf=conf,
        )

        for result_index, result in enumerate(results):
            block = blocks_for_size[result_index]

            for box in result.boxes:
                detection_count += 1
                draw_detection(annotated, box, block, model)

    cuda_sync(device)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    overlay_text = (
        f"Spatial DSL + YOLO | selected {len(selected_blocks)}/{rows * cols} tiles | "
        f"detections {detection_count} | latency {latency_ms:.2f} ms"
    )

    cv2.rectangle(annotated, (20, 20), (1450, 75), (0, 0, 0), -1)
    draw_label(annotated, overlay_text, 35, 60, (255, 255, 255))

    stats = {
        "latency_ms": latency_ms,
        "selected_blocks": len(selected_blocks),
        "total_blocks": rows * cols,
        "detections": detection_count,
    }

    return annotated, current_gray, stats


def resize_for_display(frame: np.ndarray, display_width: int) -> np.ndarray:
    height, width = frame.shape[:2]

    if width <= display_width:
        return frame

    scale = display_width / width
    new_size = (display_width, int(height * scale))

    return cv2.resize(frame, new_size)


def run_demo(args: argparse.Namespace) -> None:
    source_path = BASE_DIR / args.source
    policy_path = BASE_DIR / args.policy
    save_path = BASE_DIR / args.save if args.save else None

    policy = load_policy(policy_path)

    device = get_device()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"Source: {source_path}")
    print(f"Policy: {policy_path}")
    print(f"Device: {device_name}")

    model = YOLO(str(DEFAULT_MODEL_PATH))

    capture = cv2.VideoCapture(str(source_path))

    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {source_path}")

    ok, warmup_frame = capture.read()

    if not ok:
        raise RuntimeError("Could not read first frame for warm-up.")

    for _ in range(args.warmup):
        model.predict(
            source=warmup_frame,
            device=device,
            imgsz=640,
            conf=args.conf,
            verbose=False,
        )

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    previous_gray = None
    frame_count = 0

    while frame_count < args.frames:
        ok, frame = capture.read()

        if not ok:
            break

        frame_count += 1

        annotated, previous_gray, stats = process_frame(
            frame=frame,
            previous_gray=previous_gray,
            model=model,
            policy=policy,
            device=device,
            conf=args.conf,
        )

        if save_path is not None and writer is None:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            height, width = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                str(save_path),
                fourcc,
                args.output_fps,
                (width, height),
            )

        if writer is not None:
            writer.write(annotated)

        if args.display:
            display_frame = resize_for_display(annotated, args.display_width)
            cv2.imshow("Spatial DSL + YOLO Demo", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        print(
            f"Frame {frame_count}: "
            f"{stats['latency_ms']:.2f} ms, "
            f"selected {stats['selected_blocks']}/{stats['total_blocks']}, "
            f"detections {stats['detections']}"
        )

    capture.release()

    if writer is not None:
        writer.release()

    if args.display:
        cv2.destroyAllWindows()

    if save_path is not None:
        print(f"Annotated demo video written to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show or save a live-style Spatial DSL + YOLO annotated video demo."
    )

    parser.add_argument(
        "--source",
        default="videos/sample2.mp4",
        help="Video path relative to Homework-6.",
    )

    parser.add_argument(
        "--policy",
        default="policies/yolo_spatial_policy.json",
        help="Policy path relative to Homework-6.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warm-up predictions before demo.",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Show OpenCV live display window.",
    )

    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Resize live display window to this width.",
    )

    parser.add_argument(
        "--save",
        default="results/live_demo/dashcam_feedback_demo.mp4",
    )

    parser.add_argument(
        "--output-fps",
        type=float,
        default=15.0,
        help="FPS for saved annotated video.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.save == "":
        args.save = None

    run_demo(args)


if __name__ == "__main__":
    main()