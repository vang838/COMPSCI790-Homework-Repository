import csv
import time
import urllib.request
from pathlib import Path
from statistics import mean

import torch
from ultralytics import YOLO


RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "yolo"

MODEL_PATH = BASE_DIR / "models/yolo11n.pt"
IMAGE_PATH = BASE_DIR / "images/bus.jpg"

IMAGE_URL = "https://ultralytics.com/images/bus.jpg"

SUMMARY_TXT_PATH = OUTPUT_DIR / "yolo_image_benchmark.txt"
SUMMARY_CSV_PATH = OUTPUT_DIR / "yolo_image_benchmark.csv"


def ensure_image_exists() -> None:
    """Download the sample image if it does not already exist."""
    if IMAGE_PATH.exists():
        return

    print(f"Downloading sample image to {IMAGE_PATH}")
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)


def get_device() -> int | str:
    """Use CUDA GPU if available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        return 0

    return "cpu"


def get_device_name(device: int | str) -> str:
    if device == "cpu":
        return "CPU"

    return torch.cuda.get_device_name(0)


def synchronize_if_needed(device: int | str) -> None:
    """Synchronize CUDA timing so wall-clock measurements are more accurate."""
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_benchmark(
    model: YOLO,
    device: int | str,
    warmup_runs: int = 3,
    measured_runs: int = 10,
) -> list[dict[str, float | int]]:
    """
    Run repeated YOLO inference on one image.

    Warm-up runs are excluded from the final benchmark because the first few
    calls may include model setup, CUDA initialization, and caching overhead.
    """
    for _ in range(warmup_runs):
        model.predict(
            source=str(IMAGE_PATH),
            device=device,
            imgsz=640,
            conf=0.25,
            verbose=False,
        )

    measurements: list[dict[str, float | int]] = []

    for run_index in range(1, measured_runs + 1):
        synchronize_if_needed(device)
        start = time.perf_counter()

        results = model.predict(
            source=str(IMAGE_PATH),
            device=device,
            imgsz=640,
            conf=0.25,
            verbose=False,
        )

        synchronize_if_needed(device)
        end = time.perf_counter()

        result = results[0]
        boxes = result.boxes

        detection_count = len(boxes)
        confidence_values = boxes.conf.tolist() if detection_count > 0 else []
        mean_confidence = mean(confidence_values) if confidence_values else 0.0

        speed = result.speed
        preprocess_ms = float(speed.get("preprocess", 0.0))
        inference_ms = float(speed.get("inference", 0.0))
        postprocess_ms = float(speed.get("postprocess", 0.0))

        wall_clock_ms = (end - start) * 1000

        measurements.append(
            {
                "run": run_index,
                "wall_clock_ms": wall_clock_ms,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "detection_count": detection_count,
                "mean_confidence": mean_confidence,
            }
        )

    return measurements


def write_csv(measurements: list[dict[str, float | int]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "run",
            "wall_clock_ms",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "detection_count",
            "mean_confidence",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in measurements:
            writer.writerow(
                {
                    "run": row["run"],
                    "wall_clock_ms": f"{float(row['wall_clock_ms']):.2f}",
                    "preprocess_ms": f"{float(row['preprocess_ms']):.2f}",
                    "inference_ms": f"{float(row['inference_ms']):.2f}",
                    "postprocess_ms": f"{float(row['postprocess_ms']):.2f}",
                    "detection_count": row["detection_count"],
                    "mean_confidence": f"{float(row['mean_confidence']):.4f}",
                }
            )


def write_summary(
    measurements: list[dict[str, float | int]],
    device_name: str,
    device: int | str,
    warmup_runs: int,
) -> None:
    wall_clock_times = [float(row["wall_clock_ms"]) for row in measurements]
    inference_times = [float(row["inference_ms"]) for row in measurements]
    detection_counts = [int(row["detection_count"]) for row in measurements]
    confidence_values = [float(row["mean_confidence"]) for row in measurements]

    summary = f"""YOLO Image Benchmark Summary
============================

Model: {MODEL_PATH.name}
Image: {IMAGE_PATH.name}
Device argument: {device}
Device name: {device_name}
Warm-up runs excluded: {warmup_runs}
Measured runs: {len(measurements)}

Wall-clock latency after warm-up:
- Average: {mean(wall_clock_times):.2f} ms
- Minimum: {min(wall_clock_times):.2f} ms
- Maximum: {max(wall_clock_times):.2f} ms

YOLO-reported inference time:
- Average: {mean(inference_times):.2f} ms
- Minimum: {min(inference_times):.2f} ms
- Maximum: {max(inference_times):.2f} ms

Detection results:
- Average detections: {mean(detection_counts):.2f}
- Average mean confidence: {mean(confidence_values):.4f}

Output files:
- {SUMMARY_TXT_PATH}
- {SUMMARY_CSV_PATH}
"""

    SUMMARY_TXT_PATH.write_text(summary, encoding="utf-8")


def print_summary(
    measurements: list[dict[str, float | int]],
    device_name: str,
    device: int | str,
    warmup_runs: int,
) -> None:
    wall_clock_times = [float(row["wall_clock_ms"]) for row in measurements]
    inference_times = [float(row["inference_ms"]) for row in measurements]
    detection_counts = [int(row["detection_count"]) for row in measurements]
    confidence_values = [float(row["mean_confidence"]) for row in measurements]

    print("YOLO image benchmark complete.")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Image: {IMAGE_PATH.name}")
    print(f"Device argument: {device}")
    print(f"Device name: {device_name}")
    print(f"Warm-up runs excluded: {warmup_runs}")
    print(f"Measured runs: {len(measurements)}")
    print()
    print("Wall-clock latency after warm-up")
    print(f"Average: {mean(wall_clock_times):.2f} ms")
    print(f"Min:     {min(wall_clock_times):.2f} ms")
    print(f"Max:     {max(wall_clock_times):.2f} ms")
    print()
    print("YOLO-reported inference time")
    print(f"Average: {mean(inference_times):.2f} ms")
    print(f"Min:     {min(inference_times):.2f} ms")
    print(f"Max:     {max(inference_times):.2f} ms")
    print()
    print("Detection results")
    print(f"Average detections:      {mean(detection_counts):.2f}")
    print(f"Average mean confidence: {mean(confidence_values):.4f}")
    print()
    print(f"Summary written to: {SUMMARY_TXT_PATH}")
    print(f"CSV written to: {SUMMARY_CSV_PATH}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_image_exists()

    device = get_device()
    device_name = get_device_name(device)

    warmup_runs = 3
    measured_runs = 10

    model = YOLO(str(MODEL_PATH))

    measurements = run_benchmark(
        model=model,
        device=device,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )

    write_csv(measurements)
    write_summary(
        measurements=measurements,
        device_name=device_name,
        device=device,
        warmup_runs=warmup_runs,
    )

    print_summary(
        measurements=measurements,
        device_name=device_name,
        device=device,
        warmup_runs=warmup_runs,
    )


if __name__ == "__main__":
    main()