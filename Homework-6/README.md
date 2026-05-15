# Domain-Specific Spatial Policy for Adaptive Object Detection

This project implements a domain-specific spatial policy runtime for adaptive object detection. The runtime divides each video frame into a 3x3 grid of tiles, computes tile-level features such as motion, importance, and feedback memory, and uses JSON policy rules to decide which tiles should receive YOLO inference.

The project includes:

- A JSON-based spatial policy language
- A Python policy interpreter
- A synthetic Remix-style spatial adaptive inference experiment
- Real-video tiled YOLO11n experiments
- Strict, multi-resolution, and feedback-memory policy variants
- CSV/text outputs for experiment summaries, tile decisions, detections, and demo runs

## Project Structure

```text
Homework-6/
├── images/
│   └── bus.jpg
│
├── models/
│   └── yolo11n.pt
│
├── policies/
│   ├── policy.json
│   ├── spatial_policy.json
│   ├── yolo_spatial_policy.json
│   ├── yolo_spatial_policy_640.json
│   ├── yolo_spatial_policy_960.json
│   ├── yolo_spatial_policy_feedback.json
│   ├── yolo_spatial_policy_feedback_strict.json
│   └── yolo_spatial_policy_multi_res.json
│
├── results/
│   ├── dashcam_feedback/
│   ├── dashcam_feedback_strict/
│   ├── dashcam_original/
│   ├── dashcam_strict_feedback/
│   ├── dashcam_strict_no_feedback/
│   ├── experiment/
│   ├── live_demo/
│   ├── spatial/
│   ├── yolo/
│   ├── yolo_spatial/
│   ├── yolo_tiled_sample1/
│   ├── yolo_tiled_sample2/
│   ├── yolo_tiled_sample2_640/
│   ├── yolo_tiled_sample2_960/
│   ├── yolo_tiled_sample2_feedback/
│   └── yolo_tiled_sample2_multi_res/
│
├── src/
│   ├── __init__.py
│   ├── experiment.py
│   ├── run_policy.py
│   ├── spatial_experiment.py
│   ├── spatial_runtime.py
│   ├── yolo_feedback_live_demo.py
│   ├── yolo_image_benchmark.py
│   ├── yolo_live_demo.py
│   ├── yolo_spatial_experiment.py
│   └── yolo_tiled_experiment.py
│
├── tests/
│   ├── __init__.py
│   ├── test_policy.py
│   ├── test_spatial_baseline.py
│   └── test_spatial_policy.py
│
├── videos/
│   ├── dashcam_footage.mp4
│   ├── parking_lot_footage.mp4
│   └── street_road_footage.mp4
│
├── README.md
└── requirements.txt
```

## Source Code Overview

| File | Purpose |
|---|---|
| `src/run_policy.py` | Loads and evaluates JSON policy rules. |
| `src/experiment.py` | Runs the original policy experiment. |
| `src/spatial_runtime.py` | Contains shared spatial tiling/runtime logic. |
| `src/spatial_experiment.py` | Runs the synthetic Remix-style spatial policy experiment. |
| `src/yolo_image_benchmark.py` | Benchmarks YOLO11n on image input. |
| `src/yolo_live_demo.py` | Runs a live YOLO demonstration. |
| `src/yolo_feedback_live_demo.py` | Runs a live YOLO demonstration with feedback memory. |
| `src/yolo_spatial_experiment.py` | Runs YOLO with spatial policy selection. |
| `src/yolo_tiled_experiment.py` | Runs tiled YOLO experiments comparing full-frame, uniform tiled, and Spatial Policy tiled inference. |

## Dependencies

Recommended environment:

- Python 3.10 or newer
- PyTorch
- Ultralytics
- OpenCV
- NumPy
- pandas
- pytest

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Optional conda setup:

```bash
conda create -n cs790-hw6 python=3.11
conda activate cs790-hw6
pip install -r requirements.txt
```

If CUDA-specific PyTorch installation is needed, install the correct PyTorch build for your system from the official PyTorch installation instructions, then run:

```bash
pip install ultralytics opencv-python numpy pandas pytest
```

## Running Tests

From the `Homework-6/` directory, run:

```bash
python -m pytest tests
```

Expected result:

```text
All policy and spatial runtime tests should pass.
```

## Running Policies with the Policy Interpreter

The policy interpreter is located at:

```text
src/run_policy.py
```

Use this file when you want to test a JSON policy directly without running the full YOLO experiment.

### 1. Check the interpreter options

From the `Homework-6/` directory, run:

```bash
python -m src.run_policy --help
```

This shows the command-line arguments supported by the interpreter.

### 2. Run the basic policy

```bash
python -m src.run_policy --policy policies/policy.json
```

The basic policy is useful for checking that the interpreter can load and evaluate a JSON policy file.

### 3. Run the synthetic spatial policy

```bash
python -m src.run_policy --policy policies/spatial_policy.json
```

This policy is used for the synthetic spatial experiment where each tile is assigned a compute mode such as `skip`, `fast`, `balanced`, or `accurate`.

### 4. Run a YOLO spatial policy

```bash
python -m src.run_policy --policy policies/yolo_spatial_policy.json
```

This policy is used by the tiled YOLO experiments to decide which tiles should receive YOLO inference.

### 5. Run a feedback-memory policy

```bash
python -m src.run_policy --policy policies/yolo_spatial_policy_feedback.json
```

The feedback-memory policy uses previous detections to influence future tile selections.

### 6. Run the strict feedback-memory policy

```bash
python -m src.run_policy --policy policies/yolo_spatial_policy_feedback_strict.json
```

The strict feedback policy uses tighter selection behavior to test the tradeoff between selecting fewer tiles and preserving detection recovery.

### 7. Run resolution-specific policies

```bash
python -m src.run_policy --policy policies/yolo_spatial_policy_640.json
python -m src.run_policy --policy policies/yolo_spatial_policy_960.json
python -m src.run_policy --policy policies/yolo_spatial_policy_multi_res.json
```

These policies are used to test different resolution-related policy settings.

> Note: If your local version of `src/run_policy.py` uses different command-line argument names, run `python -m src.run_policy --help` and use the argument names shown there. The important input is the JSON file inside the `policies/` directory.

## Reproducing the Synthetic Spatial Experiment

Run:

```bash
python -m src.spatial_experiment
```

This experiment simulates a Remix-style spatial adaptive inference workload. Each frame is divided into a 3x3 grid of tiles. Each tile receives simulated motion and importance features, and the policy assigns one of four compute modes:

- `skip`
- `fast`
- `balanced`
- `accurate`

Expected results are written under:

```text
results/spatial/
```

Expected summary from the final report:

```text
Spatial DSL:
Average latency: 129.13 ms/frame
Weighted quality: 0.74
Budget violations: 0
```

## Reproducing the YOLO Image Benchmark

Run:

```bash
python -m src.yolo_image_benchmark
```

This benchmarks YOLO11n on image input.

Expected results are written under:

```text
results/yolo/
```

## Running the Original Tiled YOLO Experiment

Run:

```bash
python -m src.yolo_tiled_experiment \
  --source videos/dashcam_footage.mp4 \
  --frames 50 \
  --policy policies/yolo_spatial_policy.json
```

This compares:

- Full-frame YOLO
- Uniform tiled YOLO
- Spatial Policy tiled YOLO

Expected output directory:

```text
results/dashcam_original/
```

## Running the Feedback-Memory Policy in the Tiled YOLO Experiment

Run:

```bash
python -m src.yolo_tiled_experiment \
  --source videos/dashcam_footage.mp4 \
  --frames 50 \
  --policy policies/yolo_spatial_policy_feedback.json
```

Expected output directory:

```text
results/dashcam_feedback/
```

The feedback-memory policy uses previous YOLO detections to influence future tile selection. If a tile recently produced detections, its feedback score increases. If it stops producing detections, the feedback score decays over time.

Expected feedback-memory result from the final report:

```text
Original Spatial Policy:
Selected tiles/frame: 2.82 / 9
Average latency: 50.11 ms/frame
Detection recovery: 76.58%
Reference coverage: 85.08%

Feedback-Memory Spatial Policy:
Selected tiles/frame: 3.26 / 9
Average latency: 44.51 ms/frame
Detection recovery: 77.41%
Reference coverage: 85.48%
```

## Running the Strict Feedback Policy

Run:

```bash
python -m src.yolo_tiled_experiment \
  --source videos/dashcam_footage.mp4 \
  --frames 50 \
  --policy policies/yolo_spatial_policy_feedback_strict.json
```

Expected output directory:

```text
results/dashcam_feedback_strict/
```

## Running Multi-Resolution Policies

Example using the 640 policy:

```bash
python -m src.yolo_tiled_experiment \
  --source videos/parking_lot_footage.mp4 \
  --frames 50 \
  --policy policies/yolo_spatial_policy_640.json
```

Example using the 960 policy:

```bash
python -m src.yolo_tiled_experiment \
  --source videos/parking_lot_footage.mp4 \
  --frames 50 \
  --policy policies/yolo_spatial_policy_960.json
```

Example using the multi-resolution policy:

```bash
python -m src.yolo_tiled_experiment \
  --source videos/parking_lot_footage.mp4 \
  --frames 50 \
  --policy policies/yolo_spatial_policy_multi_res.json
```

Expected output directories include:

```text
results/yolo_tiled_sample2_640/
results/yolo_tiled_sample2_960/
results/yolo_tiled_sample2_multi_res/
```

## Expected Outputs

The main result files are written inside subdirectories under `results/`.

Depending on the experiment, outputs may include:

| Output | Description |
|---|---|
| Summary files | Average latency, selected tiles, detection recovery, and reference coverage. |
| Decision files | Per-frame tile-selection decisions from the Spatial Policy. |
| Detection files | YOLO detections remapped from tile coordinates to full-frame coordinates. |
| Benchmark files | YOLO image or video benchmark measurements. |
| Demo outputs | Outputs from live or feedback-memory demo runs. |

## Notes on Results

The synthetic experiment demonstrates that the Spatial Policy can express useful compute-allocation behavior under a latency budget.

The real-video YOLO experiments show that the policy can control a real detector pipeline. However, full-frame YOLO11n is still faster in several experiments because naive tiled inference introduces overhead from:

- Frame splitting
- Tile scoring
- Crop creation
- Batching
- Detection remapping
- Postprocessing

Therefore, this project demonstrates a working policy control layer, but consistent real-world speedup would require a more optimized tiled execution backend.
