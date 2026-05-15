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
| `src/run_policy.py` | Loads and evaluates the basic JSON policy in `policies/policy.json`. |
| `src/experiment.py` | Runs the original policy experiment using generated frame metadata. |
| `src/spatial_runtime.py` | Contains shared spatial policy, tiling, budget, and baseline logic. |
| `src/spatial_experiment.py` | Runs the synthetic Remix-style spatial policy experiment. |
| `src/yolo_image_benchmark.py` | Benchmarks YOLO11n on image input. |
| `src/yolo_live_demo.py` | Runs or saves a live-style YOLO spatial policy demo. |
| `src/yolo_feedback_live_demo.py` | Runs or saves a live-style YOLO demo with feedback memory. |
| `src/yolo_spatial_experiment.py` | Runs a minimal YOLO spatial policy experiment. |
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

## Running the Basic Policy Interpreter

The basic policy interpreter is located at:

```text
src/run_policy.py
```

This script tests the small JSON policy in:

```text
policies/policy.json
```

It evaluates generated frame metadata with fields such as `motion` and `queue`, then selects one of the modes defined in the policy file.

Check the interpreter options:

```bash
python -m src.run_policy --help
```

Run the default policy:

```bash
python -m src.run_policy --policy policies/policy.json
```

Validate the policy JSON syntax:

```bash
python -m json.tool policies/policy.json
```

Expected behavior:

```text
Frame-level policy decisions are printed to the terminal.
A baseline comparison is printed for fixed fast, fixed balanced, fixed accurate, and Adaptive DSL modes.
```

Note: `src/run_policy.py` is only for the basic policy interpreter. The spatial and YOLO policies should be run through the experiment scripts below.

## Running the Original Generated-Frame Experiment

Run:

```bash
python -m src.experiment
```

Expected output files:

```text
results/experiment/frame_decisions.csv
results/experiment/experiment_summary.csv
```

This experiment uses generated frame metadata and the basic adaptive policy from `policies/policy.json`.

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

Expected output files:

```text
results/spatial/spatial_frame_decisions.csv
results/spatial/spatial_experiment_summary.csv
results/spatial/spatial_mode_distribution.csv
results/spatial/spatial_multiseed_summary.csv
results/spatial/spatial_budget_sweep.csv
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

Expected output files:

```text
results/yolo/yolo_image_benchmark.txt
results/yolo/yolo_image_benchmark.csv
```

## Running the Minimal YOLO Spatial Experiment

Run:

```bash
python -m src.yolo_spatial_experiment   --source videos/dashcam_footage.mp4   --frames 50
```

Expected output files:

```text
results/yolo_spatial/yolo_spatial_summary.csv
results/yolo_spatial/yolo_spatial_frame_decisions.csv
results/yolo_spatial/yolo_spatial_detections.csv
```

## Running the Original Tiled YOLO Experiment

Run:

```bash
python -m src.yolo_tiled_experiment   --source videos/dashcam_footage.mp4   --frames 50   --policy policies/yolo_spatial_policy.json
```

This compares:

- Full-frame YOLO
- Uniform tiled YOLO
- Spatial Policy tiled YOLO

Expected output files:

```text
results/dashcam_original/yolo_tiled_summary.csv
results/dashcam_original/yolo_tiled_spatial_decisions.csv
results/dashcam_original/yolo_tiled_detections.csv
```

## Running the Feedback-Memory Tiled YOLO Experiment

Run:

```bash
python -m src.yolo_tiled_experiment   --source videos/dashcam_footage.mp4   --frames 50   --policy policies/yolo_spatial_policy_feedback.json   --use-feedback
```

Expected output files:

```text
results/dashcam_feedback/yolo_tiled_summary.csv
results/dashcam_feedback/yolo_tiled_spatial_decisions.csv
results/dashcam_feedback/yolo_tiled_detections.csv
```

The feedback-memory policy uses previous YOLO detections to influence future tile selection. If a tile recently produced detections, its feedback score increases. If it stops producing detections, the feedback score decays over time.

Expected qualitative result:

```text
The feedback-memory policy should select recently active tiles more often than the stateless policy.
Detection recovery and reference coverage should remain close to or slightly above the original Spatial Policy.
Latency may vary between runs because tiled YOLO includes preprocessing, batching, GPU scheduling, and remapping overhead.
```

## Running the Strict Feedback-Memory Tiled YOLO Experiment

Run:

```bash
python -m src.yolo_tiled_experiment   --source videos/dashcam_footage.mp4   --frames 50   --policy policies/yolo_spatial_policy_feedback_strict.json   --use-feedback
```

Expected output files:

```text
results/dashcam_feedback_strict/yolo_tiled_summary.csv
results/dashcam_feedback_strict/yolo_tiled_spatial_decisions.csv
results/dashcam_feedback_strict/yolo_tiled_detections.csv
```

This run uses stricter current-frame thresholds. It usually selects fewer tiles than the regular feedback policy, but detection recovery may decrease.

## Running a Strict Policy Without Feedback

Run:

```bash
python -m src.yolo_tiled_experiment   --source videos/dashcam_footage.mp4   --frames 50   --policy policies/yolo_spatial_policy_feedback_strict.json
```

Expected output files:

```text
results/dashcam_strict_no_feedback/yolo_tiled_summary.csv
results/dashcam_strict_no_feedback/yolo_tiled_spatial_decisions.csv
results/dashcam_strict_no_feedback/yolo_tiled_detections.csv
```

This run uses the strict policy file but does not enable feedback state. It is useful as an ablation against the feedback-memory run.

## Running Multi-Resolution Policies

Example using the 640 policy:

```bash
python -m src.yolo_tiled_experiment   --source videos/parking_lot_footage.mp4   --frames 50   --policy policies/yolo_spatial_policy_640.json
```

Expected output files:

```text
results/yolo_tiled_sample2_640/yolo_tiled_summary.csv
results/yolo_tiled_sample2_640/yolo_tiled_spatial_decisions.csv
results/yolo_tiled_sample2_640/yolo_tiled_detections.csv
```

Example using the 960 policy:

```bash
python -m src.yolo_tiled_experiment   --source videos/parking_lot_footage.mp4   --frames 50   --policy policies/yolo_spatial_policy_960.json
```

Expected output files:

```text
results/yolo_tiled_sample2_960/yolo_tiled_summary.csv
results/yolo_tiled_sample2_960/yolo_tiled_spatial_decisions.csv
results/yolo_tiled_sample2_960/yolo_tiled_detections.csv
```

Example using the multi-resolution policy:

```bash
python -m src.yolo_tiled_experiment   --source videos/parking_lot_footage.mp4   --frames 50   --policy policies/yolo_spatial_policy_multi_res.json
```

Expected output files:

```text
results/yolo_tiled_sample2_multi_res/yolo_tiled_summary.csv
results/yolo_tiled_sample2_multi_res/yolo_tiled_spatial_decisions.csv
results/yolo_tiled_sample2_multi_res/yolo_tiled_detections.csv
```

## Running Live Demo Scripts

Run the non-feedback live demo:

```bash
python -m src.yolo_live_demo   --source videos/parking_lot_footage.mp4   --frames 100   --policy policies/yolo_spatial_policy.json   --save results/live_demo/yolo_live_demo_sample2.mp4
```

Run the feedback-memory live demo:

```bash
python -m src.yolo_feedback_live_demo   --source videos/dashcam_footage.mp4   --frames 100   --policy policies/yolo_spatial_policy_feedback.json   --save results/live_demo/dashcam_feedback_demo.mp4
```

Expected output files:

```text
results/live_demo/yolo_live_demo_sample2.mp4
results/live_demo/dashcam_feedback_demo.mp4
```

## Verifying Result Output Locations

After running the experiments, verify that result files are grouped by directory:

```bash
find results -maxdepth 2 -type f | sort
```

Expected directories include:

```text
results/experiment/
results/spatial/
results/yolo/
results/yolo_spatial/
results/dashcam_original/
results/dashcam_feedback/
results/dashcam_feedback_strict/
results/dashcam_strict_no_feedback/
results/yolo_tiled_sample2_640/
results/yolo_tiled_sample2_960/
results/yolo_tiled_sample2_multi_res/
results/live_demo/
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
| Demo outputs | Annotated demo videos or text outputs from live-style runs. |

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
