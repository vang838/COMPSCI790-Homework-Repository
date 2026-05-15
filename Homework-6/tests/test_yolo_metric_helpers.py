import numpy as np
import pytest

from src.yolo_spatial_experiment import (
    compute_detection_recovery,
    compute_iou,
    compute_reference_coverage,
    normalize,
    split_frame_into_blocks,
)


def test_normalize_handles_identical_values() -> None:
    assert normalize([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_normalize_scales_min_to_zero_and_max_to_one() -> None:
    assert normalize([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_split_frame_into_blocks_creates_expected_grid_metadata() -> None:
    frame = np.zeros((90, 90, 3), dtype=np.uint8)

    blocks = split_frame_into_blocks(
        frame=frame,
        previous_gray=None,
        rows=3,
        cols=3,
    )

    assert len(blocks) == 9
    assert blocks[0]["block_id"] == "r0c0"
    assert blocks[-1]["block_id"] == "r2c2"
    assert blocks[0]["crop"].shape == (30, 30, 3)
    assert blocks[0]["x1"] == 0
    assert blocks[0]["y1"] == 0
    assert blocks[-1]["x2"] == 90
    assert blocks[-1]["y2"] == 90


def test_compute_iou_for_partially_overlapping_boxes() -> None:
    a = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
    b = {"x1": 5, "y1": 5, "x2": 15, "y2": 15}

    assert compute_iou(a, b) == pytest.approx(25 / 175)


def test_detection_recovery_matches_same_class_with_enough_iou() -> None:
    reference = [
        {"class_id": 2, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
    ]
    spatial = [
        {"class_id": 2, "x1": 1, "y1": 1, "x2": 11, "y2": 11},
    ]

    assert compute_detection_recovery(reference, spatial, iou_threshold=0.30) == 1.0


def test_detection_recovery_rejects_wrong_class_even_with_overlap() -> None:
    reference = [
        {"class_id": 2, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
    ]
    spatial = [
        {"class_id": 3, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
    ]

    assert compute_detection_recovery(reference, spatial, iou_threshold=0.30) == 0.0


def test_reference_coverage_counts_detection_centers_inside_selected_blocks() -> None:
    reference = [
        {"x1": 10, "y1": 10, "x2": 20, "y2": 20},
        {"x1": 70, "y1": 70, "x2": 80, "y2": 80},
    ]
    selected_blocks = [
        {"x1": 0, "y1": 0, "x2": 45, "y2": 45},
    ]

    assert compute_reference_coverage(reference, selected_blocks) == 0.5
