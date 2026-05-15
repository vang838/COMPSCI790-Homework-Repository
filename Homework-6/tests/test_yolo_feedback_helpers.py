from src.yolo_tiled_experiment import (
    attach_feedback_to_blocks,
    block_id_for_point,
    decay_feedback_scores,
    initialize_feedback_scores,
    update_feedback_from_detections,
)


def test_initialize_feedback_scores_creates_all_block_ids() -> None:
    scores = initialize_feedback_scores(rows=2, cols=3)

    assert len(scores) == 6
    assert scores["r0c0"] == 0.0
    assert scores["r1c2"] == 0.0


def test_block_id_for_point_maps_coordinates_to_grid_cell() -> None:
    assert block_id_for_point(25, 25, frame_width=100, frame_height=100, rows=2, cols=2) == "r0c0"
    assert block_id_for_point(75, 25, frame_width=100, frame_height=100, rows=2, cols=2) == "r0c1"
    assert block_id_for_point(25, 75, frame_width=100, frame_height=100, rows=2, cols=2) == "r1c0"
    assert block_id_for_point(75, 75, frame_width=100, frame_height=100, rows=2, cols=2) == "r1c1"


def test_update_feedback_from_detections_boosts_tile_containing_detection_center() -> None:
    scores = initialize_feedback_scores(rows=2, cols=2)
    detections = [
        {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
        {"x1": 60, "y1": 60, "x2": 90, "y2": 90},
    ]

    update_feedback_from_detections(
        feedback_scores=scores,
        detections=detections,
        frame_width=100,
        frame_height=100,
        rows=2,
        cols=2,
        boost=1.0,
    )

    assert scores["r0c0"] == 1.0
    assert scores["r1c1"] == 1.0
    assert scores["r0c1"] == 0.0
    assert scores["r1c0"] == 0.0


def test_decay_feedback_scores_decays_and_clamps_small_values_to_zero() -> None:
    scores = {
        "r0c0": 1.0,
        "r0c1": 0.06,
    }

    decay_feedback_scores(scores, decay=0.5)

    assert scores["r0c0"] == 0.5
    assert scores["r0c1"] == 0.0


def test_attach_feedback_to_blocks_adds_scores_to_block_dicts() -> None:
    blocks = [
        {"block_id": "r0c0"},
        {"block_id": "r0c1"},
        {"block_id": "r1c0"},
    ]
    scores = {
        "r0c0": 0.12345,
        "r0c1": 0.9,
    }

    attach_feedback_to_blocks(blocks, scores)

    assert blocks[0]["feedback_score"] == 0.1235
    assert blocks[1]["feedback_score"] == 0.9
    assert blocks[2]["feedback_score"] == 0.0
