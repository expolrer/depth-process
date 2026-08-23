import numpy as np

from training_integrations.target_annotations import (
    box_to_patch_target,
    normalize_box,
    phase_for_frame,
)


def test_normalize_and_patch_target() -> None:
    box = normalize_box([25, 10, 75, 30], 100, 40)
    assert box == (0.25, 0.25, 0.75, 0.75)
    target = box_to_patch_target(box, 4, 4)
    np.testing.assert_array_equal(
        target,
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )


def test_action_phase_labels() -> None:
    assert phase_for_frame(4, 0, 5, 10, 12) == "approach"
    assert phase_for_frame(5, 0, 5, 10, 12) == "contact"
    assert phase_for_frame(8, 0, 5, 10, 12) == "transport"
    assert phase_for_frame(11, 0, 5, 10, 12) == "post_release"
