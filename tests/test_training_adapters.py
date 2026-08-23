import numpy as np

from training_integrations.act_adapter import add_act_roi_supervision, roi_coordinate_token
from training_integrations.openpi_adapter import OpenPiTargetCropTransform
from training_integrations.target_annotations import AnnotationIndex, TargetAnnotation


def annotation() -> TargetAnnotation:
    return TargetAnnotation(
        episode_index=1,
        frame_index=2,
        camera="cam_h",
        bbox_xyxy=(2, 2, 6, 6),
        bbox_normalized=(0.25, 0.25, 0.75, 0.75),
        visible=True,
    )


def test_act_adapter_adds_auxiliary_target() -> None:
    sample = add_act_roi_supervision({}, annotation(), "cam_h", (4, 4))
    assert sample["observation.target_roi.cam_h.valid"] == 1.0
    assert sample["observation.target_roi.cam_h.patch_target"].sum() == 4
    np.testing.assert_allclose(roi_coordinate_token(annotation()), [0.5, 0.5, 0.5, 0.5, 1.0])


def test_openpi_transform_adds_crop_and_dropout_contract() -> None:
    index = AnnotationIndex(
        [
            {
                "episode_index": 1,
                "frame_index": 2,
                "targets": {
                    "cam_h": {
                        "bbox_xyxy": [2, 2, 6, 6],
                        "bbox_normalized": [0.25, 0.25, 0.75, 0.75],
                        "visible": True,
                    }
                },
            }
        ]
    )
    transform = OpenPiTargetCropTransform(
        index, ("cam_h",), output_size=(8, 8), dropout_probability=0.0
    )
    sample = transform(
        {
            "episode_index": 1,
            "frame_index": 2,
            "observation.images.cam_h": np.zeros((10, 10, 3), dtype=np.uint8),
        }
    )
    assert sample["observation.images.cam_h_target_crop"].shape == (8, 8, 3)
    assert sample["observation.target_roi.cam_h.valid"] == 1.0
