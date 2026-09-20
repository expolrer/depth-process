import numpy as np

from depth_pipeline.cdm import (
    fuse_sensor_and_cdm,
    make_inverse_depth_prompt,
    sanitize_sensor_depth,
)
from scripts.process_cdm import infer_depth


def test_cdm_prompt_and_sensor_fusion_preserve_valid_measurements() -> None:
    raw = np.array([[0.20, 0.0, np.nan], [0.06, 0.45, 0.80]], dtype=np.float32)
    sensor, valid = sanitize_sensor_depth(raw, min_depth_m=0.07, max_depth_m=0.50)
    inverse = make_inverse_depth_prompt(sensor)
    assert valid.tolist() == [[True, False, False], [False, True, False]]
    assert np.isclose(inverse[0, 0], 5.0)
    assert inverse[0, 1] == 0.0

    cdm = np.array([[0.25, 0.30, 0.40], [0.10, 0.35, 0.60]], dtype=np.float32)
    fused, fill_mask, sensor_valid = fuse_sensor_and_cdm(
        sensor,
        cdm,
        min_depth_m=0.07,
        max_depth_m=0.50,
    )
    assert np.array_equal(sensor_valid, valid)
    assert fused[0, 0] == sensor[0, 0]
    assert fused[1, 1] == sensor[1, 1]
    assert np.isclose(fused[0, 1], 0.30)
    assert np.isclose(fused[0, 2], 0.40)
    assert np.isclose(fused[1, 0], 0.10)
    assert fused[1, 2] == 0.0
    assert fill_mask.sum() == 3


def test_official_cdm_inverse_prediction_is_converted_to_metric_depth() -> None:
    class DummyModel:
        def infer_image(self, rgb, inverse, input_size):
            assert input_size == 518
            assert rgb[0, 0].tolist() == [3, 2, 1]
            assert np.isclose(inverse[0, 0], 4.0)
            return np.array([[5.0, 0.0], [2.0, np.nan]], dtype=np.float32)

    rgb_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_bgr[0, 0] = [1, 2, 3]
    sensor = np.array([[0.25, 0.0], [0.50, 0.0]], dtype=np.float32)

    metric = infer_depth(DummyModel(), rgb_bgr, sensor, input_size=518)

    np.testing.assert_allclose(metric, [[0.20, 0.0], [0.50, 0.0]])
