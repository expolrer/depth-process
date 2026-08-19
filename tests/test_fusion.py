import numpy as np

from depth_pipeline.fusion import calibrate_relative_inverse_depth, fuse_sensor_and_prior


def test_inverse_prior_is_calibrated_to_metric_depth() -> None:
    y, x = np.mgrid[:40, :50]
    sensor = (0.8 + x * 0.02 + y * 0.01).astype(np.float32)
    prior = ((1.0 / sensor) - 0.3) / 1.7
    sensor_with_hole = sensor.copy()
    sensor_with_hole[10:20, 15:25] = 0
    calibrated, report = calibrate_relative_inverse_depth(prior, sensor_with_hole)
    fused, mask = fuse_sensor_and_prior(sensor_with_hole, calibrated)
    assert report["calibration_ok"]
    assert np.max(np.abs(fused - sensor)) < 1e-4
    assert mask.sum() == 100

