import numpy as np

from interaction_auto_labeler_v3.anomaly import detect_anomalies, quality_gate


def test_anomaly_detector_finds_timestamp_and_depth_failures() -> None:
    frames = 30
    timestamps = np.arange(frames, dtype=float) / 30
    timestamps[15:] += 0.5
    depth_ratio = np.ones(frames)
    depth_ratio[20:24] = 0.05
    anomalies = detect_anomalies(
        {"joint": np.zeros((frames, 2)), "depth_valid_ratio": depth_ratio},
        timestamps,
    )
    kinds = {item.kind for item in anomalies}
    assert "timestamp_discontinuity" in kinds
    assert "depth_dropout" in kinds
    gate = quality_gate(0.9, 0.9, 0.9, anomalies)
    assert gate["decision"] in {"review", "reject"}
