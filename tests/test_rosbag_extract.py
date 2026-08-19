from depth_pipeline.rosbag_extract import CompressedFrame, nearest_pairs


def frame(timestamp: int) -> CompressedFrame:
    return CompressedFrame(timestamp, timestamp, "frame", b"", "test")


def test_nearest_pairs_uses_bag_timestamp() -> None:
    colors = [frame(100), frame(200), frame(300)]
    depths = [frame(91), frame(214), frame(500)]
    pairs, deltas = nearest_pairs(colors, depths, max_delta_ns=20)
    assert [(color.bag_timestamp_ns, depth.bag_timestamp_ns) for color, depth in pairs] == [
        (100, 91),
        (200, 214),
    ]
    assert deltas == [9, -14]
