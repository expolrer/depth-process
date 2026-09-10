import json

from interaction_auto_labeler_v3.sharding import iter_jsonl, shard_for_episode, write_episode_shards


def test_episode_sharding_is_stable_and_streaming(tmp_path) -> None:
    rows = ({"episode_id": str(index), "value": index} for index in range(100))
    report = write_episode_shards(rows, tmp_path / "shards", shard_count=8)
    assert report["record_count"] == 100
    assert shard_for_episode("42", 8) == shard_for_episode("42", 8)
    recovered = []
    for path in (tmp_path / "shards").glob("part-*.jsonl"):
        recovered.extend(iter_jsonl(path))
    assert sorted(row["value"] for row in recovered) == list(range(100))
    manifest = json.loads((tmp_path / "shards" / "manifest.json").read_text())
    assert sum(manifest["counts"]) == 100
