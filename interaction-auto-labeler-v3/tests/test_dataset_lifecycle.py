import json
from pathlib import Path

from interaction_auto_labeler_v3.dataset_lifecycle import (
    build_dataset_version,
    official_v3_conversion_command,
    validate_lerobot_v3,
)


def test_build_version_and_validate_v3(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta" / "episodes").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "cam_h").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": {}}), encoding="utf-8"
    )
    (root / "meta" / "episodes" / "file-000.parquet").write_bytes(b"episode")
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"data")
    output = tmp_path / "versions" / "v1.json"
    manifest = build_dataset_version(root, "demo", "v1", output)
    assert manifest.layout == "lerobot_v3"
    assert manifest.metadata["file_count"] == 3
    assert validate_lerobot_v3(root)["valid"] is True


def test_official_converter_is_explicit_and_dry_run_safe() -> None:
    command = official_v3_conversion_command("org/demo", python=Path("python"))
    assert command[-2:] == ["--repo-id", "org/demo"]
