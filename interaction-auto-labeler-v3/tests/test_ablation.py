import json

from interaction_auto_labeler_v3.ablation import (
    AblationSpec,
    execute_bundle,
    paired_bootstrap_difference,
    write_ablation_bundle,
)


def test_ablation_bundle_is_paired_and_dry_run_by_default(tmp_path) -> None:
    spec = AblationSpec(
        experiment_id="toy-act",
        framework="act",
        dataset="/data/toy",
        output_root=str(tmp_path / "runs"),
        base_config="act.yaml",
        train_entrypoint=("python", "train.py", "--config", "{config}"),
        split_manifest="split.json",
        steps=50,
    )
    report = write_ablation_bundle(spec, tmp_path / "bundle")
    assert report["runs"] == 2
    runs = json.loads((tmp_path / "bundle" / "run_matrix.json").read_text())["runs"]
    assert {row["seed"] for row in runs} == {0}
    assert all(not row["fairness_contract"]["future_derived_roi_is_input"] for row in runs)
    assert all(item["state"] == "dry_run" for item in execute_bundle(tmp_path / "bundle"))


def test_paired_bootstrap_reports_positive_improvement() -> None:
    report = paired_bootstrap_difference([1.0, 1.1, 0.9], [0.8, 0.9, 0.7], False, samples=500)
    assert report["mean_improvement"] > 0
    assert report["probability_of_improvement"] > 0.95
