from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json


@dataclass(frozen=True)
class AblationVariant:
    name: str
    roi_auxiliary_loss: bool = False
    target_crop_input: bool = False
    roi_token_input: bool = False
    roi_dropout: float = 1.0
    inference_detector_required: bool = False


DEFAULT_VARIANTS = (
    AblationVariant("A_baseline"),
    AblationVariant("B_roi_aux", roi_auxiliary_loss=True),
    AblationVariant(
        "C_target_crop",
        roi_auxiliary_loss=True,
        target_crop_input=True,
        roi_dropout=0.4,
        inference_detector_required=True,
    ),
    AblationVariant(
        "D_roi_token",
        roi_auxiliary_loss=True,
        target_crop_input=True,
        roi_token_input=True,
        roi_dropout=0.4,
        inference_detector_required=True,
    ),
)


@dataclass(frozen=True)
class AblationSpec:
    experiment_id: str
    framework: str
    dataset: str
    output_root: str
    base_config: str
    train_entrypoint: tuple[str, ...]
    seed: int = 0
    steps: int = 5000
    gpu_ids: tuple[int, ...] = ()
    variants: tuple[AblationVariant, ...] = DEFAULT_VARIANTS[:2]
    metrics: tuple[str, ...] = (
        "validation_action_mae",
        "roi_iou",
        "task_success_rate",
    )
    split_manifest: str = ""
    schema: str = "vla_annotation_ablation_v1"

    def __post_init__(self) -> None:
        if self.framework not in {"act", "openpi_pi05"}:
            raise ValueError("framework must be act or openpi_pi05")
        if self.steps <= 0 or not self.train_entrypoint:
            raise ValueError("steps and train_entrypoint are required")
        if not self.split_manifest:
            raise ValueError("an episode-level split_manifest is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_matrix(spec: AblationSpec) -> list[dict[str, Any]]:
    rows = []
    for variant in spec.variants:
        output = str(Path(spec.output_root) / spec.experiment_id / variant.name)
        rows.append(
            {
                "run_id": f"{spec.experiment_id}:{variant.name}",
                "framework": spec.framework,
                "dataset": spec.dataset,
                "base_config": spec.base_config,
                "split_manifest": spec.split_manifest,
                "output_dir": output,
                "seed": spec.seed,
                "steps": spec.steps,
                "gpu_ids": list(spec.gpu_ids),
                "variant": asdict(variant),
                "fairness_contract": {
                    "same_episode_split": True,
                    "same_initial_checkpoint": True,
                    "same_optimizer_budget": True,
                    "same_augmentation_seed": True,
                    "future_derived_roi_is_input": False,
                },
            }
        )
    return rows


def write_ablation_bundle(spec: AblationSpec, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    matrix = build_run_matrix(spec)
    atomic_write_json(output / "spec.json", spec.to_dict())
    atomic_write_json(output / "run_matrix.json", {"runs": matrix})
    commands = []
    for row in matrix:
        config_path = output / f"{row['variant']['name']}.json"
        atomic_write_json(config_path, row)
        command = [
            token.format(config=str(config_path.resolve()), output=row["output_dir"])
            for token in spec.train_entrypoint
        ]
        commands.append({"run_id": row["run_id"], "argv": command})
    atomic_write_json(output / "commands.json", {"commands": commands})
    shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for item in commands:
        shell_lines.append(" ".join(shlex.quote(token) for token in item["argv"]))
    (output / "run_all.sh").write_text("\n".join(shell_lines) + "\n", encoding="utf-8")
    return {"runs": len(matrix), "bundle": str(output.resolve()), "commands": commands}


def execute_bundle(bundle: Path, dry_run: bool = True) -> list[dict[str, Any]]:
    commands = json.loads((bundle / "commands.json").read_text(encoding="utf-8"))["commands"]
    results = []
    for item in commands:
        if dry_run:
            results.append({**item, "state": "dry_run", "returncode": None})
            continue
        completed = subprocess.run(item["argv"], text=True, capture_output=True, check=False)
        result = {
            "run_id": item["run_id"],
            "argv": item["argv"],
            "state": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        results.append(result)
        if completed.returncode:
            break
    return results


def paired_bootstrap_difference(
    baseline: Iterable[float],
    treatment: Iterable[float],
    higher_is_better: bool,
    seed: int = 0,
    samples: int = 5000,
) -> dict[str, Any]:
    baseline = np.asarray(list(baseline), dtype=np.float64)
    treatment = np.asarray(list(treatment), dtype=np.float64)
    if baseline.shape != treatment.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("paired metric arrays must have equal non-zero length")
    difference = treatment - baseline if higher_is_better else baseline - treatment
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        "paired_episodes": len(difference),
        "mean_improvement": float(difference.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "probability_of_improvement": float(np.mean(means > 0)),
        "higher_is_better": higher_is_better,
    }
