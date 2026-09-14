#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ACT0 is byte-for-byte model-construction compatible")
    parser.add_argument(
        "--official-act-root",
        type=Path,
        default=Path("/ssd/hhw/depth-model/repos/RoboTwin/policy/ACT"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(args.official_act_root))

    from detr.models import build_ACT_model
    from official_act_rgbd.policy import DEFAULT_MODEL_CONFIG, as_namespace, build_official_model

    lock = json.loads((project / "UPSTREAM_LOCK.json").read_text(encoding="utf-8"))
    hashes = {}
    hash_ok = True
    for relative, expected in lock["files"].items():
        actual = sha256(args.official_act_root / relative)
        hashes[relative] = {"expected": expected, "actual": actual, "match": actual == expected}
        hash_ok = hash_ok and actual == expected

    model_args = as_namespace(DEFAULT_MODEL_CONFIG)
    torch.manual_seed(20260914)
    upstream = build_ACT_model(model_args)
    torch.manual_seed(20260914)
    candidate = build_official_model(model_args, "ACT0_RGB")
    upstream_state = upstream.state_dict()
    candidate_state = candidate.state_dict()
    keys_match = tuple(upstream_state) == tuple(candidate_state)
    mismatches = []
    if keys_match:
        for key in upstream_state:
            left = upstream_state[key]
            right = candidate_state[key]
            if left.shape != right.shape or not torch.equal(left, right):
                mismatches.append(key)
                if len(mismatches) >= 20:
                    break

    report = {
        "passed": hash_ok and keys_match and not mismatches,
        "upstream_hashes_match": hash_ok,
        "state_dict_keys_match": keys_match,
        "parameter_values_match": not mismatches,
        "mismatches": mismatches,
        "upstream_trainable_parameters": sum(p.numel() for p in upstream.parameters() if p.requires_grad),
        "candidate_trainable_parameters": sum(p.numel() for p in candidate.parameters() if p.requires_grad),
        "hashes": hashes,
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

