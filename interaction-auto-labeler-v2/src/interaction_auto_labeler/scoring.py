from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schema import EvidenceWeights

EVIDENCE_FIELDS = tuple(asdict(EvidenceWeights()).keys())


def _unit(value: Any) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def score_candidate(
    evidence: dict[str, Any],
    weights: EvidenceWeights | None = None,
) -> dict[str, Any]:
    weights = weights or EvidenceWeights()
    weight_values = asdict(weights)
    available = {field: _unit(evidence.get(field)) for field in EVIDENCE_FIELDS}
    denominator = sum(
        weight_values[field] for field, value in available.items() if value is not None
    )
    score = (
        sum(weight_values[field] * value for field, value in available.items() if value is not None)
        / denominator
        if denominator
        else 0.0
    )
    return {
        **evidence,
        "evidence": available,
        "available_weight": denominator,
        "interaction_instance_score": float(score),
    }


def rank_candidates(
    candidates: list[dict[str, Any]],
    weights: EvidenceWeights | None = None,
    ambiguity_margin: float = 0.08,
) -> dict[str, Any]:
    ranked = sorted(
        (score_candidate(candidate, weights) for candidate in candidates),
        key=lambda row: row["interaction_instance_score"],
        reverse=True,
    )
    top = ranked[0]["interaction_instance_score"] if ranked else 0.0
    second = ranked[1]["interaction_instance_score"] if len(ranked) > 1 else 0.0
    margin = top - second
    return {
        "ranked_candidates": ranked,
        "selected_instance_id": ranked[0].get("instance_id") if ranked else None,
        "score": top,
        "margin": margin,
        "needs_review": not ranked or margin < ambiguity_margin,
    }
