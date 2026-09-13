"""Confidence & approval gating, part 1 (Section 8 stretch goal): measure how
consistently an artifact replays.

Stability is about *consistency*, not success rate: an artifact that always
returns the same `business_outcome` for a given bad input is stable, not
flaky -- that's the whole point of the three-way result contract in
replay/errors.py. Flakiness is when repeated, identical-input runs disagree
with each other.
"""
from __future__ import annotations

import uuid
from collections import Counter

from src.artifact.schema import Capability
from src.replay.executor import replay


def measure_stability(capability: Capability, params: dict[str, str], runs: int = 5) -> dict:
    """Replay `capability` `runs` times with identical params (headless, no
    escalation/assisted-fallback -- this measures the deterministic path
    only) and report how often the most common outcome recurred."""
    counts: Counter[str] = Counter()
    for _ in range(runs):
        result = replay(capability, params, run_id=f"stability-{uuid.uuid4().hex[:8]}", headless=True)
        counts[result.status] += 1

    most_common_count = counts.most_common(1)[0][1] if counts else 0
    return {
        "score": most_common_count / runs if runs else 0.0,
        "runs": runs,
        "status_counts": dict(counts),
    }
