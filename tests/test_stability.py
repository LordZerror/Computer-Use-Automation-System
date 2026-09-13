from types import SimpleNamespace

from src.replay import stability
from src.replay.errors import ReplayResult


def _capability_stub():
    # measure_stability only reads .id off the capability before delegating
    # to replay(), which we stub out entirely below.
    return SimpleNamespace(id="x")


def test_all_consistent_successes_score_1(monkeypatch):
    monkeypatch.setattr(stability, "replay", lambda *a, **kw: ReplayResult(status="success"))
    result = stability.measure_stability(_capability_stub(), {}, runs=5)
    assert result == {"score": 1.0, "runs": 5, "status_counts": {"success": 5}}


def test_consistent_business_outcome_is_stable_not_flaky(monkeypatch):
    """A capability that always reports the same legitimate business outcome
    for a given input is stable -- flakiness is about disagreement between
    runs, not about whether the outcome was 'success'."""
    monkeypatch.setattr(stability, "replay", lambda *a, **kw: ReplayResult(status="business_outcome", outcome_code="not_found"))
    result = stability.measure_stability(_capability_stub(), {}, runs=5)
    assert result["score"] == 1.0


def test_mixed_outcomes_score_below_1(monkeypatch):
    outcomes = iter(["success", "success", "success", "failure", "success"])
    monkeypatch.setattr(stability, "replay", lambda *a, **kw: ReplayResult(status=next(outcomes)))
    result = stability.measure_stability(_capability_stub(), {}, runs=5)
    assert result["score"] == 0.8
    assert result["status_counts"] == {"success": 4, "failure": 1}
