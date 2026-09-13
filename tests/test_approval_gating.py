import pytest

from src.artifact.schema import Capability, Checkpoint, TargetSpec
from src.replay import executor


@pytest.fixture(autouse=True)
def _isolate_evidence(tmp_path, monkeypatch):
    # replay() always creates an EvidenceLogger and logs replay_start before
    # the approval gate runs -- keep that out of the project's real evidence/.
    monkeypatch.setattr("src.evidence.logger.EVIDENCE_ROOT", tmp_path)


def _capability(status: str) -> Capability:
    return Capability(
        id="x", name="x", goal="g",
        target=TargetSpec(base_url="https://www.saucedemo.com/", app_id="saucedemo"),
        steps=[],
        checkpoint=Checkpoint(kind="text_present", text="whatever"),
        created_from_run_id="r",
        status=status,
    )


def test_draft_capability_is_rejected_before_any_browser_is_launched(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("sync_playwright should never be called for an unapproved capability")

    monkeypatch.setattr(executor, "sync_playwright", _boom)

    result = executor.replay(_capability("draft"), {}, run_id="t1", require_approved=True)

    assert result.status == "failure"
    assert result.error.step_id == "approval"


def test_approved_capability_proceeds_past_the_gate(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("reached sync_playwright, as expected")

    monkeypatch.setattr(executor, "sync_playwright", _boom)

    with pytest.raises(RuntimeError, match="reached sync_playwright"):
        executor.replay(_capability("approved"), {}, run_id="t2", require_approved=True)


def test_require_approved_false_ignores_status(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("reached sync_playwright, as expected")

    monkeypatch.setattr(executor, "sync_playwright", _boom)

    with pytest.raises(RuntimeError, match="reached sync_playwright"):
        executor.replay(_capability("draft"), {}, run_id="t3", require_approved=False)
