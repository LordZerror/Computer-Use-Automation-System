from types import SimpleNamespace

import pytest
from playwright.sync_api import sync_playwright

from src.agent import loop as loop_module
from src.agent.llm import AgentLLM, ToolCall


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    p = browser.new_page()
    p.set_content('<button data-test="foo">Click me</button>')
    yield p
    p.close()


def test_a_failed_action_does_not_crash_discovery(page, monkeypatch, tmp_path):
    """Regression: a real Playwright exception mid-action (observed live: a
    Bootstrap modal intercepting a click on automationexercise.com) used to
    propagate all the way out of run_discovery and crash the whole process,
    instead of being surfaced to the model as feedback the way every other
    'this action didn't work' case already is."""
    monkeypatch.setattr("src.evidence.logger.EVIDENCE_ROOT", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")

    calls = iter([
        ToolCall(name="click", arguments={"element_index": 0}),
        ToolCall(name="finish", arguments={"summary": "done"}),
    ])
    monkeypatch.setattr(AgentLLM, "decide", lambda self, *a, **kw: next(calls))

    def _boom(*a, **kw):
        raise RuntimeError("Locator.click: Timeout 30000ms exceeded (modal intercepted pointer events)")

    monkeypatch.setattr(loop_module, "_locator_from_element", lambda page, element: SimpleNamespace(click=_boom))

    # Patch sync_playwright to hand back our already-open, pre-seeded page
    # instead of launching a fresh browser.
    class _FakeBrowser:
        def new_page(self, **kw):
            return page

        def close(self):
            pass

    class _FakePW:
        def __enter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(launch=lambda **kw: _FakeBrowser()))

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(loop_module, "sync_playwright", lambda: _FakePW())
    monkeypatch.setattr(page, "goto", lambda *a, **kw: None)  # keep the pre-seeded content, no real navigation

    result = loop_module.run_discovery(goal="click the button", target_url="https://www.saucedemo.com/", run_id="robustness1")

    assert result.success is True  # reached `finish` despite the failed click
    assert result.transcript == []  # the failed click was never recorded as a step


def test_repeating_the_identical_failing_action_stops_instead_of_looping(page, monkeypatch, tmp_path):
    """Regression: observed live against automationexercise.com -- the model
    ignored 'that action failed, try something else' and retried the exact
    same click 8 times in a row, burning the whole step budget. A headless
    run (no human to escalate to) should give up as a dead end well before
    max_steps, not loop until the budget runs out."""
    monkeypatch.setattr("src.evidence.logger.EVIDENCE_ROOT", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")

    decide_calls = []

    def _always_same_click(self, *a, **kw):
        decide_calls.append(1)
        return ToolCall(name="click", arguments={"element_index": 0})

    monkeypatch.setattr(AgentLLM, "decide", _always_same_click)
    monkeypatch.setattr(
        loop_module, "_locator_from_element",
        lambda page, element: SimpleNamespace(click=lambda: (_ for _ in ()).throw(RuntimeError("boom"))),
    )

    class _FakeBrowser:
        def new_page(self, **kw):
            return page

        def close(self):
            pass

    class _FakePW:
        def __enter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(launch=lambda **kw: _FakeBrowser()))

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(loop_module, "sync_playwright", lambda: _FakePW())
    monkeypatch.setattr(page, "goto", lambda *a, **kw: None)

    result = loop_module.run_discovery(
        goal="click the button", target_url="https://www.saucedemo.com/",
        run_id="robustness2", max_steps=10, headless=True,
    )

    assert result.success is False
    assert len(decide_calls) < 10  # stopped well short of max_steps
