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


def test_discovery_dispatches_select_option_hover_and_keypress(page, monkeypatch, tmp_path):
    """select_option/hover/keypress are generalization additions to the
    click/type/navigate action set -- exercise all three end to end through
    the real dispatch and perception (only the LLM decision is mocked)."""
    monkeypatch.setattr("src.evidence.logger.EVIDENCE_ROOT", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")

    page.set_content(
        '<select id="country"><option>USA</option><option>Canada</option></select>'
        '<div id="trigger" role="button">Account</div>'
        '<div id="menu">Settings</div>'
        '<style>#menu{display:none}#trigger:hover + #menu{display:block}</style>'
        '<input id="q">'
        '<div id="status"></div>'
        '<script>document.getElementById("q").addEventListener("keydown", e => {'
        'if (e.key === "Enter") document.getElementById("status").textContent = "submitted";'
        '});</script>'
    )
    # element order = document order: select(0), trigger(1), input(2)
    calls = iter([
        ToolCall(name="select_option", arguments={"element_index": 0, "value": "Canada"}),
        ToolCall(name="hover", arguments={"element_index": 1}),
        ToolCall(name="keypress", arguments={"element_index": 2, "key": "Enter"}),
        ToolCall(name="finish", arguments={"summary": "done"}),
    ])
    monkeypatch.setattr(AgentLLM, "decide", lambda self, *a, **kw: next(calls))

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

    result = loop_module.run_discovery(goal="fill out the form", target_url="https://www.saucedemo.com/", run_id="actions1")

    assert result.success is True
    actions = [t["action"] for t in result.transcript]
    assert actions == ["select_option", "hover", "keypress"]
    assert page.eval_on_selector("#country", "el => el.selectedOptions[0].text") == "Canada"
    assert page.locator("#menu").is_visible()
    assert page.locator("#status").inner_text() == "submitted"


def test_select_option_is_blocked_when_the_chosen_value_is_risky(page, monkeypatch, tmp_path):
    """Regression: risk classification only looked at the <select>'s own
    accessible name, so a neutrally-named dropdown ("Account Actions") with
    a destructive option ("Delete") sailed through unblocked even though the
    identical text on a button would be blocked -- policy must also check
    the option actually being chosen, not just the control's label. The
    aria-label makes the "neutrally-named control, risky option" scenario
    concrete and readable rather than relying on an unlabeled <select>'s
    (now-empty, see test_perceive.py) accessible name being blank."""
    monkeypatch.setattr("src.evidence.logger.EVIDENCE_ROOT", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")

    page.set_content(
        '<select id="actions" aria-label="Account Actions">'
        '<option>View Profile</option><option>Delete</option></select>'
    )
    monkeypatch.setattr(
        AgentLLM, "decide",
        lambda self, *a, **kw: ToolCall(name="select_option", arguments={"element_index": 0, "value": "Delete"}),
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
        goal="pick an action", target_url="https://www.saucedemo.com/",
        run_id="risk1", max_steps=3, headless=True,
    )

    assert result.transcript == []  # blocked, never executed or recorded
    assert page.eval_on_selector("#actions", "el => el.selectedOptions[0].text") == "View Profile"  # unchanged


def test_keypress_is_blocked_when_the_key_itself_is_risky(page, monkeypatch, tmp_path):
    """Regression: keypress risk classification only looked at the focused
    element's own name -- a neutrally-named element ("row-42") never got
    blocked no matter what key was sent, even a key ("Delete") that would
    have blocked an identically-named click."""
    monkeypatch.setattr("src.evidence.logger.EVIDENCE_ROOT", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")

    page.set_content(
        '<div id="row" role="textbox" aria-label="row-42" '
        'onkeydown="if(event.key===\'Delete\') this.setAttribute(\'data-deleted\',\'1\')">item</div>'
    )
    monkeypatch.setattr(
        AgentLLM, "decide",
        lambda self, *a, **kw: ToolCall(name="keypress", arguments={"element_index": 0, "key": "Delete"}),
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
        goal="delete the row", target_url="https://www.saucedemo.com/",
        run_id="risk2", max_steps=3, headless=True,
    )

    assert result.transcript == []  # blocked, never executed or recorded
    assert page.locator("#row").get_attribute("data-deleted") is None


def test_select_option_on_a_non_native_select_fails_clearly_not_silently(page, monkeypatch, tmp_path):
    """A JS-built listbox widget (<div role="combobox">) gets the same
    perceived role as a real <select> but has no `options`, and
    Playwright's .select_option() only works on a real <select> -- this
    should be caught before even trying, with an actionable message the
    model can act on, not left to throw a generic Playwright error."""
    monkeypatch.setattr("src.evidence.logger.EVIDENCE_ROOT", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")

    page.set_content('<div role="combobox" aria-label="Country">React-select widget</div>')
    monkeypatch.setattr(
        AgentLLM, "decide",
        lambda self, *a, **kw: ToolCall(name="select_option", arguments={"element_index": 0, "value": "Canada"}),
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
        goal="pick a country", target_url="https://www.saucedemo.com/",
        run_id="combobox1", max_steps=3, headless=True,
    )

    assert result.transcript == []  # never executed, no crash
