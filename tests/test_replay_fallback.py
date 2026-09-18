import json
from types import SimpleNamespace

import pytest
from playwright.sync_api import sync_playwright

from src.artifact.schema import LocatorStrategy, Step
from src.replay import fallback
from src.replay.errors import HardFailure
from src.safety.policy import load_policy


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    p = browser.new_page()
    yield p
    p.close()


@pytest.fixture()
def logger(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evidence.logger.EVIDENCE_ROOT", tmp_path)
    from src.evidence.logger import EvidenceLogger

    return EvidenceLogger("fallback_unit", "test")


def _fake_groq(monkeypatch, tool_name: str, arguments: dict):
    """Stub the Groq client so no real API call happens in tests."""
    call = SimpleNamespace(function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments)))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    monkeypatch.setattr(fallback, "Groq", lambda api_key: fake_client)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")


def _broken_step(action="click", param_ref=None, value=None, description="click add to cart") -> Step:
    return Step(
        step_id="s0", action=action, value=value, param_ref=param_ref,
        locators=[LocatorStrategy(kind="test_id", test_id="this-id-does-not-exist")],
        description=description,
    )


def test_recovers_by_picking_the_matching_element(page, logger, monkeypatch):
    page.set_content('<button data-test="foo" onclick="this.setAttribute(\'data-clicked\',\'1\')">Add to cart</button>')
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    fallback.recover(page, _broken_step(), {}, load_policy(), logger)

    assert page.locator('[data-test="foo"]').get_attribute("data-clicked") == "1"


def test_none_found_raises_hard_failure(page, logger, monkeypatch):
    page.set_content('<button data-test="foo">Add to cart</button>')
    _fake_groq(monkeypatch, "none_found", {})

    with pytest.raises(HardFailure):
        fallback.recover(page, _broken_step(), {}, load_policy(), logger)


def test_invalid_index_raises_hard_failure(page, logger, monkeypatch):
    page.set_content('<button data-test="foo">Add to cart</button>')
    _fake_groq(monkeypatch, "pick_element", {"element_index": 99})

    with pytest.raises(HardFailure):
        fallback.recover(page, _broken_step(), {}, load_policy(), logger)


def test_recovered_risky_element_is_blocked_not_executed(page, logger, monkeypatch):
    page.set_content('<button data-test="foo" onclick="this.setAttribute(\'data-clicked\',\'1\')">Finish</button>')
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    with pytest.raises(HardFailure):
        fallback.recover(page, _broken_step(description="click finish"), {}, load_policy(), logger)

    assert page.locator('[data-test="foo"]').get_attribute("data-clicked") is None


def test_type_action_uses_param_ref_value(page, logger, monkeypatch):
    page.set_content('<input data-test="foo" />')
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    fallback.recover(page, _broken_step(action="type", param_ref="zip_code", description="type zip"), {"zip_code": "94107"}, load_policy(), logger)

    assert page.locator('[data-test="foo"]').input_value() == "94107"


def test_select_option_action_recovers(page, logger, monkeypatch):
    page.set_content('<select data-test="foo"><option>USA</option><option>Canada</option></select>')
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    fallback.recover(page, _broken_step(action="select_option", value="Canada", description="pick country"), {}, load_policy(), logger)

    assert page.eval_on_selector('[data-test="foo"]', "el => el.selectedOptions[0].text") == "Canada"


def test_hover_action_recovers(page, logger, monkeypatch):
    page.set_content(
        '<style>#menu{display:none}[data-test="foo"]:hover + #menu{display:block}</style>'
        '<div data-test="foo">Account</div><div id="menu">Settings</div>'
    )
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    fallback.recover(page, _broken_step(action="hover", description="hover account menu"), {}, load_policy(), logger)

    assert page.locator("#menu").is_visible()


def test_keypress_action_recovers(page, logger, monkeypatch):
    page.set_content(
        '<input data-test="foo"><div id="status"></div>'
        '<script>document.querySelector(\'[data-test="foo"]\').addEventListener("keydown", e => {'
        'if (e.key === "Enter") document.getElementById("status").textContent = "submitted";'
        '});</script>'
    )
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    fallback.recover(page, _broken_step(action="keypress", value="Enter", description="submit search"), {}, load_policy(), logger)

    assert page.locator("#status").inner_text() == "submitted"


def test_hover_recovery_blocked_when_the_control_name_is_risky(page, logger, monkeypatch):
    """hover goes through the same risky_text_markers gate as every other
    action -- a hover-triggered control whose name matches a marker (e.g.
    "Delete Options") is blocked like a click on the same name would be."""
    page.set_content(
        '<style>#menu{display:none}[data-test="foo"]:hover + #menu{display:block}</style>'
        '<div data-test="foo">Delete Options</div><div id="menu">Settings</div>'
    )
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    with pytest.raises(HardFailure):
        fallback.recover(page, _broken_step(action="hover", description="hover delete options menu"), {}, load_policy(), logger)

    assert not page.locator("#menu").is_visible()


def test_select_option_recovery_blocked_when_chosen_value_is_risky(page, logger, monkeypatch):
    """Same fix as agent/loop.py's _classify_and_block_if_risky: a
    neutrally-named dropdown with a risky option must still be blocked."""
    page.set_content(
        '<select data-test="foo" aria-label="Account Actions">'
        '<option>View Profile</option><option>Delete</option></select>'
    )
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    with pytest.raises(HardFailure):
        fallback.recover(page, _broken_step(action="select_option", value="Delete", description="pick action"), {}, load_policy(), logger)

    assert page.eval_on_selector('[data-test="foo"]', "el => el.selectedOptions[0].text") == "View Profile"


def test_select_option_recovery_fails_clearly_on_a_non_native_select(page, logger, monkeypatch):
    page.set_content('<div data-test="foo" role="combobox" aria-label="Country">React-select widget</div>')
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    with pytest.raises(HardFailure):
        fallback.recover(page, _broken_step(action="select_option", value="Canada", description="pick country"), {}, load_policy(), logger)


def test_keypress_recovery_blocked_when_the_key_itself_is_risky(page, logger, monkeypatch):
    """Same fix as agent/loop.py's keypress regression: the key being sent
    must be classified too, not just the focused element's own name."""
    page.set_content(
        '<div data-test="foo" role="textbox" aria-label="row-42" '
        'onkeydown="if(event.key===\'Delete\') this.setAttribute(\'data-deleted\',\'1\')">item</div>'
    )
    _fake_groq(monkeypatch, "pick_element", {"element_index": 0})

    with pytest.raises(HardFailure):
        fallback.recover(page, _broken_step(action="keypress", value="Delete", description="delete row"), {}, load_policy(), logger)

    assert page.locator('[data-test="foo"]').get_attribute("data-deleted") is None
