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
