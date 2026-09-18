from types import SimpleNamespace

import pytest
from playwright.sync_api import sync_playwright

from src.artifact.schema import Checkpoint, LocatorStrategy
from src.replay.errors import BusinessOutcome, HardFailure, load_error_signatures
from src.replay.executor import _check_for_app_errors, _execute_step, _verify_checkpoint
from src.replay.locator import resolve
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


def test_locator_prefers_test_id_when_available(page):
    page.set_content('<button data-test="foo">Add to cart</button>')
    loc = resolve(page, [LocatorStrategy(kind="test_id", test_id="foo")], "step_0")
    assert loc.inner_text() == "Add to cart"


def test_locator_falls_back_to_css_when_test_id_missing(page):
    page.set_content('<button class="only-css">Continue</button>')
    strategies = [
        LocatorStrategy(kind="test_id", test_id="does-not-exist"),
        LocatorStrategy(kind="css", css=".only-css"),
    ]
    loc = resolve(page, strategies, "step_0")
    assert loc.inner_text() == "Continue"


def test_locator_raises_hard_failure_when_no_strategy_resolves(page):
    page.set_content("<div>nothing interactive here</div>")
    with pytest.raises(HardFailure):
        resolve(page, [LocatorStrategy(kind="test_id", test_id="missing")], "step_0")


def test_locator_refuses_to_guess_between_ambiguous_matches(page):
    # Two structurally-identical "Add to Cart" buttons (e.g. a product grid) --
    # replay must not silently click whichever one Playwright returns first.
    page.set_content(
        '<button class="add">Add to Cart</button>'
        '<button class="add">Add to Cart</button>'
    )
    with pytest.raises(HardFailure):
        resolve(page, [LocatorStrategy(kind="text", text="Add to Cart")], "step_0")


def test_locator_falls_through_ambiguous_strategy_to_a_unique_one(page):
    # The top-ranked strategy (text) is ambiguous, but a lower-ranked, more
    # specific strategy (css) still resolves uniquely -- degrade-through-ranking
    # should recover here rather than failing outright.
    page.set_content(
        '<button class="add">Add to Cart</button>'
        '<button id="target" class="add only-this-one">Add to Cart</button>'
    )
    strategies = [
        LocatorStrategy(kind="text", text="Add to Cart"),
        LocatorStrategy(kind="css", css="#target"),
    ]
    loc = resolve(page, strategies, "step_0")
    assert loc.get_attribute("id") == "target"


def test_checkpoint_text_present_passes_when_text_found(page):
    page.set_content("<body>Checkout: Overview</body>")
    _verify_checkpoint(page, Checkpoint(kind="text_present", text="Checkout: Overview"))  # no raise


def test_checkpoint_text_present_fails_when_text_missing(page):
    page.set_content("<body>Still on the cart page</body>")
    with pytest.raises(HardFailure):
        _verify_checkpoint(page, Checkpoint(kind="text_present", text="Checkout: Overview"))


def test_error_signature_classifies_known_business_outcome():
    sigs = load_error_signatures()
    category, code = sigs.classify(
        "Epic sadface: Username and password do not match any user in this service"
    )
    assert category == "business_outcome"
    assert code == "invalid_login"


def test_error_signature_returns_none_for_unrecognized_text():
    sigs = load_error_signatures()
    assert sigs.classify("something totally unexpected happened") is None


def test_empty_error_container_is_not_treated_as_an_error(page):
    """Regression: an always-present-but-empty error container (common in
    real apps -- the banner div exists before any error occurs) must not
    be mistaken for an active error just because a selector matches it."""
    page.set_content('<div class="error-message-container" style="padding:10px"></div>')
    sigs = load_error_signatures()
    _check_for_app_errors(page, sigs)  # must not raise


def test_nonempty_matching_error_container_raises_business_outcome(page):
    page.set_content(
        '<div data-test="error">Epic sadface: Username and password do not match any user in this service</div>'
    )
    sigs = load_error_signatures()
    with pytest.raises(BusinessOutcome):
        _check_for_app_errors(page, sigs)


def test_select_option_step_chooses_by_visible_label(page):
    page.set_content(
        '<select id="country"><option>USA</option><option>Canada</option><option>UK</option></select>'
    )
    step = SimpleNamespace(
        action="select_option", risk="safe", step_id="s0", value="Canada",
        locators=[LocatorStrategy(kind="css", css="#country")],
    )
    _execute_step(page, step, {}, load_policy())
    assert page.eval_on_selector("#country", "el => el.selectedOptions[0].text") == "Canada"


def test_select_option_step_on_non_select_fails_with_clear_attribution(page):
    """No perception/element dict exists at replay time (only the recorded
    locators), so a JS-built listbox can't be caught upfront the way
    discovery/fallback do -- but Playwright's own exception should still be
    attributed to this step_id and given a readable message, not left to
    fall through to replay()'s generic 'unknown step' handler."""
    page.set_content('<div id="widget" role="combobox">React-select widget</div>')
    step = SimpleNamespace(
        action="select_option", risk="safe", step_id="s0", value="Canada",
        locators=[LocatorStrategy(kind="css", css="#widget")],
    )
    with pytest.raises(HardFailure) as exc_info:
        _execute_step(page, step, {}, load_policy())
    assert exc_info.value.step_id == "s0"


def test_select_option_step_blocked_when_risky(page):
    page.set_content('<select id="country"><option>USA</option><option>Canada</option></select>')
    step = SimpleNamespace(
        action="select_option", risk="risky", step_id="s0", value="Canada",
        locators=[LocatorStrategy(kind="css", css="#country")],
    )
    with pytest.raises(HardFailure):
        _execute_step(page, step, {}, load_policy())


def test_hover_step_reveals_a_hover_triggered_element(page):
    page.set_content(
        '<style>#menu{display:none}#trigger:hover + #menu{display:block}</style>'
        '<div id="trigger">Account</div><div id="menu">Settings</div>'
    )
    assert not page.locator("#menu").is_visible()
    step = SimpleNamespace(
        action="hover", risk="safe", step_id="s0",
        locators=[LocatorStrategy(kind="css", css="#trigger")],
    )
    _execute_step(page, step, {}, load_policy())
    assert page.locator("#menu").is_visible()


def test_keypress_step_sends_key_to_the_resolved_element(page):
    page.set_content(
        '<input id="q"><div id="status"></div>'
        '<script>document.getElementById("q").addEventListener("keydown", e => {'
        'if (e.key === "Enter") document.getElementById("status").textContent = "submitted";'
        '});</script>'
    )
    step = SimpleNamespace(
        action="keypress", risk="safe", step_id="s0", value="Enter",
        locators=[LocatorStrategy(kind="css", css="#q")],
    )
    _execute_step(page, step, {}, load_policy())
    assert page.locator("#status").inner_text() == "submitted"


def test_keypress_step_blocked_when_risky(page):
    page.set_content('<input id="q">')
    step = SimpleNamespace(
        action="keypress", risk="risky", step_id="s0", value="Enter",
        locators=[LocatorStrategy(kind="css", css="#q")],
    )
    with pytest.raises(HardFailure):
        _execute_step(page, step, {}, load_policy())
