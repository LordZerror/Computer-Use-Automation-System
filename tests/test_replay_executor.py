import pytest
from playwright.sync_api import sync_playwright

from src.artifact.schema import Checkpoint, LocatorStrategy
from src.replay.errors import BusinessOutcome, HardFailure, load_error_signatures
from src.replay.executor import _check_for_app_errors, _verify_checkpoint
from src.replay.locator import resolve


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
