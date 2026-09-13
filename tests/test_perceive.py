import pytest
from playwright.sync_api import sync_playwright

from src.agent import perceive


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


def test_accessible_name_falls_back_to_associated_label(page):
    """Regression: the-internet.herokuapp.com's login form has no
    aria-label/placeholder on its inputs, only a standard <label for="id">
    -- a completely ordinary, real-world pattern our accessible-name
    computation was missing (it only checked aria-label/innerText/placeholder
    /value/alt/title, none of which a screen reader actually relies on for a
    bare labeled <input>)."""
    page.set_content(
        '<label for="username">Username</label><input id="username" type="text">'
        '<label for="password">Password</label><input id="password" type="password">'
    )
    elements = perceive.snapshot(page)
    names = [el["name"] for el in elements]
    assert "Username" in names
    assert "Password" in names
