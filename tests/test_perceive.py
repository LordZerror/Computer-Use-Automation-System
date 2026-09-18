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


def test_select_options_are_surfaced(page):
    """select_option is unusable if the model can't see what's choosable --
    <option> isn't its own collected element, so the <select> itself must
    carry its options."""
    page.set_content(
        '<select><option>USA</option><option>Canada</option><option>UK</option></select>'
    )
    elements = perceive.snapshot(page)
    select = next(el for el in elements if el["role"] == "combobox")
    assert select["options"] == ["USA", "Canada", "UK"]


def test_non_select_elements_have_no_options(page):
    page.set_content('<button>Click me</button>')
    elements = perceive.snapshot(page)
    assert elements[0]["options"] is None


def test_unlabeled_select_gets_empty_name_not_concatenated_option_text(page):
    """Regression: accessibleName() used to fall back to innerText/.value for
    a <select> with no aria-label/<label>, producing every option's text
    concatenated ('Option 1\\nOption 2\\n...') as its 'name' -- which then
    became a role/text LocatorStrategy that can never resolve, since that's
    not the real accessible name Playwright's get_by_role/get_by_text match
    against (verified live: both return 0 matches for that string). An empty
    name here is correct -- recorder.py already skips role/text strategies
    when name is empty, falling back to the still-working css locator."""
    page.set_content('<select id="c"><option>Option 1</option><option>Option 2</option></select>')
    elements = perceive.snapshot(page)
    select = next(el for el in elements if el["role"] == "combobox")
    assert select["name"] == ""

    # And the real accessible-name match Playwright would do confirms it:
    assert page.get_by_role("combobox", name="Option 1\nOption 2").count() == 0
    assert page.get_by_role("combobox").count() == 1  # matches with no name filter


def test_select_options_are_capped_at_30(page):
    """An unrelated large dropdown elsewhere on the page (a country/state/
    timezone picker) shouldn't dump hundreds of options into every
    perception turn's prompt -- capped the same way accessibleName() caps
    a name."""
    options_html = "".join(f"<option>Item {i}</option>" for i in range(50))
    page.set_content(f'<select>{options_html}</select>')
    elements = perceive.snapshot(page)
    select = next(el for el in elements if el["role"] == "combobox")
    assert len(select["options"]) == 30
    assert select["options"][0] == "Item 0"
    assert select["options_total"] == 50


def test_format_for_llm_flags_truncated_options_instead_of_hiding_them(page):
    """Regression: a dropdown with more than 30 real options was silently
    truncated with no signal -- the model could easily assume only 30
    options exist and pick a wrong nearby value instead of discovering the
    one it actually needs isn't shown."""
    options_html = "".join(f"<option>Item {i}</option>" for i in range(50))
    page.set_content(f'<select>{options_html}</select>')
    elements = perceive.snapshot(page)
    text = perceive.format_for_llm(elements, "https://example.com")
    assert "+20 more not shown" in text


def test_format_for_llm_no_truncation_note_when_options_fit(page):
    page.set_content('<select><option>USA</option><option>Canada</option></select>')
    elements = perceive.snapshot(page)
    text = perceive.format_for_llm(elements, "https://example.com")
    assert "more not shown" not in text


def test_wrapping_label_excludes_the_selects_own_option_text(page):
    """Regression: a <select> wrapped by a <label> with no `for`/id (a
    common, valid pattern -- <label>Country <select>...</select></label>)
    had its label read via closest('label').innerText, which includes the
    select's own rendered text -- for a <select> that's every <option>
    concatenated, reproducing the exact same dead-locator bug the aria-label
    fix addressed, just via a second path."""
    page.set_content('<label>Country <select><option>USA</option><option>Canada</option></select></label>')
    elements = perceive.snapshot(page)
    select = next(el for el in elements if el["role"] == "combobox")
    assert select["name"] == "Country"


def test_long_option_labels_are_not_truncated(page):
    """Regression: options were once truncated to 60 chars like
    accessibleName() truncates a name -- but select_option matches by exact
    label text, so truncating an option's own text would make anything
    past 60 chars permanently unselectable."""
    long_label = "United Kingdom of Great Britain and Northern Ireland and then some more words past sixty chars"
    page.set_content(f'<select><option>USA</option><option>{long_label}</option></select>')
    elements = perceive.snapshot(page)
    select = next(el for el in elements if el["role"] == "combobox")
    assert select["options"][1] == long_label


def test_labeled_select_still_gets_its_aria_label(page):
    page.set_content(
        '<select id="c" aria-label="Country"><option>USA</option><option>Canada</option></select>'
    )
    elements = perceive.snapshot(page)
    select = next(el for el in elements if el["role"] == "combobox")
    assert select["name"] == "Country"


def test_select_with_only_placeholder_or_alt_gets_empty_name(page):
    """Regression: the same 'phantom accessible name' bug fixed for
    innerText/.value also applies to placeholder/alt -- neither is a real
    accessible-name source for a <select> (verified live: a placeholder-only
    select gets 0 matches from get_by_role(name=...)), so using them here
    would reproduce the exact dead-locator bug via a different attribute."""
    page.set_content('<select placeholder="Choose a country"><option>USA</option></select>')
    elements = perceive.snapshot(page)
    select = next(el for el in elements if el["role"] == "combobox")
    assert select["name"] == ""


def test_select_title_attribute_still_works(page):
    """Unlike placeholder/alt, `title` genuinely is a valid accessible-name
    source for a <select> (verified live: it does match get_by_role), so it
    stays unguarded."""
    page.set_content('<select title="Choose a country"><option>USA</option></select>')
    elements = perceive.snapshot(page)
    select = next(el for el in elements if el["role"] == "combobox")
    assert select["name"] == "Choose a country"
