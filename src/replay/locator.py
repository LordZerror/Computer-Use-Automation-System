"""Resolve a step's ranked locator strategies against the live page.

Each strategy is tried in the order the recorder ranked them (test_id ->
role/name -> text -> css/xpath); the first one that resolves to exactly
one visible element wins. This is what lets replay survive incidental
DOM noise without needing to be "smart" -- and it's the seam a legacy or
desktop surface would plug into differently (see REPORT.md).
"""
from __future__ import annotations

from playwright.sync_api import Locator, Page

from src.artifact.schema import LocatorStrategy
from src.replay.errors import HardFailure

_TEST_ID_ATTRS = ["data-test", "data-testid", "data-cy", "data-qa"]


def _build(page: Page, s: LocatorStrategy) -> Locator:
    if s.kind == "test_id":
        selector = ", ".join(f'[{attr}="{s.test_id}"]' for attr in _TEST_ID_ATTRS)
        return page.locator(selector)
    if s.kind == "role":
        return page.get_by_role(s.role, name=s.name) if s.name else page.get_by_role(s.role)
    if s.kind == "text":
        return page.get_by_text(s.text, exact=False)
    if s.kind == "css":
        return page.locator(s.css)
    if s.kind == "xpath":
        return page.locator(f"xpath={s.xpath}")
    raise ValueError(f"unknown locator kind: {s.kind}")


def resolve(page: Page, strategies: list[LocatorStrategy], step_id: str) -> Locator:
    attempted: list[str] = []
    ambiguous: list[str] = []
    for s in strategies:
        attempted.append(s.kind)
        try:
            loc = _build(page, s)
            count = loc.count()
        except Exception:
            continue
        visible = [i for i in range(count) if loc.nth(i).is_visible()]
        if len(visible) == 1:
            return loc.nth(visible[0])
        if len(visible) > 1:
            # Don't guess which one the recorder meant -- fall through to a
            # lower-ranked (often more specific, e.g. structural css) strategy
            # instead of silently clicking whichever Playwright returns first.
            ambiguous.append(s.kind)
    if ambiguous:
        raise HardFailure(
            step_id=step_id,
            expected=f"exactly one visible element for strategies {attempted}",
            observed=f"strategies {ambiguous} each matched multiple visible elements -- refusing to guess",
        )
    raise HardFailure(
        step_id=step_id,
        expected=f"one of locator strategies {attempted} to resolve to a visible element",
        observed="no strategy matched a visible element",
    )
