"""Perception: turn the live page into a compact, indexable list of
interactive controls the LLM can reason over.

Primary path (3.1's bias toward "would still work with no clean DOM"):
walk the DOM for interactive roles the way a screen reader would --
tag/role/name/test-id -- rather than relying on stable CSS selectors or
test IDs being present. This is deliberately closer to "accessibility
tree" than "raw DOM query": we compute an ARIA role and accessible name
for every candidate the same way assistive tech would, which is what
still works on legacy markup (nested tables, no semantic tags, frames).

Fallback path: screenshot bytes for a vision-capable model to reason over
coordinates when no interactive node is usable (e.g. a canvas-rendered
control). Implemented and unit-tested, but not exercised live in this
submission -- see REPORT.md ("Cuts"): the Groq model used for the agent
loop (llama-3.3-70b-versatile) is text-only.
"""
from __future__ import annotations

from playwright.sync_api import Page

_COLLECT_JS = """
() => {
  const TEST_ID_ATTRS = ['data-testid', 'data-test', 'data-cy', 'data-qa'];
  const ROLE_BY_TAG = {
    button: 'button', a: 'link', select: 'combobox', textarea: 'textbox',
  };
  const INPUT_ROLE_BY_TYPE = {
    submit: 'button', button: 'button', checkbox: 'checkbox', radio: 'radio',
  };

  function isVisible(el) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return false;
    const style = window.getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none';
  }

  function accessibleName(el) {
    return (
      el.getAttribute('aria-label') ||
      el.innerText ||
      el.getAttribute('placeholder') ||
      el.value ||
      el.getAttribute('alt') ||
      el.getAttribute('title') ||
      ''
    ).trim().slice(0, 60);
  }

  function role(el) {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'input') {
      return INPUT_ROLE_BY_TYPE[el.type] || 'textbox';
    }
    return ROLE_BY_TAG[tag] || tag;
  }

  function cssPath(el) {
    const parts = [];
    let node = el;
    for (let depth = 0; node && node.nodeType === 1 && depth < 5; depth++) {
      let part = node.tagName.toLowerCase();
      const parent = node.parentElement;
      if (parent) {
        const siblings = Array.from(parent.children).filter(c => c.tagName === node.tagName);
        if (siblings.length > 1) {
          part += `:nth-of-type(${siblings.indexOf(node) + 1})`;
        }
      }
      parts.unshift(part);
      node = parent;
    }
    return parts.join(' > ');
  }

  const INTERACTIVE_ROLES = new Set([
    'button', 'link', 'textbox', 'combobox', 'checkbox', 'radio',
    'menuitem', 'tab', 'switch', 'searchbox', 'spinbutton',
  ]);

  const testIdSelector = TEST_ID_ATTRS.map(a => `[${a}]`).join(', ');
  const nodes = document.querySelectorAll(
    `button, a, input, select, textarea, [role], [onclick], ${testIdSelector}`
  );
  const out = [];
  for (const el of nodes) {
    if (!isVisible(el)) continue;
    let testId = null;
    for (const attr of TEST_ID_ATTRS) {
      const v = el.getAttribute(attr);
      if (v) { testId = v; break; }
    }
    const interactive = INTERACTIVE_ROLES.has(role(el));
    // Plain, non-interactive text still gets surfaced (as role "text") when it
    // carries a stable test id -- that's exactly the kind of read-only value
    // (a total, a status label) `extract` needs to target, and a test id makes
    // it a reliable extraction locator even though it's not clickable/typeable.
    if (!interactive && !testId) continue;
    // Skip a non-interactive container whose own test id just wraps other
    // test-id'd descendants (header-container, cart-list, ...) -- those
    // descendants are already surfaced individually and are more specific.
    if (!interactive && testId && el.querySelector(testIdSelector)) continue;
    out.push({
      tag: el.tagName.toLowerCase(),
      role: interactive ? role(el) : 'text',
      name: accessibleName(el),
      test_id: testId,
      css: cssPath(el),
    });
  }
  return out;
}
"""


def snapshot(page: Page) -> list[dict]:
    """Return the indexed list of currently interactive, visible elements."""
    elements = page.evaluate(_COLLECT_JS)
    for i, el in enumerate(elements):
        el["index"] = i
    return elements


def format_for_llm(elements: list[dict], url: str) -> str:
    lines = [f"URL: {url}", "Interactive elements:"]
    for el in elements:
        marker = f" [test_id={el['test_id']}]" if el["test_id"] else ""
        lines.append(f"  [{el['index']}] {el['role']} \"{el['name']}\"{marker}")
    return "\n".join(lines)


def screenshot_fallback(page: Page) -> bytes:
    """Vision-fallback perception. See module docstring: implemented, not
    wired into the live loop because the configured model has no vision."""
    return page.screenshot()
