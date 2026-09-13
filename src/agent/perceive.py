"""Perception: turn the live page into a compact, indexable list of
interactive controls the LLM can reason over.

Primary path (3.1's bias toward "would still work with no clean DOM"):
walk the DOM for interactive roles the way a screen reader would --
tag/role/name/test-id -- rather than relying on stable CSS selectors or
test IDs being present. This is deliberately closer to "accessibility
tree" than "raw DOM query": we compute an ARIA role and accessible name
for every candidate the same way assistive tech would, which is what
still works on legacy markup (nested tables, no semantic tags, frames).

Fallback path: when snapshot() finds zero usable elements (e.g. a
canvas-rendered control with no DOM to speak of), agent/loop.py takes a
screenshot instead and hands it to a vision+tool-calling model
(agent/llm.py's decide_vision, qwen/qwen3.6-27b on Groq) which replies
with pixel coordinates. See fixtures/canvas_button.html for a genuinely
DOM-less demo target and artifact/schema.py's "coordinates" locator kind
for how a vision-discovered step still becomes a replayable artifact step.
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

  function associatedLabelText(el) {
    // Standard <label for="id">Text</label> association -- how a screen
    // reader (and a sighted user) actually names a plain form field with no
    // aria-label/placeholder. Missing this is exactly the "assumes clean
    // markup" failure mode Section 3.1 warns against.
    if (el.id) {
      const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (label) return label.innerText;
    }
    return el.closest('label')?.innerText || null;
  }

  function accessibleName(el) {
    return (
      el.getAttribute('aria-label') ||
      associatedLabelText(el) ||
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


_GRID_OVERLAY_JS = """
() => {
  const overlay = document.createElement('canvas');
  overlay.id = '__vision_fallback_grid__';
  overlay.width = window.innerWidth;
  overlay.height = window.innerHeight;
  overlay.style.cssText = 'position:fixed;top:0;left:0;z-index:2147483647;pointer-events:none';
  const ctx = overlay.getContext('2d');
  ctx.strokeStyle = 'rgba(255,0,0,0.4)';
  ctx.fillStyle = 'red';
  ctx.font = '10px monospace';
  for (let x = 0; x < overlay.width; x += 50) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, overlay.height); ctx.stroke();
    ctx.fillText(String(x), x + 2, 10);
  }
  for (let y = 0; y < overlay.height; y += 50) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(overlay.width, y); ctx.stroke();
    ctx.fillText(String(y), 2, y + 10);
  }
  document.body.appendChild(overlay);
}
"""


def screenshot_fallback_b64(page: Page) -> str:
    """Vision-fallback perception: used when snapshot() finds zero elements
    (e.g. a canvas-rendered surface with no accessible DOM at all).

    Overlays a labeled pixel grid before capturing -- measured live to
    meaningfully improve small-vision-model coordinate grounding (without
    it, the model estimates scale/position from vibes; with it, it reads
    labeled gridlines). Removed again immediately after the screenshot so
    it never affects the real page or a later DOM-based perception pass.
    """
    import base64

    page.evaluate(_GRID_OVERLAY_JS)
    try:
        png_bytes = page.screenshot()
    finally:
        page.evaluate("() => document.getElementById('__vision_fallback_grid__')?.remove()")
    return base64.b64encode(png_bytes).decode("ascii")
