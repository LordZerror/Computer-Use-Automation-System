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

  function labelTextExcludingControl(labelEl) {
    // A <label> wrapping its control (<label>Country <select>...</select>
    // </label>, no `for` needed) has the control's own rendered text as
    // part of labelEl.innerText -- for a <select> that's every <option>
    // concatenated, which is exactly the not-a-real-accessible-name problem
    // accessibleName() below works around for the direct case. Strip any
    // nested form control out of a clone before reading text so only the
    // label's own wording comes back, the same fix applied either way a
    // label can be associated with its control.
    const clone = labelEl.cloneNode(true);
    clone.querySelectorAll('select, input, textarea').forEach(c => c.remove());
    return clone.innerText;
  }

  function associatedLabelText(el) {
    // Standard <label for="id">Text</label> association -- how a screen
    // reader (and a sighted user) actually names a plain form field with no
    // aria-label/placeholder. Missing this is exactly the "assumes clean
    // markup" failure mode Section 3.1 warns against.
    if (el.id) {
      const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (label) return labelTextExcludingControl(label);
    }
    const wrapping = el.closest('label');
    return wrapping ? labelTextExcludingControl(wrapping) : null;
  }

  function accessibleName(el, isSelect) {
    // A <select>'s real accessible name (per the browser's own accname
    // computation, and what Playwright's get_by_role/get_by_text actually
    // match against) comes from label association only -- neither its
    // rendered content/current value (innerText/.value -- every <option>'s
    // text concatenated / the selected option's text) nor `placeholder`/
    // `alt` (not applicable to <select> at all; verified live: a
    // placeholder-only <select> gets 0 matches from get_by_role(name=...))
    // is the real accessible name. `title` is still legitimate (verified:
    // it does match) so it's the one attribute fallback left unguarded.
    // Falling through to an empty name here is correct and honest:
    // recorder.py's _ranked_locators() already skips the role/text
    // strategies when name is empty, leaving just the (still-usable) css
    // fallback rather than claiming two dead strategies.
    return (
      el.getAttribute('aria-label') ||
      associatedLabelText(el) ||
      (isSelect ? null : el.innerText) ||
      (isSelect ? null : el.getAttribute('placeholder')) ||
      (isSelect ? null : el.value) ||
      (isSelect ? null : el.getAttribute('alt')) ||
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
    const isSelect = el.tagName.toLowerCase() === 'select';
    out.push({
      tag: el.tagName.toLowerCase(),
      role: interactive ? role(el) : 'text',
      name: accessibleName(el, isSelect),
      test_id: testId,
      css: cssPath(el),
      // <option> children aren't collected as their own elements (they're
      // not independently clickable -- you select via the parent), but
      // select_option needs to know what's actually choosable, so surface
      // them here on the <select> itself. Capped at 30 *options* -- an
      // unrelated country/timezone/state picker elsewhere on the page
      // shouldn't dump hundreds of entries into every perception turn's
      // prompt -- but NOT per-label truncated the way accessibleName() caps
      // a name: select_option matches by exact label text, so truncating
      // an individual option's text would make any option past 60 chars
      // permanently unselectable. `options_total` (separate from the
      // possibly-truncated `options` list itself) is what format_for_llm
      // uses to tell the model more exist rather than silently hiding them
      // -- a fake "...N more" entry inside `options` risks the model trying
      // to select that literal string.
      options: isSelect ? Array.from(el.options).slice(0, 30).map(o => o.text.trim()) : null,
      options_total: isSelect ? el.options.length : null,
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


def is_native_select(element: dict) -> bool:
    """Whether `select_option` is actually usable on this element -- only a
    real <select> gets an `options` list (see _COLLECT_JS); a JS-built
    listbox widget shares the same perceived `combobox` role but Playwright's
    `.select_option()` only operates on the real thing. One shared check
    (agent/loop.py and replay/fallback.py both call this) instead of two
    independent copies of `element.get("options") is None`."""
    return element.get("options") is not None


def format_for_llm(elements: list[dict], url: str) -> str:
    lines = [f"URL: {url}", "Interactive elements:"]
    for el in elements:
        marker = f" [test_id={el['test_id']}]" if el["test_id"] else ""
        options = ""
        if el.get("options"):
            total = el.get("options_total") or len(el["options"])
            truncated = f" (+{total - len(el['options'])} more not shown)" if total > len(el["options"]) else ""
            options = f" options={el['options']}{truncated}"
        lines.append(f"  [{el['index']}] {el['role']} \"{el['name']}\"{marker}{options}")
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
