# Design Report

## 1. Architecture

Single Python process, synchronous, CLI-driven (`src/cli.py`) — no queues, no
services, no database. The brief explicitly says scaling infrastructure isn't
rewarded, and a discover/replay tool used by one operator (or invoked by an
agent, one call at a time, in the stretch-goal API) has no concurrency problem
to solve yet.

Three pipeline stages, each independently testable:

1. **Discovery** (`src/agent/`): `perceive.py` turns the live DOM into an
   indexed list of interactive controls (role/name/test-id/CSS path), computed
   the way a screen reader would rather than assuming clean markup. `llm.py`
   wraps Groq's OpenAI-compatible tool-calling API — the model sees the goal, a
   manifest of available named parameters, and the current perception list, and
   must call exactly one tool (`click` / `type` / `navigate` / `extract` /
   `finish` / `escalate`) per turn. `loop.py` executes that tool call against
   Playwright and appends a transcript entry.
2. **Recording** (`src/artifact/recorder.py`): a pure function from transcript
   to `Capability`. It never sees the model's reasoning text or retries — only
   what actually happened and the DOM metadata captured at the moment of each
   action. This is the decoupling Section 3.2 asks for.
3. **Replay** (`src/replay/executor.py`): walks the artifact's steps with the
   LLM entirely out of the loop, using the same locator vocabulary the
   recorder wrote down.

Key trade-off: the agent loop is text-only (accessibility-tree-style
perception), not screenshot+vision. This is deliberately biased toward "works
with no clean DOM" (Section 3.1) — a role/name/test-id list survives table-based
legacy layouts and frames far better than pixel coordinates do — at the cost of
not handling canvas-rendered or image-only controls. See Cuts.

## 2. Artifact schema

`src/artifact/schema.py`. A `Capability` is:

- `input_params`: typed, with a `sensitive` flag.
- `steps`: ordered `navigate` / `click` / `type` actions. Each carries a
  **ranked list of locator strategies** (`test_id → role+accessible-name →
  visible text → CSS path`), not a single selector — replay tries them in
  order and only fails if none resolve. This is the single biggest
  reliability lever available, and it's also the seam a legacy or desktop
  surface would extend (see §4): the ranking vocabulary stays the same, only
  how each strategy is *resolved* changes per surface.
- `output_spec`: declared outputs are resolved against **final page state**
  after all steps run and the checkpoint passes, each with its own locator —
  deliberately not "whatever `extract` steps happened to run," so the
  contract an agent calls against doesn't depend on incidental discovery
  behavior.
- `checkpoint`: a single assertable condition (URL pattern / text present /
  element present) that proves the goal was reached, not just that clicks
  fired.
- A **secrets rule enforced by construction**: a `type` step's value is either
  a literal (non-sensitive) or a `param_ref` into a `sensitive` input param —
  never both. A sensitive value is supplied at replay time and is never
  written into the JSON artifact (`tests/test_artifact_schema.py` asserts
  this directly).

Reviewability was the design driver: every field a human reviewer would ask
("what does this click?", "why do we think that's stable?", "what does the
caller get back?") has a dedicated, typed field rather than being buried in a
generic `metadata` blob.

## 3. Determinism & error handling

Determinism comes from three things together: (a) ranked, multi-strategy
locators instead of one brittle selector; (b) an explicit **settle** step
(`_settle()` in `loop.py`/`executor.py`) that waits past `networkidle` before
perceiving or asserting anything — the demo target renders its checkout
overview and info-form fields visibly *after* network idle, which is exactly
the "transient slowness" runtime condition Section 1 calls out, discovered by
walking the real flow manually before wiring up the LLM; and (c) the
checkpoint assertion at the end of replay, so a run that clicked all the right
things but landed somewhere unexpected still fails loudly instead of
reporting success.

Error handling is a three-way split (`src/replay/errors.py`), not a
success/fail boolean:

- **`business_outcome`** — a legitimate answer, e.g. `invalid_login`. Detected
  generically: after every step, replay checks a small set of app-agnostic
  error-banner selectors (`config/error_signatures.yaml`) and classifies
  non-empty matched text against known patterns. Config lives outside the
  artifact deliberately — it's a vendor-product-level policy ("what does an
  error look like in this app family"), reusable across every tenant running
  the same product rather than re-encoded per capability.
- **`recoverable`** — a known dismissible interstitial is closed and the same
  step retried (`dismissible_selectors`); none exist on the demo target, so
  this path is exercised only by its unit test, not live evidence.
- **`failure`** — everything else: no locator strategy resolved, an
  unrecognized error banner appeared, the checkpoint didn't verify, or an
  unexpected exception was raised. Always carries `step_id` / `expected` /
  `observed` and a screenshot. `executor.py`'s outermost `except Exception`
  guarantees replay never crashes its caller — it always returns a structured
  `ReplayResult`.

A regression test (`test_empty_error_container_is_not_treated_as_an_error`)
exists because the first version of this check had exactly the bug Section 1
warns about: the demo site's error container is *always present* in the DOM,
just empty until an error occurs. Matching on selector-present instead of
selector-present-with-content would have classified every successful run as a
failure.

## 4. Heterogeneity & multi-tenant

**Surface abstraction.** The seam is `perceive.py` (produces a normalized
element list) + the action executor (consumes locator strategies) — steps and
artifacts only ever talk in terms of `LocatorStrategy` (role/name, test-id,
text, css/xpath), never raw DOM APIs. A legacy web app with frames/tables needs
no schema change, only a perception function that still knows how to compute
role/name/test-id from ugly markup (which is what the current implementation
already does — it doesn't assume semantic HTML). A desktop surface would swap
Playwright for `pywinauto`/UIA: `role` becomes the UIA control type, `name` the
UIA `Name` property, `test_id` an `AutomationId` if the app happens to set one,
and `css` is replaced by a UIA tree path. `Capability`, `Step`, and the
replay/recorder logic are unchanged — only `_build()` in `locator.py` and the
snapshot JS in `perceive.py` get a per-surface implementation behind the same
interface.

**Multi-tenant reuse.** Keep a `Capability` at the *vendor-product* level
(what this implementation already does — `target.app_id` identifies the
product, not the tenant) plus a small, separate per-tenant override document:
`{app_id, base_url, locator_overrides: {step_id: LocatorStrategy}}`. Replay
resolves a step's locators as `overrides.get(step_id, step.locators)` before
falling into the normal ranked-resolution loop — most tenants running the same
vendor product need zero overrides because role/name/test-id survive
white-labeling; a tenant with a customized build only needs an override for
the specific steps that actually changed. **Drift detection**: store the
resolved locator `kind` actually used on the last N successful replays per
step; if resolution silently downgrades from `test_id` to `css` (or fails
outright) for a chunk of tenants running the same `app_id`, that's the signal
a vendor-side markup change shipped and the base artifact (or a specific
tenant's override) needs attention — this is a monitoring policy on top of the
existing `ReplayResult`/evidence log, not a new subsystem.

## 5. Escalation & handoff

`src/escalation/manager.py`. "Stuck" is detected in two places: the agent
loop's own `escalate` tool call (the model decides it can't proceed) and a
replay `HardFailure` when `allow_escalation=True` is passed to the executor.
Both raise the same `escalate()` call with full context: the goal/capability
name, the step id, the current URL, the reason, and a screenshot — all written
to `evidence/.../intervention_<step>.json` before anything blocks.

The handoff mechanism is real, and deliberately minimal per the brief's scope
note: the browser is launched **non-headless**, so the human takes control by
operating that same visible window directly — same cookies, same DOM, same
live session, not a fresh one. Pausing means the automation loop stops issuing
commands and blocks on a CLI prompt; resuming is an Enter keypress; what the
human did is captured as a short free-text note logged alongside a
`control.json` file recording who holds control and since when. A replay that
escalates retries the failed step exactly once after control returns.

What a real operator console would add: a second process (or browser tab)
attaching to the same session over the Chrome DevTools Protocol instead of
sharing an OS window, an authenticated UI instead of a terminal prompt, and
structured action diffing instead of a free-text note — none of which changes
the underlying model (pause → cede control on the same session → resume →
record what happened), which is the part this implementation makes real.

## 6. Safety

`src/safety/policy.py` + `config/allowlist.yaml`, consulted by both discovery
and replay from the same object:

- **Domain allowlist**: any `navigate`/`goto` outside `allowed_domains` raises
  `PolicyViolation` before it happens.
- **Action-type allowlist**: only `navigate`/`click`/`type`/`finish` are
  permitted at all.
- **Risk classification**: a control is `risky` if its visible text matches a
  configured marker (`finish`, `place order`, `pay`, `delete`, …) — chosen
  over an action-type-based rule because on this target the single
  irreversible action *is* a click (saucedemo's own "Finish" button, which
  places the order). Risky clicks are **blocked outright**, not merely
  flagged, both during discovery (the model is told the click was refused) and
  in replay (`HardFailure`) — conservative by design, since this project's
  goal deliberately stops at the checkout *review* page and never needs to
  cross that line.
- **Redaction**: any input param marked `sensitive` (or whose name matches a
  configured pattern like `password`/`card_number` even if the artifact author
  forgot to flag it) is masked before being written to a log line, and, as
  described in §2, can never appear as a literal value inside a saved
  artifact in the first place.

**Limits**: the allowlist is domain + action-type only — it doesn't reason
about *which record* a risky-adjacent action touches (e.g. "delete" on one's
own draft vs. someone else's live account), and risk classification is a
keyword match on visible text, which a differently-worded UI could evade. A
production version would need per-field data-classification rules and a
signed/reviewed approval step before a capability with any `risky` step could
run unattended (this overlaps with the "confidence & approval" stretch goal in
Section 8, not built here).

## 7. Cuts

- **Screenshot/coordinate vision fallback**: implemented
  (`perceive.screenshot_fallback`) but not wired into the live loop — Groq's
  `llama-3.3-70b-versatile` is text-only. The seam exists (perception returns
  an empty element list → today that escalates to a human; a vision-capable
  model would instead receive the screenshot and reply with coordinates).
- **Recoverable-condition path**: the code and one unit test exist
  (`dismissible_selectors`), but the demo target has no dismissible
  interstitial to exercise it live.
- **Auto-derived checkpoints**: the checkpoint is supplied explicitly on the
  `discover` CLI call rather than inferred from the run. Asking the model one
  more question ("what visible text confirms the goal?") would automate this;
  skipped as unnecessary complexity for a single demo goal.
- **Multi-tenant overrides and desktop perception**: designed in §4, not
  built — the brief explicitly doesn't ask for them to be implemented.
- **Operator console UI**: a CLI prompt over the same live browser window
  stands in for a real console, per the brief's own scope note (§5 above).
- **Stretch goals not attempted**: confidence/approval scoring, code
  generation from an artifact, and multi-run stability sampling. The one
  stretch goal built is the agent-facing capability API
  (`src/capabilities/`), since it most directly demonstrates the "artifact =
  reusable capability" framing the brief centers on.

**Next with more time**: wire a vision-capable model behind the same
`perceive.py` seam for the fallback path; add the per-tenant override
resolution described in §4 with a second recorded artifact against a
deliberately-varied clone of the target to prove cross-tenant reuse; add the
confidence/approval stretch goal on top of the existing evidence log (it's
mostly a rollup query over `ReplayResult.status` across runs).
