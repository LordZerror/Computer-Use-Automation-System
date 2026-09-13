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

Perception is hybrid, both halves genuinely exercised: accessibility-tree-style
text (role/name/test-id) is the default and primary path — it survives
table-based legacy layouts and frames far better than pixel coordinates do
(Section 3.1's bias toward "works with no clean DOM"). When that returns zero
elements (a canvas-rendered or otherwise DOM-less control), `loop.py` falls
back to a screenshot handed to a vision+tool-calling model
(`qwen/qwen3.8-27b` on Groq) that replies with pixel coordinates, with a
labeled grid overlaid on the screenshot first — measured live to
meaningfully improve this model's coordinate grounding, which otherwise
estimates position from scale alone and gets it wrong. A vision-discovered
step becomes an ordinary artifact step with a `coordinates` locator (see §2)
— it isn't a parallel, second-class system. Demonstrated end-to-end against
`fixtures/canvas_button.html` (a genuinely DOM-less surface, `perceive.snapshot()`
returns `[]` on it — asserted by a test, not just assumed):
`evidence/discovery-vision1/` (real vision-driven discovery) and
`evidence/replay-vision-replay1/` (deterministic replay of the resulting
artifact, no LLM call).

## 2. Artifact schema

`src/artifact/schema.py`. A `Capability` is:

- `input_params`: typed, with a `sensitive` flag.
- `steps`: ordered `navigate` / `click` / `type` actions. Each carries a
  **ranked list of locator strategies** (`test_id → role+accessible-name →
  visible text → CSS path → pixel coordinates`), not a single selector —
  replay tries them in order and only fails if none resolve. This is the
  single biggest reliability lever available, and it's also the seam a
  legacy or desktop surface would extend (see §4): the ranking vocabulary
  stays the same, only how each strategy is *resolved* changes per surface.
  `coordinates` is the literal bottom rung — no DOM node exists at all, only
  a pixel position from the vision fallback (§1/§3) — which is what makes a
  vision-discovered step a first-class, replayable artifact step rather than
  a separate, second-class mechanism bolted on beside the schema.
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

**Assisted fallback (Section 8 stretch goal), the one narrow exception to "no
LLM in replay":** `src/replay/fallback.py`. When a step's `HardFailure` comes
from every recorded locator strategy failing to resolve, and only when
`--allow-assisted-fallback` is passed, replay allows exactly **one** bounded
Groq call: the model sees the step's `description` and its now-broken
recorded locators, a fresh perception snapshot, and a two-tool schema —
`pick_element(index)` or `none_found()` — nothing else. It is never asked
what to do, only which currently-visible element matches a step whose action
is already fixed, so it cannot invent a new action, navigate, or loop. A
recovered element still goes through the same risk classification a normal
click would (`test_recovered_risky_element_is_blocked_not_executed`), and the
whole exchange is logged as evidence
(`assisted_fallback_attempted`/`_recovered`/`_declined_risky`/`_not_found`).
Demonstrated live: a copy of the real artifact with one step's locators
deliberately corrupted (`add_to_cart_checkout__fault_demo.json`, name flags
it as a fault-injection copy, not the canonical capability) fails normally
and recovers correctly with the flag — see `evidence/replay-fallback-*/`. Unit
tests stub the Groq response so the suite stays offline
(`tests/test_replay_fallback.py`).

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

**Portability check, done rather than just argued.** To test whether the
above is real, the same unmodified pipeline (only `config/allowlist.yaml` and
`config/error_signatures.yaml` were extended, exactly as §4 predicts a
cross-tenant deployment would) was pointed at two more public sites, neither
seen during development:

- `automationexercise.com` (a different e-commerce vendor's checkout flow) —
  `artifacts/automationexercise_add_to_cart.json`,
  `evidence/discovery-portability-ae1/`,
  `evidence/replay-portability-ae-replay1/`.
- `the-internet.herokuapp.com/login` (a plain server-rendered login form,
  test-id-free) — `artifacts/herokuapp_login.json`,
  `evidence/discovery-portability-heroku1/`, plus a genuine cross-site
  `business_outcome` (`invalid_login`) once `#flash` and an `"is invalid"`
  pattern were added to `error_signatures.yaml` —
  `evidence/replay-portability-heroku-replay-bad/`.

This surfaced three real bugs the saucedemo-only development had never hit,
each fixed and covered by a test rather than special-cased for one site:

1. **Accessible-name gap**: herokuapp's login fields have no
   `aria-label`/placeholder, only a standard `<label for="id">` —
   `accessibleName()` didn't check label association at all
   (`tests/test_perceive.py`).
2. **Discovery could crash outright**: a Bootstrap modal intercepting a click
   on automationexercise.com threw a Playwright timeout that propagated all
   the way out of `run_discovery` and killed the process — replay already
   guaranteed it would never crash its caller, discovery didn't
   (`tests/test_agent_loop_robustness.py`; also dropped the default
   actionability timeout from 30s to 5s for discovery specifically, so one
   wrong guess doesn't cost 30 real seconds).
3. **The model doesn't reliably self-escalate on repetition**: despite the
   system prompt explicitly saying "don't retry a failed action," it retried
   an identical failing click 8 times in a row on one attempt. Section 3.6's
   own "repeated the same action with no progress" stuck-state is now
   detected in code (2 identical consecutive failures) rather than trusted to
   model judgment — escalates to a human when a person is watching
   (non-headless), stops as a dead end otherwise
   (`tests/test_agent_loop_robustness.py`).

None of these are saucedemo-specific patches — they're gaps in the generic
DOM-walking/loop-control code, which is exactly what a real second tenant
onboarding should be expected to surface and fix once, centrally.

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

**Confidence & approval gating (Section 8 stretch goal):** `Capability` carries
a `status` (`draft`/`approved`) plus a `stability_score`. `cli.py stability`
replays an artifact N times with identical params and scores how often the
outcome agreed with the majority — deliberately a *consistency* measure, not a
success-rate one, so an artifact that reliably returns the same
`business_outcome` for bad input scores as stable, not flaky (the three-way
result contract from §3 stays intact rather than being collapsed back into
pass/fail). `cli.py approve` flips `status` to `approved` only above a
threshold (or `--force`, loudly logged). `replay()`'s new
`require_approved` flag rejects a non-approved capability before a browser is
even launched; the capability API's `/invoke` — the one genuinely
*unattended* caller in this project, matching Section 8's wording exactly —
always sets it, while the CLI's own `replay` stays unattended-gate-free by
default for manual testing. Verified live: `evidence/replay-stability-*/` (5
real replays, score 1.0) and `evidence/replay-api-*/` (the capability API
refusing a `draft` copy, then succeeding once approved).

**Limits**: the allowlist is domain + action-type only — it doesn't reason
about *which record* a risky-adjacent action touches (e.g. "delete" on one's
own draft vs. someone else's live account), and risk classification is a
keyword match on visible text, which a differently-worded UI could evade. A
production version would need per-field data-classification rules; the
confidence/approval gate above covers *whether an artifact runs unattended at
all*, but doesn't yet condition approval on which risk-classified steps it
contains — an artifact with only `safe` steps and one with a blocked `risky`
step are approved the same way today.

## 7. Cuts

- **Correction, not a cut**: an earlier version of this report claimed no
  vision-capable model was available on the Groq account in use, and left
  the screenshot/coordinate fallback unwired on that basis. That premise was
  wrong — `qwen/qwen3.6-27b`/`qwen3.8-27b` are vision+tool-calling models
  already visible to that key — and it's now wired and demonstrated live
  (§1, §3). Also worth recording honestly: `qwen3.6-27b` turned out to
  hallucinate a tiled/repeated layout on the small, mostly-white demo
  screenshot and gave unreliable coordinates; `qwen3.8-27b` with a labeled
  grid overlay did not have that problem. Model choice for a vision fallback
  is not a solved default — it needs the same "verify against the live
  surface" discipline as everything else in this project.
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
- **Stretch goals not attempted**: code generation from an artifact, and
  canonicalization/cross-tenant reuse. The brief asks for "at most one or
  two — depth over breadth"; three were built anyway, past that guidance, in
  response to an explicit follow-up ask: the agent-facing capability API
  (`src/capabilities/`), since it most directly demonstrates the "artifact =
  reusable capability" framing the brief centers on; assisted fallback on
  replay failure (`src/replay/fallback.py`, §3), since it most directly
  exercises the error-handling and safety machinery already built rather than
  adding a new subsystem; and confidence & approval gating (§6), because it
  composes with both of the others (multi-run stability feeds approval, which
  gates the capability API) instead of sitting off to the side.

**Next with more time**: wire a vision-capable model behind the same
`perceive.py` seam for the fallback path; add the per-tenant override
resolution described in §4 with a second recorded artifact against a
deliberately-varied clone of the target to prove cross-tenant reuse; extend
approval gating to condition on a capability's risk-classified steps, not just
its stability score.
