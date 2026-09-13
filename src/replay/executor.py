"""Deterministic replay: the production execution path (Section 3.3).

No LLM in the loop. Steps run in recorded order using ranked locator
resolution; after every step a generic error-signature check separates a
legitimate business outcome from an unrecognized (hard) failure; at the
end the checkpoint is verified before any declared output is trusted.
"""
from __future__ import annotations

import re

from playwright.sync_api import Page, sync_playwright

from src.artifact.schema import Capability, Checkpoint
from src.escalation.manager import escalate
from src.evidence.logger import EvidenceLogger
from src.replay import fallback
from src.replay.errors import BusinessOutcome, HardFailure, ReplayResult, load_error_signatures
from src.replay.locator import resolve
from src.safety.policy import load_policy


def _settle(page: Page) -> None:
    """See agent/loop.py's _settle: this target keeps rendering a couple
    seconds after `networkidle` fires. Replay treats that as expected
    transient slowness, not a bug -- wait past idle before asserting anything."""
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass
    page.wait_for_timeout(500)


def _check_for_app_errors(page: Page, error_sigs) -> None:
    for selector in error_sigs.error_selectors:
        loc = page.locator(selector)
        if loc.count() and loc.first.is_visible():
            text = loc.first.inner_text().strip()
            if not text:
                continue  # an empty error container (present-but-inactive) is not an error
            classification = error_sigs.classify(text)
            if classification and classification[0] == "business_outcome":
                raise BusinessOutcome(code=classification[1], message=text)
            raise HardFailure(step_id="app_error", expected="no application error banner", observed=text)


def _dismiss_known_interstitials(page: Page, error_sigs, logger: EvidenceLogger) -> None:
    for selector in error_sigs.dismissible_selectors:
        loc = page.locator(selector)
        if loc.count() and loc.first.is_visible():
            loc.first.click()
            logger.log({"event": "recoverable_dismissed", "selector": selector})


def _verify_checkpoint(page: Page, checkpoint: Checkpoint) -> None:
    if checkpoint.kind == "url_pattern":
        if not re.search(checkpoint.pattern, page.url):
            raise HardFailure("checkpoint", f"url matching {checkpoint.pattern!r}", page.url)
    elif checkpoint.kind == "text_present":
        body_text = page.locator("body").inner_text()
        if checkpoint.text not in body_text:
            raise HardFailure("checkpoint", f"text {checkpoint.text!r} present", "text not found on page")
    elif checkpoint.kind == "element_present":
        resolve(page, [checkpoint.locator], "checkpoint")  # raises HardFailure itself if absent


def _validate_params(capability: Capability, params: dict[str, str]) -> None:
    for p in capability.input_params:
        if p.required and p.name not in params:
            raise HardFailure("params", f"required param {p.name!r} supplied", "missing")


def replay(
    capability: Capability,
    params: dict[str, str],
    run_id: str,
    headless: bool = True,
    allow_escalation: bool = False,
    allow_assisted_fallback: bool = False,
    require_approved: bool = False,
) -> ReplayResult:
    policy = load_policy()
    error_sigs = load_error_signatures()
    logger = EvidenceLogger(run_id, "replay")

    declared_sensitive = {p.name: p.sensitive for p in capability.input_params}
    logger.log({
        "event": "replay_start",
        "capability": capability.id,
        "params": policy.redact_params(params, declared_sensitive),
    })

    if require_approved and capability.status != "approved":
        # The one real "unattended" caller in this project (the capability
        # API) always sets this. Rejected before a browser is even launched --
        # cheap, fast, and not a crash: it's a policy decision, not an error.
        logger.log({"event": "rejected_not_approved", "status": capability.status})
        return ReplayResult(
            status="failure",
            error={
                "step_id": "approval",
                "expected": "an approved capability",
                "observed": f"capability status is {capability.status!r}",
            },
        )

    try:
        _validate_params(capability, params)
        policy.check_domain(capability.target.base_url)
    except HardFailure as e:
        return ReplayResult(status="failure", error={"step_id": e.step_id, "expected": e.expected, "observed": e.observed})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            page.goto(capability.target.base_url)
            _settle(page)
            for step in capability.steps:
                policy.check_action_type(step.action)
                _dismiss_known_interstitials(page, error_sigs, logger)

                try:
                    _execute_step(page, step, params, policy)
                except HardFailure as e:
                    recovered = False
                    if allow_assisted_fallback:
                        try:
                            fallback.recover(page, step, params, policy, logger)
                            recovered = True
                        except HardFailure:
                            pass  # bounded to one attempt -- fall through to escalation/raise below
                    if not recovered:
                        if not allow_escalation:
                            raise
                        escalate(logger, page, capability.name, step.step_id, str(e))
                        _execute_step(page, step, params, policy)  # one retry after human handoff

                _settle(page)
                if step.wait_after_ms:
                    page.wait_for_timeout(step.wait_after_ms)

                logger.log({
                    "event": "step_executed",
                    "step_id": step.step_id,
                    "action": step.action,
                    "value": _redacted_step_value(step, params, declared_sensitive),
                })
                _check_for_app_errors(page, error_sigs)

            _verify_checkpoint(page, capability.checkpoint)
            outputs = _collect_outputs(page, capability)
            logger.log({"event": "replay_success", "outputs": outputs})
            return ReplayResult(status="success", outputs=outputs, message="goal reached and checkpoint verified")

        except BusinessOutcome as e:
            screenshot = logger.screenshot(page, "business_outcome")
            logger.log({"event": "business_outcome", "code": e.code, "message": e.message})
            return ReplayResult(status="business_outcome", outcome_code=e.code, message=e.message, screenshot_path=screenshot)

        except HardFailure as e:
            screenshot = logger.screenshot(page, "failure")
            logger.log({"event": "failure", "step_id": e.step_id, "expected": e.expected, "observed": e.observed})
            return ReplayResult(
                status="failure",
                error={"step_id": e.step_id, "expected": e.expected, "observed": e.observed},
                screenshot_path=screenshot,
            )

        except Exception as e:  # never crash the caller -- surface a structured, debuggable failure
            screenshot = logger.screenshot(page, "failure")
            logger.log({"event": "failure", "step_id": "unknown", "expected": "no unhandled exception", "observed": str(e)})
            return ReplayResult(
                status="failure",
                error={"step_id": "unknown", "expected": "no unhandled exception", "observed": str(e)},
                screenshot_path=screenshot,
            )

        finally:
            browser.close()


def _execute_step(page: Page, step, params: dict[str, str], policy) -> None:
    if step.action == "navigate":
        policy.check_domain(step.value)
        page.goto(step.value)
        return

    if step.action == "click":
        if step.risk == "risky":
            raise HardFailure(step.step_id, "a safe/allowed action", "step is classified risky and blocked by policy")
        coords = step.locators[0] if step.locators and step.locators[0].kind == "coordinates" else None
        if coords:
            # No DOM node exists for a vision-fallback step -- resolve() has
            # nothing to return a Locator for, so drive the mouse directly.
            page.mouse.click(coords.x, coords.y)
        else:
            resolve(page, step.locators, step.step_id).click()
        return

    if step.action == "type":
        value = params[step.param_ref] if step.param_ref else step.value
        coords = step.locators[0] if step.locators and step.locators[0].kind == "coordinates" else None
        if coords:
            page.mouse.click(coords.x, coords.y)
            page.keyboard.type(value or "")
        else:
            resolve(page, step.locators, step.step_id).fill(value or "")
        return

    raise HardFailure(step.step_id, "a known action type", step.action)


def _redacted_step_value(step, params: dict[str, str], declared_sensitive: dict[str, bool]):
    if step.action != "type":
        return step.value
    if step.param_ref and declared_sensitive.get(step.param_ref):
        return "***REDACTED***"
    return step.value if not step.param_ref else params.get(step.param_ref)


def _collect_outputs(page: Page, capability: Capability) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for spec in capability.output_spec:
        loc = resolve(page, [spec.extract], f"output:{spec.name}")
        outputs[spec.name] = loc.input_value() if spec.attribute == "value" else loc.inner_text().strip()
    return outputs
