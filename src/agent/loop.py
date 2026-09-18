"""The goal-driven observe -> decide -> act loop (Section 3.1).

Runs an LLM-driven loop against a live, visible (non-headless) browser
until the goal is met, the model calls escalate, or a stopping condition
fires (max steps / timeout / no perceivable elements). Produces a
transcript (actions + the DOM metadata captured at the moment of each
action) that artifact.recorder turns into a Capability -- this module
never touches the artifact schema directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from playwright.sync_api import Page, sync_playwright

from src.agent import perceive
from src.agent.llm import AgentLLM
from src.escalation.manager import escalate
from src.evidence.logger import EvidenceLogger
from src.safety.policy import load_policy


@dataclass
class DiscoveryResult:
    success: bool
    transcript: list[dict] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    run_id: str = ""


def _locator_from_element(page: Page, element: dict):
    return page.locator(element["css"]).first


def _classify_and_block_if_risky(
    policy, action_name: str, control_label: str, history: list, logger, value: str | None = None,
) -> str:
    """Classify risk via `policy.classify_action_risk` (the shared rule
    replay's assisted fallback also uses) and, if risky, append the standard
    block message + log the event. Returns the risk level -- caller must
    `continue` the loop when it comes back "risky" rather than run the
    action."""
    risk = policy.classify_action_risk(action_name, control_label, value)
    if risk == "risky":
        # Name the value too (not just the control) when there is one, so
        # e.g. a model told "'Account Actions' choosing 'Delete' is blocked"
        # can learn a different, safe option on the same control is fine --
        # "'Account Actions' is blocked" alone gives it no way to know that.
        detail = f"'{control_label}' (value: {value!r})" if value else f"'{control_label}'"
        history.append(
            f"[BLOCKED {action_name} on risky/irreversible control {detail} by policy -- "
            "do not retry it. If the stated goal is already satisfied, call finish now.]"
        )
        logger.log({"event": "policy_blocked_risky", "control": control_label, "value": value})
    return risk


# Vision-mode tool names map onto the same allowlisted action types as their
# DOM-mode counterparts -- a coordinate click is still a "click" for policy
# purposes, just located differently.
_POLICY_ACTION_FOR_TOOL = {"click_at": "click", "type_at": "type"}


def _redact_call_args(call, sensitive_params: set[str]) -> dict:
    """Never let a sensitive value reach the log, even as a raw tool-call
    argument -- redaction has to happen before the very first place anything
    gets written, not just in the human-readable history trail."""
    if call.name == "type" and call.arguments.get("param_name") in sensitive_params:
        return {**call.arguments, "text": "***REDACTED***"}
    return call.arguments


def _settle(page: Page) -> None:
    """This target (and, we'd guess, plenty of real enterprise apps) finishes
    its network activity before it finishes rendering -- content can still be
    hydrating a couple seconds after `networkidle`. Treat that as the
    'transient slowness' runtime condition from Section 1, not a bug: wait a
    beat past idle rather than perceiving/asserting against a half-drawn page."""
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass
    page.wait_for_timeout(500)


def run_discovery(
    goal: str,
    target_url: str,
    run_id: str,
    params: dict[str, str] | None = None,
    sensitive_params: set[str] | None = None,
    max_steps: int = 15,
    timeout_s: int = 180,
    headless: bool = False,
    viewport: tuple[int, int] | None = None,
) -> DiscoveryResult:
    params = params or {}
    sensitive_params = sensitive_params or set()
    policy = load_policy()
    policy.check_domain(target_url)
    logger = EvidenceLogger(run_id, "discovery")
    llm = AgentLLM()

    transcript: list[dict] = []
    outputs: dict[str, str] = {}
    history: list[str] = []
    start = time.time()
    success = False
    summary = ""
    last_failed_signature: tuple | None = None
    consecutive_identical_failures = 0

    logger.log({"event": "discovery_start", "goal": goal, "target": target_url})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        # A viewport much larger than the actual content hurts a vision
        # model's coordinate grounding -- worth controlling per-target rather
        # than always taking Playwright's default.
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]} if viewport else None)
        # Discovery should fail fast on a wrong guess (Playwright's 30s
        # default actionability wait means one bad element_index burns 30
        # real seconds -- observed live against automationexercise.com,
        # three wrong guesses in a row cost 90s for nothing). A replay of an
        # already-verified artifact keeps the normal default (replay/executor.py).
        page.set_default_timeout(5000)
        page.goto(target_url)
        _settle(page)

        for i in range(max_steps):
            if time.time() - start > timeout_s:
                logger.log({"event": "stopped", "reason": "timeout"})
                break

            elements = perceive.snapshot(page)
            vision_mode = not elements

            if vision_mode:
                screenshot_b64 = perceive.screenshot_fallback_b64(page)
                call = llm.decide_vision(goal, screenshot_b64, history)
            else:
                perception_text = perceive.format_for_llm(elements, page.url)
                call = llm.decide(goal, perception_text, history, params)
            logger.log({
                "event": "vision_decision" if vision_mode else "llm_decision", "step": i, "tool": call.name,
                "args": _redact_call_args(call, sensitive_params),
            })

            if call.name == "finish":
                summary = call.arguments.get("summary", "")
                success = True
                logger.log({"event": "finished", "summary": summary})
                break

            if call.name == "escalate":
                note = escalate(logger, page, goal, f"step_{i}", call.arguments.get("reason", ""))
                history.append(f"[human intervened: {note}]")
                continue

            try:
                policy.check_action_type(_POLICY_ACTION_FOR_TOOL.get(call.name, call.name))
            except Exception as e:
                history.append(f"[blocked by policy: {e}]")
                logger.log({"event": "policy_blocked", "action": call.name, "reason": str(e)})
                continue

            step_id = f"step_{len(transcript)}"

            try:
                if call.name == "click_at":
                    x, y, label = call.arguments["x"], call.arguments["y"], call.arguments.get("label", "")
                    risk = _classify_and_block_if_risky(policy, "click", label, history, logger)
                    if risk == "risky":
                        continue
                    page.mouse.click(x, y)
                    _settle(page)
                    transcript.append({"step_id": step_id, "action": "click", "coordinates": {"x": x, "y": y}, "label": label, "risk": risk})
                    history.append(f"clicked at ({x}, {y}) [{label}]")
                    continue

                if call.name == "type_at":
                    x, y, text, label = call.arguments["x"], call.arguments["y"], call.arguments["text"], call.arguments.get("label", "")
                    page.mouse.click(x, y)
                    page.keyboard.type(text)
                    _settle(page)
                    transcript.append({"step_id": step_id, "action": "type", "coordinates": {"x": x, "y": y}, "value": text, "label": label})
                    history.append(f"typed '{text}' at ({x}, {y}) [{label}]")
                    continue

                if call.name in ("click", "type", "extract", "select_option", "hover", "keypress"):
                    idx = call.arguments.get("element_index", -1)
                    if not (0 <= idx < len(elements)):
                        history.append(f"[invalid element index {idx}, ignored]")
                        continue
                    element = elements[idx]
                else:
                    element = None

                if call.name == "click":
                    risk = _classify_and_block_if_risky(policy, "click", element["name"], history, logger)
                    if risk == "risky":
                        continue
                    _locator_from_element(page, element).click()
                    _settle(page)
                    transcript.append({"step_id": step_id, "action": "click", "element": element, "risk": risk})
                    history.append(f"clicked [{idx}] {element['role']} \"{element['name']}\"")

                elif call.name == "type":
                    text = call.arguments["text"]
                    param_name = call.arguments.get("param_name")
                    _locator_from_element(page, element).fill(text)
                    _settle(page)
                    transcript.append({
                        "step_id": step_id, "action": "type", "element": element,
                        "value": text, "param_name": param_name,
                    })
                    logged = "***REDACTED***" if param_name in sensitive_params else text
                    history.append(f"typed '{logged}' into [{idx}] {element['name']}")

                elif call.name == "select_option":
                    value = call.arguments["value"]
                    if not perceive.is_native_select(element):
                        # Not a native <select> -- a JS-built listbox widget
                        # (e.g. <div role="combobox">) gets the same
                        # "combobox" role in perception but Playwright's
                        # .select_option() only works on the real thing, and
                        # would otherwise fail with a generic, confusing
                        # error instead of this actionable one.
                        history.append(f"[select_option failed: [{idx}] {element['name']!r} is not a native <select> element (no options detected) -- try click/hover instead]")
                        continue
                    risk = _classify_and_block_if_risky(policy, "select_option", element["name"], history, logger, value=value)
                    if risk == "risky":
                        continue
                    _locator_from_element(page, element).select_option(label=value)
                    _settle(page)
                    transcript.append({"step_id": step_id, "action": "select_option", "element": element, "value": value, "risk": risk})
                    history.append(f"selected '{value}' on [{idx}] {element['name']}")

                elif call.name == "hover":
                    risk = _classify_and_block_if_risky(policy, "hover", element["name"], history, logger)
                    if risk == "risky":
                        continue
                    _locator_from_element(page, element).hover()
                    _settle(page)
                    transcript.append({"step_id": step_id, "action": "hover", "element": element, "risk": risk})
                    history.append(f"hovered [{idx}] {element['role']} \"{element['name']}\"")

                elif call.name == "keypress":
                    key = call.arguments["key"]
                    risk = _classify_and_block_if_risky(policy, "keypress", element["name"], history, logger, value=key)
                    if risk == "risky":
                        continue
                    _locator_from_element(page, element).press(key)
                    _settle(page)
                    transcript.append({"step_id": step_id, "action": "keypress", "element": element, "value": key, "risk": risk})
                    history.append(f"pressed '{key}' on [{idx}] {element['name']}")

                elif call.name == "navigate":
                    url = call.arguments["url"]
                    policy.check_domain(url)
                    page.goto(url)
                    _settle(page)
                    transcript.append({"step_id": step_id, "action": "navigate", "value": url})
                    history.append(f"navigated to {url}")

                elif call.name == "extract":
                    output_name = call.arguments["output_name"]
                    value = _locator_from_element(page, element).inner_text().strip()
                    outputs[output_name] = value
                    transcript.append({
                        "step_id": step_id, "action": "extract", "element": element, "output_name": output_name,
                    })
                    history.append(f"extracted {output_name}='{value}' from [{idx}] {element['name']}")

            except Exception as e:
                # A live surface can make any action throw (a modal
                # intercepting a click, a detached/stale element, a
                # navigation timeout) -- this is "transient slowness"/"dead
                # end" territory (Section 1), not a reason to crash the whole
                # discovery run. Surface it to the model as feedback instead.
                short_error = str(e).splitlines()[0][:200]
                history.append(f"[action {call.name} failed: {short_error} -- try a different approach]")
                logger.log({"event": "action_failed", "tool": call.name, "error": short_error})

                # Observed live: the model doesn't reliably follow its own
                # "don't retry a failed action" instruction and can loop on
                # the exact same failing call. Don't trust that judgment
                # call to the model alone -- this is precisely the "repeated
                # the same action with no progress" stuck-state Section 3.6
                # asks us to detect, so enforce it rather than hope for it.
                signature = (call.name, tuple(sorted(call.arguments.items())))
                consecutive_identical_failures = (
                    consecutive_identical_failures + 1 if signature == last_failed_signature else 1
                )
                last_failed_signature = signature
                if consecutive_identical_failures >= 2:
                    reason = f"repeated the identical failing action ({call.name}) {consecutive_identical_failures}x in a row"
                    if headless:
                        # No human is watching a headless run to hand control
                        # to -- escalate()'s input() prompt would just hang.
                        # Stop as a dead-end instead of pretending to escalate.
                        logger.log({"event": "stopped", "reason": f"dead_end: {reason}"})
                    else:
                        note = escalate(logger, page, goal, step_id, reason)
                        history.append(f"[human intervened: {note}]")
                        consecutive_identical_failures = 0
                        continue
                    break

        logger.screenshot(page, "final")
        browser.close()

    logger.log({"event": "discovery_end", "success": success, "steps_recorded": len(transcript)})
    return DiscoveryResult(success=success, transcript=transcript, outputs=outputs, summary=summary, run_id=run_id)
