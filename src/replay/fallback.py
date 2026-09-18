"""Assisted fallback (Section 8, stretch goal): on a replay step that fails
to resolve, allow exactly ONE bounded, policy-checked LLM call to recover --
never open-ended. This is narrower than the discovery loop on purpose: the
model is never asked to decide *what* to do, only *which currently-visible
element* matches a step that already has a fixed, recorded action. No loop,
no new action types, no navigation -- so it can't drift into open-ended
agentic behavior, and it's still subject to the same risk policy a normal
click would get.

Off by default: `replay()` only calls this when `allow_assisted_fallback=True`
is passed explicitly.
"""
from __future__ import annotations

import json
import os

from groq import Groq
from playwright.sync_api import Page

from src.agent import perceive
from src.agent.llm import MODEL
from src.artifact.schema import Step
from src.evidence.logger import EvidenceLogger
from src.replay.errors import HardFailure
from src.safety.policy import Policy

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pick_element",
            "description": "The element at this index matches what the failed step was trying to act on.",
            "parameters": {"type": "object", "properties": {"element_index": {"type": "integer"}}, "required": ["element_index"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "none_found",
            "description": "None of the current elements match what the step needed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _describe_original_target(step: Step) -> str:
    parts = []
    for loc in step.locators:
        if loc.kind == "test_id":
            parts.append(f"test_id={loc.test_id}")
        elif loc.kind == "role":
            parts.append(f"role={loc.role} name={loc.name!r}")
        elif loc.kind == "text":
            parts.append(f"text={loc.text!r}")
        elif loc.kind == "css":
            parts.append(f"css={loc.css}")
    return "; ".join(parts) or "(no locator info recorded)"


def recover(page: Page, step: Step, params: dict[str, str], policy: Policy, logger: EvidenceLogger) -> None:
    """Attempt one bounded recovery for `step`. Executes the step's action
    on success; raises HardFailure (never retried again by this function)
    on any failure to recover."""
    elements = perceive.snapshot(page)
    logger.log({"event": "assisted_fallback_attempted", "step_id": step.step_id})

    if not elements:
        logger.log({"event": "assisted_fallback_not_found", "step_id": step.step_id, "reason": "no elements on page"})
        raise HardFailure(step.step_id, "at least one candidate element", "no interactive elements perceived")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = [
        {
            "role": "system",
            "content": (
                "A recorded UI automation step failed to find its target element. "
                "You get exactly one attempt to identify the right element from the "
                "CURRENT page so this single step can be retried -- do not reason "
                "about anything beyond picking the matching element."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Failed step description: {step.description}\n"
                f"Original locators it can no longer resolve: {_describe_original_target(step)}\n\n"
                f"{perceive.format_for_llm(elements, page.url)}\n\n"
                "Call pick_element with the matching index, or none_found."
            ),
        },
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=_TOOLS, tool_choice="required", temperature=0)
    call = response.choices[0].message.tool_calls[0]

    if call.function.name == "none_found":
        logger.log({"event": "assisted_fallback_not_found", "step_id": step.step_id})
        raise HardFailure(step.step_id, "the assisted fallback to find a match", "model reported none_found")

    idx = json.loads(call.function.arguments).get("element_index", -1)
    if not (0 <= idx < len(elements)):
        logger.log({"event": "assisted_fallback_not_found", "step_id": step.step_id, "reason": f"invalid index {idx}"})
        raise HardFailure(step.step_id, "a valid element index from the assisted fallback", f"invalid index {idx}")

    element = elements[idx]
    if step.action == "select_option" and not perceive.is_native_select(element):
        # Same guard as agent/loop.py: a JS-built listbox got recovered
        # under a step recorded against a real <select> -- Playwright's
        # .select_option() only works on the real thing.
        logger.log({"event": "assisted_fallback_not_found", "step_id": step.step_id, "reason": "recovered element is not a native <select>"})
        raise HardFailure(step.step_id, "a native <select> element", f"recovered element {element['name']!r} has no options (not a <select>)")

    # Same rule agent/loop.py's discovery-time check uses, via the one
    # shared `classify_action_risk` -- this used to be a separate,
    # hand-written copy of that logic here, which is exactly how it drifted
    # out of sync with loop.py's keypress fix the first time around.
    risk = policy.classify_action_risk(step.action, element["name"], step.value)
    if risk == "risky":
        logger.log({"event": "assisted_fallback_declined_risky", "step_id": step.step_id, "control": element["name"]})
        raise HardFailure(step.step_id, "a safe/allowed recovered element", f"recovered element {element['name']!r} is risky; blocked")

    locator = page.locator(element["css"]).first
    if step.action == "click":
        locator.click()
    elif step.action == "type":
        value = params[step.param_ref] if step.param_ref else step.value
        locator.fill(value or "")
    elif step.action == "select_option":
        locator.select_option(label=step.value)
    elif step.action == "hover":
        locator.hover()
    elif step.action == "keypress":
        locator.press(step.value)
    else:
        raise HardFailure(step.step_id, "a known recoverable action type", step.action)

    logger.log({"event": "assisted_fallback_recovered", "step_id": step.step_id, "element": element["name"]})
