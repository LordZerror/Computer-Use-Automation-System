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

    logger.log({"event": "discovery_start", "goal": goal, "target": target_url})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(target_url)
        _settle(page)

        for i in range(max_steps):
            if time.time() - start > timeout_s:
                logger.log({"event": "stopped", "reason": "timeout"})
                break

            elements = perceive.snapshot(page)
            if not elements:
                escalate(logger, page, goal, f"step_{i}", "no interactive elements perceived on this page")
                break

            perception_text = perceive.format_for_llm(elements, page.url)
            call = llm.decide(goal, perception_text, history, params)
            logger.log({"event": "llm_decision", "step": i, "tool": call.name, "args": call.arguments})

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
                policy.check_action_type(call.name)
            except Exception as e:
                history.append(f"[blocked by policy: {e}]")
                logger.log({"event": "policy_blocked", "action": call.name, "reason": str(e)})
                continue

            step_id = f"step_{len(transcript)}"

            if call.name in ("click", "type", "extract"):
                idx = call.arguments.get("element_index", -1)
                if not (0 <= idx < len(elements)):
                    history.append(f"[invalid element index {idx}, ignored]")
                    continue
                element = elements[idx]
            else:
                element = None

            if call.name == "click":
                risk = policy.classify_risk(element["name"])
                if risk == "risky":
                    history.append(f"[BLOCKED click on risky control '{element['name']}']")
                    logger.log({"event": "policy_blocked_risky", "control": element["name"]})
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

        logger.screenshot(page, "final")
        browser.close()

    logger.log({"event": "discovery_end", "success": success, "steps_recorded": len(transcript)})
    return DiscoveryResult(success=success, transcript=transcript, outputs=outputs, summary=summary, run_id=run_id)
