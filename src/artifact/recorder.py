"""Convert a discovery transcript into a Capability artifact.

Decoupled from the raw model transcript on purpose (Section 3.2): this
only looks at *what happened* (actions + the DOM metadata captured at the
moment of each action), never the model's reasoning text, tool-call
plumbing, or retries.
"""
from __future__ import annotations

from src.artifact.schema import Capability, Checkpoint, InputParam, LocatorStrategy, OutputSpec, Step, TargetSpec

_RISK_DESCRIPTIONS = {
    "test_id": "test id is the most stable identifier available",
    "role": "role + accessible name mirrors how a screen reader would find this control",
    "text": "visible text, more brittle than role/test-id but survives markup churn",
    "css": "structural CSS path, most brittle -- last resort when nothing semantic was available",
}


def _ranked_locators(element: dict) -> list[LocatorStrategy]:
    strategies: list[LocatorStrategy] = []
    if element.get("test_id"):
        strategies.append(LocatorStrategy(kind="test_id", test_id=element["test_id"]))
    if element.get("name"):
        strategies.append(LocatorStrategy(kind="role", role=element["role"], name=element["name"]))
        strategies.append(LocatorStrategy(kind="text", text=element["name"]))
    strategies.append(LocatorStrategy(kind="css", css=element["css"]))
    return strategies


def _describe(element: dict) -> str:
    kind = "test_id" if element.get("test_id") else ("role" if element.get("name") else "css")
    return f"Identified via {kind} ({_RISK_DESCRIPTIONS[kind]}); recorded {len(_ranked_locators(element))} fallback strategies."


def build_capability(
    goal: str,
    base_url: str,
    app_id: str,
    transcript: list[dict],
    input_params: list[InputParam],
    checkpoint: Checkpoint,
    run_id: str,
    capability_id: str,
    capability_name: str,
) -> Capability:
    steps: list[Step] = []
    output_spec: list[OutputSpec] = []

    for t in transcript:
        action = t["action"]
        if action == "extract":
            element = t["element"]
            output_spec.append(
                OutputSpec(
                    name=t["output_name"],
                    type="string",
                    description=f"Extracted from {element['role']} \"{element['name']}\" at discovery time.",
                    source_step_id=t["step_id"],
                    extract=_ranked_locators(element)[0],
                    attribute="text",
                )
            )
            continue

        if action == "navigate":
            steps.append(Step(step_id=t["step_id"], action="navigate", value=t["value"], description="Direct navigation."))
            continue

        element = t["element"]
        risk = t.get("risk", "safe")
        if action == "click":
            steps.append(
                Step(
                    step_id=t["step_id"],
                    action="click",
                    locators=_ranked_locators(element),
                    risk=risk,
                    description=_describe(element),
                )
            )
        elif action == "type":
            param_name = t.get("param_name")
            steps.append(
                Step(
                    step_id=t["step_id"],
                    action="type",
                    locators=_ranked_locators(element),
                    value=None if param_name else t["value"],
                    param_ref=param_name,
                    description=_describe(element),
                )
            )

    return Capability(
        id=capability_id,
        name=capability_name,
        goal=goal,
        target=TargetSpec(base_url=base_url, app_id=app_id),
        input_params=input_params,
        steps=steps,
        output_spec=output_spec,
        checkpoint=checkpoint,
        created_from_run_id=run_id,
    )
