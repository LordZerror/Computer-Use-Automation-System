"""The capability artifact: a typed, versioned, agent-invocable contract.

This is the seam between "the model discovered how to do this once" and
"an AI agent can call this reliably in production." Everything here is
plain, JSON-serializable Pydantic — no behavior, just the contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

ActionType = Literal["navigate", "click", "type"]
LocatorKind = Literal["role", "test_id", "text", "css", "xpath", "coordinates"]
ParamType = Literal["string", "number", "boolean"]
RiskLevel = Literal["safe", "risky"]
CheckpointKind = Literal["url_pattern", "text_present", "element_present"]


class LocatorStrategy(BaseModel):
    """One way to find a control. Steps carry a ranked list of these —
    replay tries them in order so a single dead attribute doesn't fail
    the whole step. Field-per-kind keeps each strategy self-describing
    instead of a stringly-typed selector nobody can review.
    """

    kind: LocatorKind
    role: Optional[str] = None      # kind="role": ARIA role, e.g. "button"
    name: Optional[str] = None      # kind="role": accessible name
    test_id: Optional[str] = None   # kind="test_id"
    text: Optional[str] = None      # kind="text": visible text (exact or substring)
    css: Optional[str] = None       # kind="css"
    xpath: Optional[str] = None     # kind="xpath": last resort, most brittle
    x: Optional[float] = None       # kind="coordinates": pixel position, no DOM node exists
    y: Optional[float] = None       # kind="coordinates": (e.g. a canvas-drawn control)


class InputParam(BaseModel):
    name: str
    type: ParamType
    required: bool = True
    description: str = ""
    sensitive: bool = False  # redact from artifacts/logs (credentials, PII)


class OutputSpec(BaseModel):
    """A declared output is resolved against final page state after all
    steps run and the checkpoint passes -- it's a read-only contract, not
    another mutating step. `source_step_id` is informational (which
    discovery step this was learned from), not a replay dependency."""

    name: str
    type: ParamType
    description: str = ""
    source_step_id: str
    extract: LocatorStrategy
    attribute: Literal["text", "value"] = "text"


class Checkpoint(BaseModel):
    """What proves the goal was actually reached, not just that clicks fired."""

    kind: CheckpointKind
    pattern: Optional[str] = None       # url_pattern: regex against page URL
    text: Optional[str] = None          # text_present: substring expected on page
    locator: Optional[LocatorStrategy] = None  # element_present


class Step(BaseModel):
    step_id: str
    action: ActionType
    locators: list[LocatorStrategy] = Field(default_factory=list)
    value: Optional[str] = None       # literal (navigate URL, type text)
    param_ref: Optional[str] = None   # if set, value comes from this input_param at replay time,
                                       # never a literal in the artifact (this is how secrets/PII
                                       # stay out of persisted artifacts -- see InputParam.sensitive)
    risk: RiskLevel = "safe"
    wait_after_ms: int = 0
    description: str = ""             # why this locator/step is expected to be robust


class TargetSpec(BaseModel):
    base_url: str
    app_id: str


class Capability(BaseModel):
    """A reusable, reviewable, parameterized flow an AI agent can invoke by name."""

    id: str
    name: str
    version: str = "1.0.0"
    goal: str
    target: TargetSpec
    input_params: list[InputParam] = Field(default_factory=list)
    steps: list[Step]
    output_spec: list[OutputSpec] = Field(default_factory=list)
    checkpoint: Checkpoint
    created_from_run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Confidence & approval gating (Section 8 stretch goal). New artifacts
    # start "draft" and unmeasured; see replay/stability.py + `cli.py approve`.
    status: Literal["draft", "approved"] = "draft"
    stability_score: Optional[float] = None
    stability_sample_size: int = 0
