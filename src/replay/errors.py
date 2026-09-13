"""The three-way replay outcome split (Section 3.3):

  success           -- goal reached, declared outputs returned.
  business_outcome  -- a legitimate answer the caller needs ("no such
                        member", "invalid login") -- not a crash.
  failure           -- a hard, debuggable failure: what step, what was
                        expected, what was observed.

Conflating business_outcome and failure is, per the brief's own glossary,
"the most common design mistake here" -- so it's modeled as three
distinct branches, not a boolean success/fail plus an error string.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "error_signatures.yaml"

ReplayStatus = Literal["success", "business_outcome", "failure"]


class ReplayError(BaseModel):
    step_id: str
    expected: str
    observed: str


class ReplayResult(BaseModel):
    status: ReplayStatus
    outputs: dict[str, object] = {}
    outcome_code: Optional[str] = None   # set when status == business_outcome
    message: str = ""
    error: Optional[ReplayError] = None  # set when status == failure
    screenshot_path: Optional[str] = None


class BusinessOutcome(Exception):
    """Raised internally to unwind to a business_outcome ReplayResult."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class HardFailure(Exception):
    """Raised internally to unwind to a failure ReplayResult."""

    def __init__(self, step_id: str, expected: str, observed: str):
        self.step_id = step_id
        self.expected = expected
        self.observed = observed
        super().__init__(f"[{step_id}] expected {expected!r}, observed {observed!r}")


class ErrorSignatures:
    def __init__(self, config: dict):
        self.error_selectors: list[str] = config.get("error_selectors", [])
        self.patterns: list[dict] = config.get("patterns", [])
        self.dismissible_selectors: list[str] = config.get("dismissible_selectors", [])

    def classify(self, text: str) -> tuple[str, str] | None:
        """Return (category, code) for the first matching pattern, or None
        if the text doesn't match anything known (caller should treat that
        as an unrecognized/hard failure)."""
        lowered = text.lower()
        for p in self.patterns:
            if p["match"].lower() in lowered:
                return p["category"], p["code"]
        return None


def load_error_signatures(path: Path = CONFIG_PATH) -> ErrorSignatures:
    return ErrorSignatures(yaml.safe_load(path.read_text()))
