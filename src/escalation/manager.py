"""Human-in-the-loop escalation & handoff (Section 3.6).

Scope note taken literally: no separate operator console. The browser the
agent drives is launched non-headless, so "hand the human the live
session" is done by pausing the automation loop and letting the operator
click directly in that same, already-open window -- same cookies, same
DOM, same session, not a fresh one. A `control.json` file in the run's
evidence directory is the seam a real multi-process operator console
would read/write instead of a CLI prompt (who is/should be in control).

The mechanism is real: pause, cede control, signal resume, capture what
the human did. The console UI around it is the mocked part -- documented
here and in REPORT.md.
"""
from __future__ import annotations

import builtins
import json
from datetime import datetime, timezone

from playwright.sync_api import Page

from src.evidence.logger import EvidenceLogger


class InterventionRequest(Exception):
    """Not really an error -- raised to unwind the loop/executor to the
    escalation handler with full context attached."""

    def __init__(self, step_id: str, reason: str):
        self.step_id = step_id
        self.reason = reason
        super().__init__(reason)


def _set_control(logger: EvidenceLogger, controller: str) -> None:
    (logger.dir / "control.json").write_text(
        json.dumps({"controller": controller, "since": datetime.now(timezone.utc).isoformat()})
    )


def escalate(
    logger: EvidenceLogger,
    page: Page,
    goal_or_capability: str,
    step_id: str,
    reason: str,
    input_fn=builtins.input,
) -> str:
    """Pause automation, hand the live session to a human, wait for them to
    hand it back, and record what they did. Returns their free-text note."""
    screenshot = logger.screenshot(page, f"escalation_{step_id}")
    request = {
        "goal_or_capability": goal_or_capability,
        "step_id": step_id,
        "current_url": page.url,
        "reason": reason,
        "screenshot": screenshot,
    }
    logger.write_json(f"intervention_{step_id}", request)
    logger.log({"event": "escalation_raised", **request})
    _set_control(logger, "human")

    print("\n=== INTERVENTION REQUESTED ===")
    print(f"Goal/capability : {goal_or_capability}")
    print(f"Stuck at step   : {step_id}")
    print(f"Reason          : {reason}")
    print(f"Current URL     : {page.url}")
    print("The live browser window is open for you to operate directly.")
    input_fn("Press Enter once you've completed the manual step(s) to hand control back...")

    human_action = input_fn("Briefly describe what you did (for the record): ")
    logger.log({"event": "escalation_resolved", "step_id": step_id, "human_action": human_action})
    _set_control(logger, "agent")
    return human_action
