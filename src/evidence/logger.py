"""Structured evidence for a run (Section 3.5): a JSONL log of what
happened and why, plus screenshots as the richer signal on failure.

Callers are responsible for redacting sensitive values before logging
(see safety.policy.redact_params) -- this module just persists whatever
it's handed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import Page

EVIDENCE_ROOT = Path(__file__).resolve().parent.parent.parent / "evidence"


class EvidenceLogger:
    def __init__(self, run_id: str, kind: str):
        self.run_id = run_id
        self.dir = EVIDENCE_ROOT / f"{kind}-{run_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "log.jsonl"

    def log(self, event: dict) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def screenshot(self, page: Page, name: str) -> str:
        path = self.dir / f"{name}.png"
        page.screenshot(path=str(path))
        return str(path.relative_to(EVIDENCE_ROOT.parent))

    def write_json(self, name: str, data: dict) -> str:
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        return str(path.relative_to(EVIDENCE_ROOT.parent))
