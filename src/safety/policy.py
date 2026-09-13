"""Allowlist enforcement, risk classification, and redaction (Section 3.4).

Loaded once from config/allowlist.yaml and consulted by both the discovery
loop (before the LLM's chosen action executes) and the replay executor
(before every step) -- the same policy object, so there is exactly one
place that decides what's permitted.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "allowlist.yaml"
REDACTED = "***REDACTED***"


class PolicyViolation(Exception):
    """An action was blocked by the allowlist. Never caught and retried --
    it means the goal or the model asked for something out of bounds."""


class Policy:
    def __init__(self, config: dict):
        self.allowed_domains: set[str] = set(config.get("allowed_domains", []))
        self.allowed_action_types: set[str] = set(config.get("allowed_action_types", []))
        self.risky_text_markers: list[str] = [
            m.lower() for m in config.get("risky_text_markers", [])
        ]
        self.sensitive_param_names: set[str] = {
            n.lower() for n in config.get("sensitive_param_names", [])
        }

    def check_domain(self, url: str) -> None:
        host = urlparse(url).netloc
        if host not in self.allowed_domains:
            raise PolicyViolation(f"domain {host!r} is not in the allowlist")

    def check_action_type(self, action_type: str) -> None:
        if action_type not in self.allowed_action_types:
            raise PolicyViolation(f"action type {action_type!r} is not in the allowlist")

    def classify_risk(self, control_text: str | None) -> str:
        """safe/reversible vs. risky/irreversible, per 3.4. Conservative:
        anything matching a risky marker is 'risky' regardless of action type."""
        if not control_text:
            return "safe"
        lowered = control_text.lower()
        return "risky" if any(m in lowered for m in self.risky_text_markers) else "safe"

    def is_sensitive_param(self, name: str, declared_sensitive: bool) -> bool:
        return declared_sensitive or name.lower() in self.sensitive_param_names

    def redact_params(self, params: dict[str, object], declared: dict[str, bool]) -> dict[str, object]:
        """Mask sensitive values before they're written to an artifact or a log line."""
        return {
            k: (REDACTED if self.is_sensitive_param(k, declared.get(k, False)) else v)
            for k, v in params.items()
        }


def load_policy(path: Path = CONFIG_PATH) -> Policy:
    return Policy(yaml.safe_load(path.read_text()))
