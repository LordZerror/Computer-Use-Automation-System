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

# Every action that resolves and acts on a single element gets a risk
# classification (Section 3.4) -- one shared constant instead of the same
# literal tuple hardcoded separately in artifact/recorder.py (which action
# gets a Step built with a risk field at all) and replay/executor.py (which
# action's step.risk gets checked before running). This is exactly the kind
# of duplicated-knowledge drift that let keypress's risk check go unfixed in
# one of two places the first time around -- see Policy.classify_action_risk.
RISK_GATED_ACTIONS = frozenset({"click", "select_option", "hover", "keypress"})


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
        parsed = urlparse(url)
        if parsed.scheme == "file":
            # A local fixture on the operator's own machine isn't the
            # exfiltration/lateral-movement risk a remote-domain allowlist
            # exists to prevent -- used for the vision-fallback demo target.
            return
        host = parsed.netloc
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

    def classify_action_risk(self, action: str, control_text: str | None, value: str | None = None) -> str:
        """The one place that decides *what text* gets risk-classified for a
        given action -- shared by the discovery loop and replay's assisted
        fallback so the rule can't drift between the two (it already did
        once: keypress was fixed to fold in its key here, in loop.py, but
        not in fallback.py's separate copy, until both were pointed at this).

        Every action, `hover` included, classifies by the same
        risky_text_markers rule -- hover doesn't mutate the page on its own,
        but a hover-triggered handler on a destructively-named control isn't
        impossible, and there's no reason to special-case it away from the
        one generic gate everything else goes through. `select_option`/
        `keypress` additionally fold `value` (the chosen option / the key
        sent) into the classified text -- a neutrally-named control can
        still have a risky option, or send a risky key, that the control's
        own name alone would never catch.
        """
        if action in ("select_option", "keypress") and value:
            control_text = f"{control_text or ''} {value}".strip()
        return self.classify_risk(control_text)

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
