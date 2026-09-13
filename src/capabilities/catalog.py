"""Stretch goal (Section 8): expose saved artifacts as a catalog of
callable capabilities -- a small function-calling surface an AI agent
could discover and invoke by name with typed args."""
from __future__ import annotations

from src.artifact.schema import Capability
from src.artifact.store import ARTIFACTS_DIR, load


def list_capabilities() -> list[Capability]:
    if not ARTIFACTS_DIR.exists():
        return []
    return [load(p) for p in sorted(ARTIFACTS_DIR.glob("*.json"))]


def to_tool_spec(capability: Capability) -> dict:
    """The same JSON-schema-function shape an LLM tool-calling API expects,
    generated straight from the artifact's own input/output contract."""
    return {
        "name": capability.name,
        "description": capability.goal,
        "version": capability.version,
        "parameters": {
            "type": "object",
            "properties": {
                p.name: {"type": p.type, "description": p.description}
                for p in capability.input_params
            },
            "required": [p.name for p in capability.input_params if p.required],
        },
        "returns": {
            "type": "object",
            "properties": {o.name: {"type": o.type, "description": o.description} for o in capability.output_spec},
        },
    }
