"""Save/load capability artifacts as JSON files under artifacts/."""
from __future__ import annotations

from pathlib import Path

from src.artifact.schema import Capability

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


def save(capability: Capability) -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f"{capability.id}.json"
    path.write_text(capability.model_dump_json(indent=2))
    return path


def load(path: str | Path) -> Capability:
    return Capability.model_validate_json(Path(path).read_text())
