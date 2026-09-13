"""Minimal agent-facing capability API (stretch goal, Section 8).

    uvicorn src.capabilities.server:app --port 8000

    GET  /capabilities              -> catalog of callable capabilities
    POST /capabilities/{name}/invoke -> run the replay executor, typed args in, structured result out
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException

from src.capabilities.catalog import list_capabilities, to_tool_spec
from src.replay.executor import replay

app = FastAPI(title="interface.ai capability catalog (demo)")


@app.get("/capabilities")
def get_capabilities():
    return [to_tool_spec(c) for c in list_capabilities()]


@app.post("/capabilities/{name}/invoke")
def invoke_capability(name: str, params: dict[str, str]):
    matches = [c for c in list_capabilities() if c.name == name]
    if not matches:
        raise HTTPException(404, f"no capability named {name!r}")
    # This endpoint is the one genuinely "unattended" caller in this project
    # (an agent invoking a capability by name) -- always gated on approval.
    result = replay(matches[0], params, run_id=f"api-{uuid.uuid4().hex[:8]}", require_approved=True)
    return result.model_dump()
