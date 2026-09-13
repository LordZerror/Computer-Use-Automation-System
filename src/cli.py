"""Entry point: discover a capability, then replay it.

    python -m src.cli discover --goal "..." --target-url https://www.saucedemo.com/ ...
    python -m src.cli replay --artifact artifacts/<id>.json --params k=v ...
    python -m src.cli serve   # stretch goal: agent-facing capability API
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from src.artifact import store
from src.artifact.recorder import build_capability
from src.artifact.schema import Checkpoint, InputParam


def _parse_kv(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        k, _, v = pair.partition("=")
        out[k] = v
    return out


def cmd_discover(args: argparse.Namespace) -> int:
    from src.agent.loop import run_discovery  # deferred: avoids importing Playwright/Groq for `replay`/`serve`

    run_id = args.run_id or f"run{int(time.time())}"
    params = _parse_kv(args.params)
    sensitive = set(args.sensitive)

    result = run_discovery(
        goal=args.goal,
        target_url=args.target_url,
        run_id=run_id,
        params=params,
        sensitive_params=sensitive,
        max_steps=args.max_steps,
        headless=args.headless,
    )

    if not result.success:
        print(f"Discovery did not complete (run_id={run_id}). See evidence/discovery-{run_id}/log.jsonl", file=sys.stderr)
        return 1

    input_params = [
        InputParam(name=k, type="string", required=True, sensitive=(k in sensitive))
        for k in params
    ]
    checkpoint = Checkpoint(kind=args.checkpoint_kind, text=args.checkpoint_text, pattern=args.checkpoint_pattern)
    capability = build_capability(
        goal=args.goal,
        base_url=args.target_url,
        app_id=args.app_id,
        transcript=result.transcript,
        input_params=input_params,
        checkpoint=checkpoint,
        run_id=run_id,
        capability_id=args.capability_id,
        capability_name=args.capability_id,
    )
    path = store.save(capability)
    print(f"Discovery succeeded: {result.summary}")
    print(f"Saved capability artifact -> {path}")
    print(f"Evidence -> evidence/discovery-{run_id}/")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from src.replay.executor import replay  # deferred, same reason as above

    run_id = args.run_id or f"run{int(time.time())}"
    capability = store.load(args.artifact)
    params = _parse_kv(args.params)
    result = replay(
        capability, params, run_id,
        headless=not args.visible,
        allow_escalation=args.allow_escalation,
    )
    print(json.dumps(result.model_dump(), indent=2, default=str))
    print(f"Evidence -> evidence/replay-{run_id}/", file=sys.stderr)
    return 0 if result.status != "failure" else 2


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("src.capabilities.server:app", host="0.0.0.0", port=args.port)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover", help="Run the LLM-driven discovery loop and save a capability artifact.")
    d.add_argument("--goal", required=True)
    d.add_argument("--target-url", required=True)
    d.add_argument("--params", nargs="*", default=[], help="key=value pairs available to the agent (e.g. checkout fields)")
    d.add_argument("--sensitive", nargs="*", default=[], help="param names to redact from artifacts/logs")
    d.add_argument("--app-id", default="demo-app")
    d.add_argument("--capability-id", required=True)
    d.add_argument("--checkpoint-kind", choices=["url_pattern", "text_present", "element_present"], default="text_present")
    d.add_argument("--checkpoint-text", default=None)
    d.add_argument("--checkpoint-pattern", default=None)
    d.add_argument("--max-steps", type=int, default=15)
    d.add_argument("--headless", action="store_true", help="run without a visible browser window")
    d.add_argument("--run-id", default=None)
    d.set_defaults(func=cmd_discover)

    r = sub.add_parser("replay", help="Deterministically replay a saved capability artifact.")
    r.add_argument("--artifact", required=True)
    r.add_argument("--params", nargs="*", default=[])
    r.add_argument("--visible", action="store_true", help="show the browser window instead of running headless")
    r.add_argument("--allow-escalation", action="store_true", help="pause for human handoff on a hard failure instead of returning it immediately")
    r.add_argument("--run-id", default=None)
    r.set_defaults(func=cmd_replay)

    s = sub.add_parser("serve", help="Serve the agent-facing capability API (stretch goal).")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
