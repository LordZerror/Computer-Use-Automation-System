"""Entry point: discover a capability, then replay it.

    python -m src.cli discover --goal "..." --target-url https://www.saucedemo.com/ ...
    python -m src.cli replay --artifact artifacts/<id>.json --params k=v ...
    python -m src.cli stability --artifact artifacts/<id>.json --params k=v --runs 5   # stretch goal
    python -m src.cli approve --artifact artifacts/<id>.json                          # stretch goal
    python -m src.cli serve   # stretch goal: agent-facing capability API (gated on approval)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from dotenv import load_dotenv

load_dotenv()  # picks up .env (e.g. GROQ_API_KEY) if present; a real env var always wins

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
    viewport = None
    if args.viewport:
        w, h = args.viewport.split("x")
        viewport = (int(w), int(h))

    result = run_discovery(
        goal=args.goal,
        target_url=args.target_url,
        run_id=run_id,
        params=params,
        sensitive_params=sensitive,
        max_steps=args.max_steps,
        headless=args.headless,
        viewport=viewport,
    )

    if not result.success:
        print(f"Discovery did not complete (run_id={run_id}). See evidence/discovery-{run_id}/log.jsonl", file=sys.stderr)
        return 1

    input_params = [
        InputParam(name=k, type="string", required=True, sensitive=(k in sensitive))
        for k in params
    ]
    checkpoint = Checkpoint(kind=args.checkpoint_kind, text=args.checkpoint_text, pattern=args.checkpoint_pattern)
    try:
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
    except ValueError as e:
        # build_capability fails loud on a transcript action it doesn't
        # recognize (see its own docstring) -- surface that the same clean
        # way every other discovery failure is reported, not a raw traceback.
        print(f"Discovery did not complete (run_id={run_id}): {e}", file=sys.stderr)
        return 1
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
        allow_assisted_fallback=args.allow_assisted_fallback,
        require_approved=args.require_approved,
    )
    print(json.dumps(result.model_dump(), indent=2, default=str))
    print(f"Evidence -> evidence/replay-{run_id}/", file=sys.stderr)
    return 0 if result.status != "failure" else 2


def cmd_stability(args: argparse.Namespace) -> int:
    from src.replay.stability import measure_stability  # deferred, same reason as cmd_replay

    capability = store.load(args.artifact)
    params = _parse_kv(args.params)
    result = measure_stability(capability, params, runs=args.runs)

    capability.stability_score = result["score"]
    capability.stability_sample_size = result["runs"]
    store.save(capability)

    print(json.dumps(result, indent=2))
    print(f"Wrote stability_score={result['score']:.2f} (n={result['runs']}) -> {args.artifact}")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    capability = store.load(args.artifact)

    if not args.force:
        if capability.stability_score is None:
            print("Refusing: no stability score on file. Run `stability` first, or pass --force.", file=sys.stderr)
            return 1
        if capability.stability_score < args.min_stability:
            print(
                f"Refusing: stability_score={capability.stability_score:.2f} is below "
                f"--min-stability={args.min_stability}. Pass --force to override.",
                file=sys.stderr,
            )
            return 1

    capability.status = "approved"
    store.save(capability)
    print(f"Approved {capability.id} (stability_score={capability.stability_score}, force={args.force})")
    return 0


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
    d.add_argument("--viewport", default=None, help="WxH, e.g. 400x200 -- match the target's real size for better vision-fallback coordinate grounding")
    d.add_argument("--run-id", default=None)
    d.set_defaults(func=cmd_discover)

    r = sub.add_parser("replay", help="Deterministically replay a saved capability artifact.")
    r.add_argument("--artifact", required=True)
    r.add_argument("--params", nargs="*", default=[])
    r.add_argument("--visible", action="store_true", help="show the browser window instead of running headless")
    r.add_argument("--allow-escalation", action="store_true", help="pause for human handoff on a hard failure instead of returning it immediately")
    r.add_argument("--allow-assisted-fallback", action="store_true", help="on a hard failure, try one bounded, policy-checked LLM recovery for that step before escalation/failure (needs GROQ_API_KEY)")
    r.add_argument("--require-approved", action="store_true", help="refuse to run unless the artifact's status is 'approved' (the capability API always sets this)")
    r.add_argument("--run-id", default=None)
    r.set_defaults(func=cmd_replay)

    st = sub.add_parser("stability", help="Replay an artifact N times and score how consistently it replays (stretch goal).")
    st.add_argument("--artifact", required=True)
    st.add_argument("--params", nargs="*", default=[])
    st.add_argument("--runs", type=int, default=5)
    st.set_defaults(func=cmd_stability)

    ap = sub.add_parser("approve", help="Mark an artifact 'approved' for unattended replay (stretch goal).")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--min-stability", type=float, default=0.8)
    ap.add_argument("--force", action="store_true", help="approve even without a passing (or any) stability score")
    ap.set_defaults(func=cmd_approve)

    s = sub.add_parser("serve", help="Serve the agent-facing capability API (stretch goal).")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
