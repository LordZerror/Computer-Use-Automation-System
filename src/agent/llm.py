"""Groq tool-calling wrapper for the discovery loop.

Groq's chat-completions API is OpenAI-compatible, including the
`tools=[...]` / `tool_calls` function-calling shape -- so this is a thin
wrapper (model name + base client), not a different design from what an
Anthropic/OpenAI version would look like. Swapping providers later means
swapping this file, not the loop.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from groq import Groq, RateLimitError

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are a computer-use agent operating a web application on \
behalf of a user goal. You are given a numbered list of currently visible, \
interactive elements and must choose exactly ONE action per turn by calling \
one of the provided tools.

Rules:
- Only reference element indices that are in the CURRENT perception list; \
they change every turn, so never reuse an index from a previous turn.
- Prefer the most direct path to the goal. Do not click things unrelated to it.
- Call `extract` when the goal asks you to read/report a value, giving it a \
short snake_case output_name.
- You may be given a set of AVAILABLE PARAMETERS (e.g. form field values). \
When you type one of those exact values into a field, pass its parameter \
name as `param_name` on the `type` call so it can be re-supplied on future \
invocations instead of hardcoded.
- Call `finish` only once the goal is fully satisfied (e.g. you can see the \
confirmation/checkout page the goal describes).
- Call `escalate` if you are stuck: the goal seems impossible from the \
current page, you've repeated the same action with no progress, or you hit \
something you should not decide alone (e.g. an unexpected payment/legal \
confirmation).
- If an action is reported as BLOCKED by policy, never retry that exact \
action -- it will be blocked again. Re-read the goal: if it's already \
satisfied by what you've done so far, call `finish` immediately instead.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an interactive element by its index.",
            "parameters": {
                "type": "object",
                "properties": {"element_index": {"type": "integer"}},
                "required": ["element_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": "Type text into an input/textarea element by its index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_index": {"type": "integer"},
                    "text": {"type": "string"},
                    "param_name": {
                        "type": "string",
                        "description": "name from AVAILABLE PARAMETERS if this text came from one, else omit",
                    },
                },
                "required": ["element_index", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate the browser to an absolute URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract",
            "description": "Read the visible text of an element and record it as a named output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_index": {"type": "integer"},
                    "output_name": {"type": "string"},
                },
                "required": ["element_index", "output_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Declare the goal accomplished.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "one-line summary of what was accomplished"}
                },
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Stop and ask a human to take over.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


class AgentLLM:
    def __init__(self, api_key: str | None = None):
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def decide(
        self, goal: str, perception_text: str, history: list[str], params: dict[str, str] | None = None
    ) -> ToolCall:
        history_text = "\n".join(history[-10:]) if history else "(no actions yet)"
        params_text = (
            "\n".join(f"  {k} = {v!r}" for k, v in params.items()) if params else "(none)"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"GOAL: {goal}\n\n"
                    f"AVAILABLE PARAMETERS:\n{params_text}\n\n"
                    f"ACTIONS SO FAR:\n{history_text}\n\n"
                    f"CURRENT PAGE:\n{perception_text}\n\n"
                    "Choose the next single action."
                ),
            },
        ]
        response = self._create_with_retry(messages)
        call = response.choices[0].message.tool_calls[0]
        return ToolCall(name=call.function.name, arguments=json.loads(call.function.arguments))

    def _create_with_retry(self, messages: list[dict], max_retries: int = 5):
        """Free-tier Groq rate limits are tight (low tokens-per-minute); a
        computer-use loop that pauses and retries is exactly the 'transient
        slowness' handling this project is otherwise arguing for, so apply
        it to our own LLM calls too instead of failing the whole run."""
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model=MODEL, messages=messages, tools=TOOLS, tool_choice="required", temperature=0,
                )
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(3 + attempt * 2)
