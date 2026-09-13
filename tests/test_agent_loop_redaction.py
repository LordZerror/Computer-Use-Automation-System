from src.agent.llm import ToolCall
from src.agent.loop import _redact_call_args


def test_sensitive_typed_value_is_redacted_before_logging():
    """Regression: the discovery log used to write raw LLM tool-call
    arguments straight to evidence, leaking a password typed via
    param_name='password' in plaintext."""
    call = ToolCall(name="type", arguments={"element_index": 1, "param_name": "password", "text": "secret_sauce"})
    redacted = _redact_call_args(call, {"password"})
    assert redacted["text"] == "***REDACTED***"
    assert redacted["element_index"] == 1


def test_non_sensitive_typed_value_is_left_alone():
    call = ToolCall(name="type", arguments={"element_index": 0, "param_name": "username", "text": "standard_user"})
    redacted = _redact_call_args(call, {"password"})
    assert redacted["text"] == "standard_user"


def test_non_type_calls_are_unaffected():
    call = ToolCall(name="click", arguments={"element_index": 3})
    assert _redact_call_args(call, {"password"}) == {"element_index": 3}
