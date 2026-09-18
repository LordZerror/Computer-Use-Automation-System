import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from playwright.sync_api import sync_playwright

from src.agent import llm as llm_module
from src.agent.llm import AgentLLM
from src.artifact.recorder import build_capability
from src.artifact.schema import Checkpoint, InputParam
from src.replay.executor import _execute_step
from src.safety.policy import load_policy

CANVAS_HTML = Path(__file__).resolve().parent.parent / "fixtures" / "canvas_button.html"


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    p = browser.new_page()
    p.set_content(CANVAS_HTML.read_text())
    yield p
    p.close()


def test_vision_tools_include_finish_and_escalate():
    """Regression: VISION_TOOLS used to grab finish/escalate out of TOOLS by
    position (TOOLS[4], TOOLS[5]); adding select_option/hover/keypress to
    TOOLS shifted those indices and silently handed vision mode keypress/
    navigate instead -- a vision-mode run could then never call finish or
    escalate, only ever end via max_steps/timeout."""
    names = {t["function"]["name"] for t in llm_module.VISION_TOOLS}
    assert names == {"click_at", "type_at", "finish", "escalate"}


def test_canvas_fixture_has_no_perceivable_elements(page):
    """This is the whole reason the fixture exists: it must genuinely force
    the vision fallback to trigger, not just be asserted to."""
    from src.agent import perceive

    assert perceive.snapshot(page) == []


def test_coordinates_step_executes_via_mouse_not_a_locator(page):
    step = SimpleNamespace(
        action="click", risk="safe", step_id="s0",
        locators=[SimpleNamespace(kind="coordinates", x=200, y=100)],
    )
    _execute_step(page, step, {}, load_policy())
    assert page.title() == "Clicked"


def test_recorder_builds_a_coordinates_step_from_a_vision_transcript_entry():
    transcript = [{
        "step_id": "step_0", "action": "click",
        "coordinates": {"x": 200, "y": 100}, "label": "Click me", "risk": "safe",
    }]
    capability = build_capability(
        goal="click the canvas button", base_url="file:///tmp/canvas_button.html", app_id="demo",
        transcript=transcript, input_params=[],
        checkpoint=Checkpoint(kind="text_present", text="Clicked"),
        run_id="r1", capability_id="c1", capability_name="c1",
    )
    step = capability.steps[0]
    assert step.locators[0].kind == "coordinates"
    assert step.locators[0].x == 200
    assert step.locators[0].y == 100


def test_decide_vision_sends_image_and_returns_tool_call(monkeypatch):
    call = SimpleNamespace(function=SimpleNamespace(
        name="click_at", arguments=json.dumps({"x": 200, "y": 100, "label": "Click me"})
    ))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return response

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(llm_module, "Groq", lambda api_key: fake_client)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")

    result = AgentLLM().decide_vision("click the button", "ZmFrZS1wbmc=", [])

    assert result.name == "click_at"
    assert result.arguments == {"x": 200, "y": 100, "label": "Click me"}
    assert captured["model"] == llm_module.VISION_MODEL
    image_part = captured["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert "ZmFrZS1wbmc=" in image_part["image_url"]["url"]
