import json

import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools import JudgeToolRegistry
from mlflow.genai.judges.tools.read_skill import ReadSkillTool
from mlflow.genai.skills import SkillSet
from mlflow.types.llm import FunctionToolCallArguments, ToolCall


@pytest.fixture
def skills(tmp_path):
    skill_path = tmp_path / "my-skill"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text("---\nname: my-skill\ndescription: Test.\n---\nBody.")
    return SkillSet([skill_path])


@pytest.fixture
def trace():
    trace_info = TraceInfo(
        trace_id="test-id",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=0,
        state=TraceState.OK,
        execution_duration=100,
    )
    return Trace(trace_info, {})


def test_registry_invokes_skill_tool_with_skills(skills):
    registry = JudgeToolRegistry()
    registry.register(ReadSkillTool())

    tool_call = ToolCall(
        id="call_1",
        function=FunctionToolCallArguments(
            name="read_skill", arguments=json.dumps({"skill_name": "my-skill"})
        ),
    )
    result = registry.invoke(tool_call=tool_call, trace=None, skills=skills)
    assert "Body" in result


def test_registry_invokes_trace_tool_with_trace(trace):
    from mlflow.genai.judges.tools.get_trace_info import GetTraceInfoTool

    registry = JudgeToolRegistry()
    registry.register(GetTraceInfoTool())

    tool_call = ToolCall(
        id="call_2",
        function=FunctionToolCallArguments(name="get_trace_info", arguments="{}"),
    )
    result = registry.invoke(tool_call=tool_call, trace=trace, skills=None)
    assert "test-id" in str(result)
