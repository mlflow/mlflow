from unittest.mock import patch

import crewai
import pytest
from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, start
from crewai.tools import BaseTool
from packaging.version import Version

import mlflow
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces

# This is a special word for CrewAI to complete the agent execution: https://github.com/crewAIInc/crewAI/blob/c6a6c918e0eba167be1fb82831c73dd664c641e3/src/crewai/agents/parser.py#L7
_FINAL_ANSWER_KEYWORD = "Final Answer:"

_LLM_ANSWER = "What about Tokyo?"


def create_sample_llm_response(content):
    from litellm import ModelResponse

    return ModelResponse(
        **{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        }
    )


_SIMPLE_CHAT_COMPLETION = create_sample_llm_response(f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}")

# Special keywords for tool calling in crewai
_TOOL_CHAT_COMPLETION = create_sample_llm_response("""
                    Action: TestTool
                    Action Input: {"argument": "a"}
                """)

_EMBEDDING = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0, 0, 0], "index": 0}],
    "model": "text-embedding-ada-002",
    "usage": {"prompt_tokens": 8, "total_tokens": 8},
}


class AnyInt(int):
    def __eq__(self, other):
        return isinstance(other, int)


ANY_INT = AnyInt()

_CREW_OUTPUT = {
    "json_dict": None,
    "pydantic": None,
    "raw": _LLM_ANSWER,
    "tasks_output": [
        {
            "agent": "City Selection Expert",
            "name": None,
            "description": "Analyze and select the best city for the trip",
            "expected_output": "Detailed report on the chosen city",
            "json_dict": None,
            "pydantic": None,
            "output_format": "raw",
            "raw": _LLM_ANSWER,
            "summary": "Analyze and select the best city for the trip...",
        }
    ],
    "token_usage": {
        "cached_prompt_tokens": ANY_INT,
        "completion_tokens": ANY_INT,
        "prompt_tokens": ANY_INT,
        "successful_requests": ANY_INT,
        "total_tokens": ANY_INT,
    },
}

_AGENT_1_GOAL = "Select the best city based on weather, season, and prices"
_AGENT_1_BACKSTORY = "An expert in analyzing travel data to pick ideal destinations"


@pytest.fixture
def simple_agent_1():
    return Agent(
        role="City Selection Expert",
        goal=_AGENT_1_GOAL,
        backstory=_AGENT_1_BACKSTORY,
        tools=[],
    )


_AGENT_2_GOAL = "Provide the BEST insights about the selected city"


@pytest.fixture
def simple_agent_2():
    return Agent(
        role="Local Expert at this city",
        goal=_AGENT_2_GOAL,
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[],
    )


class SampleTool(BaseTool):
    name: str = "TestTool"
    description: str = "test tool"

    def _run(self, argument: str) -> str:
        return "Tool Answer"


@pytest.fixture
def tool_agent_1():
    return Agent(
        role="City Selection Expert",
        goal=_AGENT_1_GOAL,
        backstory=_AGENT_1_BACKSTORY,
        tools=[SampleTool()],
    )


_TASK_1_DESCRIPTION = "Analyze and select the best city for the trip"
_TASK_1_OUTPUT = "Detailed report on the chosen city"


@pytest.fixture
def task_1(simple_agent_1):
    return Task(
        description=(_TASK_1_DESCRIPTION),
        agent=simple_agent_1,
        expected_output=_TASK_1_OUTPUT,
    )


@pytest.fixture
def task_1_with_tool(tool_agent_1):
    return Task(
        description=(_TASK_1_DESCRIPTION),
        agent=tool_agent_1,
        expected_output=_TASK_1_OUTPUT,
    )


_TASK_2_DESCRIPTION = "Compile an in-depth guide"


@pytest.fixture
def task_2(simple_agent_2):
    return Task(
        description=(_TASK_2_DESCRIPTION),
        agent=simple_agent_2,
        expected_output="Comprehensive city guide",
    )


def global_autolog():
    # Libraries used within tests or crewai library
    mlflow.autolog(exclude_flavors=["openai", "litellm", "langchain"])
    mlflow.utils.import_hooks.notify_module_loaded(crewai)


def clear_autolog_state():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for key in AUTOLOGGING_INTEGRATIONS.keys():
        AUTOLOGGING_INTEGRATIONS[key].clear()
    mlflow.utils.import_hooks._post_import_hooks = {}


@pytest.fixture(params=[mlflow.crewai.autolog, global_autolog])
def autolog(request):
    clear_autolog_state()

    yield request.param

    clear_autolog_state()


def test_kickoff_enable_disable_autolog(simple_agent_1, task_1, autolog):
    crew = Crew(
        agents=[
            simple_agent_1,
        ],
        tasks=[task_1],
    )
    with patch("litellm.completion", return_value=_SIMPLE_CHAT_COMPLETION):
        autolog()
        crew.kickoff()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 5
    # Crew
    span_0 = traces[0].data.spans[0]
    assert span_0.name == "Crew.kickoff"
    assert span_0.span_type == SpanType.CHAIN
    assert span_0.parent_id is None
    assert span_0.inputs == {}
    assert span_0.outputs == _CREW_OUTPUT
    # Task
    span_1 = traces[0].data.spans[1]
    assert span_1.name == "Task.execute_sync"
    assert span_1.span_type == SpanType.CHAIN
    assert span_1.parent_id is span_0.span_id
    assert span_1.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_1.outputs == {
        "agent": "City Selection Expert",
        "description": "Analyze and select the best city for the trip",
        "expected_output": "Detailed report on the chosen city",
        "json_dict": None,
        "name": None,
        "output_format": "raw",
        "pydantic": None,
        "raw": _LLM_ANSWER,
        "summary": "Analyze and select the best city for the trip...",
    }
    # Agent
    span_2 = traces[0].data.spans[2]
    assert span_2.name == "Agent.execute_task"
    assert span_2.span_type == SpanType.AGENT
    assert span_2.parent_id is span_1.span_id
    assert span_2.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_2.outputs == _LLM_ANSWER
    # LLM
    span_3 = traces[0].data.spans[3]
    assert span_3.name == "LLM.call"
    assert span_3.span_type == SpanType.LLM
    assert span_3.parent_id is span_2.span_id
    assert span_3.inputs["messages"] is not None
    assert span_3.outputs == f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}"
    chat_attributes = span_3.get_attribute("mlflow.chat.messages")
    assert len(chat_attributes) == 3
    assert chat_attributes[0]["role"] == "system"
    assert _AGENT_1_GOAL in chat_attributes[0]["content"]
    assert chat_attributes[1]["role"] == "user"
    assert _TASK_1_DESCRIPTION in chat_attributes[1]["content"]
    assert chat_attributes[2]["role"] == "assistant"
    assert _LLM_ANSWER in chat_attributes[2]["content"]

    # Create Long Term Memory
    span_4 = traces[0].data.spans[4]
    assert span_4.name == "CrewAgentExecutor._create_long_term_memory"
    assert span_4.span_type == SpanType.RETRIEVER
    assert span_4.parent_id is span_2.span_id
    assert span_4.inputs == {
        "output": {
            "output": _LLM_ANSWER,
            "text": f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}",
            "thought": "",
        }
    }
    assert span_4.outputs is None

    with patch("litellm.completion", return_value=_SIMPLE_CHAT_COMPLETION):
        mlflow.crewai.autolog(disable=True)
        crew.kickoff()

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


@pytest.mark.skip(
    reason=(
        "https://github.com/crewAIInc/crewAI/issues/1934. Remove skip when the issue is resolved."
    )
)
def test_kickoff_failure(simple_agent_1, task_1, autolog):
    crew = Crew(
        agents=[
            simple_agent_1,
        ],
        tasks=[task_1],
    )
    with patch("litellm.completion", side_effect=Exception("error")):
        autolog()
        with pytest.raises(Exception, match="error"):
            crew.kickoff()
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    assert len(traces[0].data.spans) == 4
    # Crew
    span_0 = traces[0].data.spans[0]
    assert span_0.name == "Crew.kickoff"
    assert span_0.span_type == SpanType.CHAIN
    assert span_0.parent_id is None
    assert span_0.inputs == {}
    assert span_0.status.status_code == "ERROR"
    # Task
    span_1 = traces[0].data.spans[1]
    assert span_1.name == "Task.execute_sync"
    assert span_1.span_type == SpanType.CHAIN
    assert span_1.parent_id is span_0.span_id
    assert span_1.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_1.status.status_code == "ERROR"
    # Agent
    span_2 = traces[0].data.spans[2]
    assert span_2.name == "Agent.execute_task"
    assert span_2.span_type == SpanType.AGENT
    assert span_2.parent_id is span_1.span_id
    assert span_2.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_2.status.status_code == "ERROR"
    # LLM
    span_3 = traces[0].data.spans[3]
    assert span_3.name == "LLM.call"
    assert span_3.span_type == SpanType.LLM
    assert span_3.parent_id is span_2.span_id
    assert span_3.inputs["messages"] is not None
    assert span_3.status.status_code == "ERROR"


def test_kickoff_tool_calling(tool_agent_1, task_1_with_tool, autolog):
    crew = Crew(
        agents=[
            tool_agent_1,
        ],
        tasks=[task_1_with_tool],
    )
    with patch("litellm.completion", side_effect=[_TOOL_CHAT_COMPLETION, _SIMPLE_CHAT_COMPLETION]):
        autolog()
        crew.kickoff()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 6
    # Crew
    span_0 = traces[0].data.spans[0]
    assert span_0.name == "Crew.kickoff"
    assert span_0.span_type == SpanType.CHAIN
    assert span_0.parent_id is None
    assert span_0.inputs == {}
    assert span_0.outputs == _CREW_OUTPUT
    # Task
    span_1 = traces[0].data.spans[1]
    assert span_1.name == "Task.execute_sync"
    assert span_1.span_type == SpanType.CHAIN
    assert span_1.parent_id is span_0.span_id
    assert len(span_1.inputs["tools"]) == 1
    assert span_1.inputs["tools"][0]["name"] == "TestTool"
    assert span_1.outputs == {
        "agent": "City Selection Expert",
        "description": "Analyze and select the best city for the trip",
        "expected_output": "Detailed report on the chosen city",
        "json_dict": None,
        "name": None,
        "output_format": "raw",
        "pydantic": None,
        "raw": _LLM_ANSWER,
        "summary": "Analyze and select the best city for the trip...",
    }
    # Agent
    span_2 = traces[0].data.spans[2]
    assert span_2.name == "Agent.execute_task"
    assert span_2.span_type == SpanType.AGENT
    assert span_2.parent_id is span_1.span_id
    assert len(span_2.inputs["tools"]) == 1
    assert span_2.inputs["tools"][0]["name"] == "TestTool"
    assert span_2.outputs == _LLM_ANSWER
    # LLM - tool calling
    span_3 = traces[0].data.spans[3]
    assert span_3.name == "LLM.call_1"
    assert span_3.span_type == SpanType.LLM
    assert span_3.parent_id is span_2.span_id
    assert span_3.inputs["messages"] is not None
    assert "Action: TestTool" in span_3.outputs
    chat_attributes = span_3.get_attribute("mlflow.chat.messages")
    assert len(chat_attributes) == 3
    assert chat_attributes[0]["role"] == "system"
    assert _AGENT_1_GOAL in chat_attributes[0]["content"]
    assert chat_attributes[1]["role"] == "user"
    assert _TASK_1_DESCRIPTION in chat_attributes[1]["content"]
    assert chat_attributes[2]["role"] == "assistant"
    assert "Action: TestTool" in chat_attributes[2]["content"]
    # LLM - return answer
    span_4 = traces[0].data.spans[4]
    assert span_4.name == "LLM.call_2"
    assert span_4.span_type == SpanType.LLM
    assert span_4.parent_id is span_2.span_id
    assert span_4.inputs["messages"] is not None
    assert span_4.outputs == f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}"
    chat_attributes = span_4.get_attribute("mlflow.chat.messages")
    assert len(chat_attributes) == 4
    assert chat_attributes[0]["role"] == "system"
    assert _AGENT_1_GOAL in chat_attributes[0]["content"]
    assert chat_attributes[1]["role"] == "user"
    assert _TASK_1_DESCRIPTION in chat_attributes[1]["content"]
    assert chat_attributes[2]["role"] == "assistant"
    assert "Tool Answer" in chat_attributes[2]["content"]
    assert chat_attributes[3]["role"] == "assistant"
    assert _LLM_ANSWER in chat_attributes[3]["content"]

    # Create Long Term Memory
    span_5 = traces[0].data.spans[5]
    assert span_5.name == "CrewAgentExecutor._create_long_term_memory"
    assert span_5.span_type == SpanType.RETRIEVER
    assert span_5.parent_id is span_2.span_id
    assert span_5.inputs == {
        "output": {
            "output": _LLM_ANSWER,
            "text": f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}",
            "thought": "",
        }
    }
    assert span_5.outputs is None


def test_multi_tasks(simple_agent_1, simple_agent_2, task_1, task_2, autolog):
    crew = Crew(
        agents=[
            simple_agent_1,
            simple_agent_2,
        ],
        tasks=[task_1, task_2],
    )
    with patch("litellm.completion", return_value=_SIMPLE_CHAT_COMPLETION):
        autolog()
        crew.kickoff()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 9
    # Crew
    span_0 = traces[0].data.spans[0]
    assert span_0.name == "Crew.kickoff"
    assert span_0.span_type == SpanType.CHAIN
    assert span_0.parent_id is None
    assert span_0.inputs == {}
    assert span_0.outputs == {
        "json_dict": None,
        "pydantic": None,
        "raw": _LLM_ANSWER,
        "tasks_output": [
            {
                "agent": "City Selection Expert",
                "name": None,
                "description": "Analyze and select the best city for the trip",
                "expected_output": "Detailed report on the chosen city",
                "json_dict": None,
                "pydantic": None,
                "output_format": "raw",
                "raw": _LLM_ANSWER,
                "summary": "Analyze and select the best city for the trip...",
            },
            {
                "agent": "Local Expert at this city",
                "description": "Compile an in-depth guide",
                "expected_output": "Comprehensive city guide",
                "json_dict": None,
                "name": None,
                "output_format": "raw",
                "pydantic": None,
                "raw": _LLM_ANSWER,
                "summary": "Compile an in-depth guide...",
            },
        ],
        "token_usage": {
            "cached_prompt_tokens": ANY_INT,
            "completion_tokens": ANY_INT,
            "prompt_tokens": ANY_INT,
            "successful_requests": ANY_INT,
            "total_tokens": ANY_INT,
        },
    }
    # Task
    span_1 = traces[0].data.spans[1]
    assert span_1.name == "Task.execute_sync_1"
    assert span_1.span_type == SpanType.CHAIN
    assert span_1.parent_id is span_0.span_id
    assert span_1.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_1.outputs == {
        "agent": "City Selection Expert",
        "description": "Analyze and select the best city for the trip",
        "expected_output": "Detailed report on the chosen city",
        "json_dict": None,
        "name": None,
        "output_format": "raw",
        "pydantic": None,
        "raw": _LLM_ANSWER,
        "summary": "Analyze and select the best city for the trip...",
    }
    # Agent
    span_2 = traces[0].data.spans[2]
    assert span_2.name == "Agent.execute_task_1"
    assert span_2.span_type == SpanType.AGENT
    assert span_2.parent_id is span_1.span_id
    assert span_2.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_2.outputs == _LLM_ANSWER
    # LLM
    span_3 = traces[0].data.spans[3]
    assert span_3.name == "LLM.call_1"
    assert span_3.span_type == SpanType.LLM
    assert span_3.parent_id is span_2.span_id
    assert span_3.inputs["messages"] is not None
    assert span_3.outputs == f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}"
    chat_attributes = span_3.get_attribute("mlflow.chat.messages")
    assert len(chat_attributes) == 3
    assert chat_attributes[0]["role"] == "system"
    assert _AGENT_1_GOAL in chat_attributes[0]["content"]
    assert chat_attributes[1]["role"] == "user"
    assert _TASK_1_DESCRIPTION in chat_attributes[1]["content"]
    assert chat_attributes[2]["role"] == "assistant"
    assert _LLM_ANSWER in chat_attributes[2]["content"]

    # Create Long Term Memory
    span_4 = traces[0].data.spans[4]
    assert span_4.name == "CrewAgentExecutor._create_long_term_memory_1"
    assert span_4.span_type == SpanType.RETRIEVER
    assert span_4.parent_id is span_2.span_id
    assert span_4.inputs == {
        "output": {
            "output": _LLM_ANSWER,
            "text": f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}",
            "thought": "",
        }
    }
    assert span_4.outputs is None

    # Task
    span_5 = traces[0].data.spans[5]
    assert span_5.name == "Task.execute_sync_2"
    assert span_5.span_type == SpanType.CHAIN
    assert span_5.parent_id is span_0.span_id
    assert span_5.inputs == {
        "context": _LLM_ANSWER,
        "tools": [],
    }
    assert span_5.outputs == {
        "agent": "Local Expert at this city",
        "description": "Compile an in-depth guide",
        "expected_output": "Comprehensive city guide",
        "json_dict": None,
        "name": None,
        "output_format": "raw",
        "pydantic": None,
        "raw": _LLM_ANSWER,
        "summary": "Compile an in-depth guide...",
    }
    # Agent
    span_6 = traces[0].data.spans[6]
    assert span_6.name == "Agent.execute_task_2"
    assert span_6.span_type == SpanType.AGENT
    assert span_6.parent_id is span_5.span_id
    assert span_6.inputs == {
        "context": _LLM_ANSWER,
        "tools": [],
    }
    assert span_6.outputs == _LLM_ANSWER
    # LLM
    span_7 = traces[0].data.spans[7]
    assert span_7.name == "LLM.call_2"
    assert span_7.span_type == SpanType.LLM
    assert span_7.parent_id is span_6.span_id
    assert span_7.inputs["messages"] is not None
    assert span_7.outputs == f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}"
    chat_attributes = span_7.get_attribute("mlflow.chat.messages")
    assert len(chat_attributes) == 3
    assert chat_attributes[0]["role"] == "system"
    assert _AGENT_2_GOAL in chat_attributes[0]["content"]
    assert chat_attributes[1]["role"] == "user"
    assert _TASK_2_DESCRIPTION in chat_attributes[1]["content"]
    assert chat_attributes[2]["role"] == "assistant"
    assert _LLM_ANSWER in chat_attributes[2]["content"]
    # Create Long Term Memory
    span_8 = traces[0].data.spans[8]
    assert span_8.name == "CrewAgentExecutor._create_long_term_memory_2"
    assert span_8.span_type == SpanType.RETRIEVER
    assert span_8.parent_id is span_6.span_id
    assert span_8.inputs == {
        "output": {
            "output": _LLM_ANSWER,
            "text": f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}",
            "thought": "",
        }
    }
    assert span_8.outputs is None


@pytest.mark.skipif(
    Version(crewai.__version__) < Version("0.83.0"),
    reason=("Memory feature in the current style is not available before 0.83.0"),
)
def test_memory(simple_agent_1, task_1, monkeypatch, autolog):
    monkeypatch.setenv("OPENAI_API_KEY", "000")
    crew = Crew(
        agents=[
            simple_agent_1,
        ],
        tasks=[task_1],
        memory=True,
    )
    with patch("litellm.completion", return_value=_SIMPLE_CHAT_COMPLETION):
        with patch("openai.OpenAI") as client:
            client().embeddings.create.return_value = _EMBEDDING
            autolog()
            crew.kickoff()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 9
    # Crew
    span_0 = traces[0].data.spans[0]
    assert span_0.name == "Crew.kickoff"
    assert span_0.span_type == SpanType.CHAIN
    assert span_0.parent_id is None
    assert span_0.inputs == {}
    assert span_0.outputs == _CREW_OUTPUT
    # Task
    span_1 = traces[0].data.spans[1]
    assert span_1.name == "Task.execute_sync"
    assert span_1.span_type == SpanType.CHAIN
    assert span_1.parent_id is span_0.span_id
    assert span_1.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_1.outputs == {
        "agent": "City Selection Expert",
        "description": "Analyze and select the best city for the trip",
        "expected_output": "Detailed report on the chosen city",
        "json_dict": None,
        "name": None,
        "output_format": "raw",
        "pydantic": None,
        "raw": _LLM_ANSWER,
        "summary": "Analyze and select the best city for the trip...",
    }
    # Agent
    span_2 = traces[0].data.spans[2]
    assert span_2.name == "Agent.execute_task"
    assert span_2.span_type == SpanType.AGENT
    assert span_2.parent_id is span_1.span_id
    assert span_2.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_2.outputs == _LLM_ANSWER

    # LongTermMemory
    span_3 = traces[0].data.spans[3]
    assert span_3.name == "LongTermMemory.search"
    assert span_3.span_type == SpanType.RETRIEVER
    assert span_3.parent_id is span_2.span_id
    assert span_3.inputs == {
        "latest_n": 2,
        "task": "Analyze and select the best city for the trip",
    }
    assert span_3.outputs is None

    # ShortTermMemory
    span_4 = traces[0].data.spans[4]
    assert span_4.name == "ShortTermMemory.search"
    assert span_4.span_type == SpanType.RETRIEVER
    assert span_4.parent_id is span_2.span_id
    assert span_4.inputs == {"query": "Analyze and select the best city for the trip"}
    assert span_4.outputs == []

    # EntityMemory
    span_5 = traces[0].data.spans[5]
    assert span_5.name == "EntityMemory.search"
    assert span_5.span_type == SpanType.RETRIEVER
    assert span_5.parent_id is span_2.span_id
    assert span_5.inputs == {
        "query": "Analyze and select the best city for the trip",
    }
    assert span_5.outputs == []

    # LLM
    span_6 = traces[0].data.spans[6]
    assert span_6.name == "LLM.call"
    assert span_6.span_type == SpanType.LLM
    assert span_6.parent_id is span_2.span_id
    assert span_6.inputs["messages"] is not None
    assert span_6.outputs == f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}"
    chat_attributes = span_6.get_attribute("mlflow.chat.messages")
    assert len(chat_attributes) == 3
    assert chat_attributes[0]["role"] == "system"
    assert _AGENT_1_GOAL in chat_attributes[0]["content"]
    assert chat_attributes[1]["role"] == "user"
    assert _TASK_1_DESCRIPTION in chat_attributes[1]["content"]
    assert chat_attributes[2]["role"] == "assistant"
    assert _LLM_ANSWER in chat_attributes[2]["content"]

    # ShortTermMemory.save
    span_7 = traces[0].data.spans[7]
    assert span_7.name == "ShortTermMemory.save"
    assert span_7.span_type == SpanType.RETRIEVER
    assert span_7.parent_id is span_2.span_id
    assert span_7.inputs == {
        "agent": "City Selection Expert",
        "metadata": {
            "observation": "Analyze and select the best city for the trip",
        },
        "value": f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}",
    }
    assert span_7.outputs is None

    # Create Long Term Memory
    span_8 = traces[0].data.spans[8]
    assert span_8.name == "CrewAgentExecutor._create_long_term_memory"
    assert span_8.span_type == SpanType.RETRIEVER
    assert span_8.parent_id is span_2.span_id
    assert span_8.inputs == {
        "output": {
            "output": _LLM_ANSWER,
            "text": f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}",
            "thought": "",
        }
    }
    assert span_8.outputs is None


@pytest.mark.skipif(
    Version(crewai.__version__) < Version("0.83.0")
    or Version(crewai.__version__) >= Version("0.85.0"),
    reason=("Knowledge feature in the current style is available only with 0.83.0"),
)
def test_knowledge(simple_agent_1, task_1, monkeypatch, autolog):
    monkeypatch.setenv("OPENAI_API_KEY", "000")
    from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

    content = "Users name is John"
    string_source = StringKnowledgeSource(content=content, metadata={"preference": "personal"})
    crew = Crew(
        agents=[
            simple_agent_1,
        ],
        tasks=[task_1],
        knowledge={"sources": [string_source], "metadata": {"preference": "personal"}},
    )
    with patch("litellm.completion", return_value=_SIMPLE_CHAT_COMPLETION):
        autolog()
        crew.kickoff()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 6
    # Crew
    span_0 = traces[0].data.spans[0]
    assert span_0.name == "Crew.kickoff"
    assert span_0.span_type == SpanType.CHAIN
    assert span_0.parent_id is None
    assert span_0.inputs == {}
    assert span_0.outputs == _CREW_OUTPUT
    # Task
    span_1 = traces[0].data.spans[1]
    assert span_1.name == "Task.execute_sync"
    assert span_1.span_type == SpanType.CHAIN
    assert span_1.parent_id is span_0.span_id
    assert span_1.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_1.outputs == {
        "agent": "City Selection Expert",
        "description": "Analyze and select the best city for the trip",
        "expected_output": "Detailed report on the chosen city",
        "json_dict": None,
        "name": None,
        "output_format": "raw",
        "pydantic": None,
        "raw": _LLM_ANSWER,
        "summary": "Analyze and select the best city for the trip...",
    }
    # Agent
    span_2 = traces[0].data.spans[2]
    assert span_2.name == "Agent.execute_task"
    assert span_2.span_type == SpanType.AGENT
    assert span_2.parent_id is span_1.span_id
    assert span_2.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_2.outputs == _LLM_ANSWER

    # Knowledge
    span_3 = traces[0].data.spans[3]
    assert span_3.name == "Knowledge.query"
    assert span_3.span_type == SpanType.RETRIEVER
    assert span_3.parent_id is span_2.span_id
    assert span_3.inputs["query"] is not None
    assert span_3.outputs is not None

    # LLM
    span_4 = traces[0].data.spans[4]
    assert span_4.name == "LLM.call"
    assert span_4.span_type == SpanType.LLM
    assert span_4.parent_id is span_2.span_id
    assert span_4.inputs["messages"] is not None
    assert span_4.outputs == f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}"
    chat_attributes = span_4.get_attribute("mlflow.chat.messages")
    assert len(chat_attributes) == 3
    assert chat_attributes[0]["role"] == "system"
    assert _AGENT_1_GOAL in chat_attributes[0]["content"]
    assert chat_attributes[1]["role"] == "user"
    assert _TASK_1_DESCRIPTION in chat_attributes[1]["content"]
    assert chat_attributes[2]["role"] == "assistant"
    assert _LLM_ANSWER in chat_attributes[2]["content"]

    # Create Long Term Memory
    span_5 = traces[0].data.spans[5]
    assert span_5.name == "CrewAgentExecutor._create_long_term_memory"
    assert span_5.span_type == SpanType.RETRIEVER
    assert span_5.parent_id is span_2.span_id
    assert span_5.inputs == {
        "output": {
            "output": _LLM_ANSWER,
            "text": f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}",
            "thought": "",
        }
    }
    assert span_5.outputs is None


def test_kickoff_for_each(simple_agent_1, task_1, autolog):
    crew = Crew(
        agents=[
            simple_agent_1,
        ],
        tasks=[task_1],
    )
    with patch("litellm.completion", return_value=_SIMPLE_CHAT_COMPLETION):
        autolog()
        crew.kickoff_for_each([{}])

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 6
    span_0 = traces[0].data.spans[0]
    # kickoff_for_each
    assert span_0.name == "Crew.kickoff_for_each"
    assert span_0.span_type == SpanType.CHAIN
    assert span_0.parent_id is None
    assert span_0.inputs == {"inputs": [{}]}
    assert span_0.outputs == [_CREW_OUTPUT]
    # Crew
    span_1 = traces[0].data.spans[1]
    assert span_1.name == "Crew.kickoff"
    assert span_1.span_type == SpanType.CHAIN
    assert span_1.parent_id == span_0.span_id
    assert span_1.inputs == {
        "inputs": {},
    }
    assert span_1.outputs == _CREW_OUTPUT
    # Task
    span_2 = traces[0].data.spans[2]
    assert span_2.name == "Task.execute_sync"
    assert span_2.span_type == SpanType.CHAIN
    assert span_2.parent_id is span_1.span_id
    assert span_2.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_2.outputs == {
        "agent": "City Selection Expert",
        "description": "Analyze and select the best city for the trip",
        "expected_output": "Detailed report on the chosen city",
        "json_dict": None,
        "name": None,
        "output_format": "raw",
        "pydantic": None,
        "raw": _LLM_ANSWER,
        "summary": "Analyze and select the best city for the trip...",
    }
    # Agent
    span_3 = traces[0].data.spans[3]
    assert span_3.name == "Agent.execute_task"
    assert span_3.span_type == SpanType.AGENT
    assert span_3.parent_id is span_2.span_id
    assert span_3.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_3.outputs == _LLM_ANSWER
    # LLM
    span_4 = traces[0].data.spans[4]
    assert span_4.name == "LLM.call"
    assert span_4.span_type == SpanType.LLM
    assert span_4.parent_id is span_3.span_id
    assert span_4.inputs["messages"] is not None
    assert span_4.outputs == f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}"
    # Create Long Term Memory
    span_5 = traces[0].data.spans[5]
    assert span_5.name == "CrewAgentExecutor._create_long_term_memory"
    assert span_5.span_type == SpanType.RETRIEVER
    assert span_5.parent_id is span_3.span_id
    assert span_5.inputs == {
        "output": {
            "output": _LLM_ANSWER,
            "text": f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}",
            "thought": "",
        }
    }
    assert span_5.outputs is None


def test_flow(simple_agent_1, task_1, autolog):
    crew = Crew(
        agents=[
            simple_agent_1,
        ],
        tasks=[task_1],
    )

    class TestFlow(Flow):
        @start()
        def start(self):
            return crew.kickoff()

    flow = TestFlow()

    with patch("litellm.completion", return_value=_SIMPLE_CHAT_COMPLETION):
        autolog()
        flow.kickoff()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 6
    span_0 = traces[0].data.spans[0]
    # kickoff_for_each
    assert span_0.name == "TestFlow.kickoff"
    assert span_0.span_type == SpanType.CHAIN
    assert span_0.parent_id is None
    assert span_0.inputs == {}
    assert span_0.outputs == _CREW_OUTPUT
    # Crew
    span_1 = traces[0].data.spans[1]
    assert span_1.name == "Crew.kickoff"
    assert span_1.span_type == SpanType.CHAIN
    assert span_1.parent_id == span_0.span_id
    assert span_1.inputs == {}
    assert span_1.outputs == _CREW_OUTPUT
    # Task
    span_2 = traces[0].data.spans[2]
    assert span_2.name == "Task.execute_sync"
    assert span_2.span_type == SpanType.CHAIN
    assert span_2.parent_id is span_1.span_id
    assert span_2.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_2.outputs == {
        "agent": "City Selection Expert",
        "description": "Analyze and select the best city for the trip",
        "expected_output": "Detailed report on the chosen city",
        "json_dict": None,
        "name": None,
        "output_format": "raw",
        "pydantic": None,
        "raw": _LLM_ANSWER,
        "summary": "Analyze and select the best city for the trip...",
    }
    # Agent
    span_3 = traces[0].data.spans[3]
    assert span_3.name == "Agent.execute_task"
    assert span_3.span_type == SpanType.AGENT
    assert span_3.parent_id is span_2.span_id
    assert span_3.inputs == {
        "context": "",
        "tools": [],
    }
    assert span_3.outputs == _LLM_ANSWER
    # LLM
    span_4 = traces[0].data.spans[4]
    assert span_4.name == "LLM.call"
    assert span_4.span_type == SpanType.LLM
    assert span_4.parent_id is span_3.span_id
    assert span_4.inputs["messages"] is not None
    assert span_4.outputs == f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}"
    chat_attributes = span_4.get_attribute("mlflow.chat.messages")
    assert len(chat_attributes) == 3
    assert chat_attributes[0]["role"] == "system"
    assert _AGENT_1_GOAL in chat_attributes[0]["content"]
    assert chat_attributes[1]["role"] == "user"
    assert _TASK_1_DESCRIPTION in chat_attributes[1]["content"]
    assert chat_attributes[2]["role"] == "assistant"
    assert _LLM_ANSWER in chat_attributes[2]["content"]
    # Create Long Term Memory
    span_5 = traces[0].data.spans[5]
    assert span_5.name == "CrewAgentExecutor._create_long_term_memory"
    assert span_5.span_type == SpanType.RETRIEVER
    assert span_5.parent_id is span_3.span_id
    assert span_5.inputs == {
        "output": {
            "output": _LLM_ANSWER,
            "text": f"{_FINAL_ANSWER_KEYWORD} {_LLM_ANSWER}",
            "thought": "",
        }
    }
    assert span_5.outputs is None
