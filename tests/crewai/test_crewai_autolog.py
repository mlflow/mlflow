from unittest.mock import patch

import crewai
import pytest
from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, start
from packaging.version import Version

import mlflow
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces

# This is a special word for CrewAI to complete the agent execution: https://github.com/crewAIInc/crewAI/blob/c6a6c918e0eba167be1fb82831c73dd664c641e3/src/crewai/agents/parser.py#L7
FINAL_ANSWER_ACTION = "Final Answer:"

SIMPLE_CHAT_COMPLETION = {
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
                "content": f"{FINAL_ANSWER_ACTION} What about Tokyo?",
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

EMBEDDING = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0, 0, 0], "index": 0}],
    "model": "text-embedding-ada-002",
    "usage": {"prompt_tokens": 8, "total_tokens": 8},
}

CREW_OUTPUT = {
    "json_dict": None,
    "pydantic": None,
    "raw": "What about Tokyo?",
    "tasks_output": [
        {
            "agent": "City Selection Expert",
            "name": None,
            "description": "Analyze and select the best city for the trip",
            "expected_output": "Detailed report on the chosen city",
            "json_dict": None,
            "pydantic": None,
            "output_format": "raw",
            "raw": "What about Tokyo?",
            "summary": "Analyze and select the best city for the trip...",
        }
    ],
    "token_usage": {
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "successful_requests": 0,
        "total_tokens": 0,
    },
}


@pytest.fixture
def simple_agent_1():
    return Agent(
        role="City Selection Expert",
        goal="Select the best city based on weather, season, and prices",
        backstory="An expert in analyzing travel data to pick ideal destinations",
        tools=[],
    )


@pytest.fixture
def simple_agent_2():
    return Agent(
        role="Local Expert at this city",
        goal="Provide the BEST insights about the selected city",
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[],
    )


@pytest.fixture
def task_1(simple_agent_1):
    return Task(
        description=("Analyze and select the best city for the trip"),
        agent=simple_agent_1,
        expected_output="Detailed report on the chosen city",
    )


@pytest.fixture
def task_2(simple_agent_2):
    return Task(
        description=("Compile an in-depth guide"),
        agent=simple_agent_2,
        expected_output="Comprehensive city guide",
    )


def test_kickoff_enable_disable_autolog(simple_agent_1, task_1):
    with patch("litellm.completion", return_value=SIMPLE_CHAT_COMPLETION):
        mlflow.crewai.autolog()
        crew = Crew(
            agents=[
                simple_agent_1,
            ],
            tasks=[task_1],
        )

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
        assert span_0.outputs == CREW_OUTPUT
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
            "raw": "What about Tokyo?",
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
        assert span_2.outputs == "What about Tokyo?"
        # LLM
        span_3 = traces[0].data.spans[3]
        assert span_3.name == "LLM.call"
        assert span_3.span_type == SpanType.LLM
        assert span_3.parent_id is span_2.span_id
        assert span_3.inputs["messages"] is not None
        assert span_3.outputs == f"{FINAL_ANSWER_ACTION} What about Tokyo?"

        # Create Long Term Memory
        span_4 = traces[0].data.spans[4]
        assert span_4.name == "CrewAgentExecutor._create_long_term_memory"
        assert span_4.span_type == SpanType.RETRIEVER
        assert span_4.parent_id is span_2.span_id
        assert span_4.inputs == {
            "output": {
                "output": "What about Tokyo?",
                "text": "Final Answer: What about Tokyo?",
                "thought": "",
            }
        }
        assert span_4.outputs is None

        mlflow.crewai.autolog(disable=True)
        crew.kickoff()

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_kickoff_failure(simple_agent_1, task_1):
    with patch("litellm.completion", side_effect=Exception("error")):
        mlflow.crewai.autolog()
        crew = Crew(
            agents=[
                simple_agent_1,
            ],
            tasks=[task_1],
        )
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


def test_multi_tasks(simple_agent_1, simple_agent_2, task_1, task_2):
    with patch("litellm.completion", return_value=SIMPLE_CHAT_COMPLETION):
        mlflow.crewai.autolog()
        crew = Crew(
            agents=[
                simple_agent_1,
                simple_agent_2,
            ],
            tasks=[task_1, task_2],
        )

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
            "raw": "What about Tokyo?",
            "tasks_output": [
                {
                    "agent": "City Selection Expert",
                    "name": None,
                    "description": "Analyze and select the best city for the trip",
                    "expected_output": "Detailed report on the chosen city",
                    "json_dict": None,
                    "pydantic": None,
                    "output_format": "raw",
                    "raw": "What about Tokyo?",
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
                    "raw": "What about Tokyo?",
                    "summary": "Compile an in-depth guide...",
                },
            ],
            "token_usage": {
                "cached_prompt_tokens": 0,
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "successful_requests": 0,
                "total_tokens": 0,
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
            "raw": "What about Tokyo?",
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
        assert span_2.outputs == "What about Tokyo?"
        # LLM
        span_3 = traces[0].data.spans[3]
        assert span_3.name == "LLM.call_1"
        assert span_3.span_type == SpanType.LLM
        assert span_3.parent_id is span_2.span_id
        assert span_3.inputs["messages"] is not None
        assert span_3.outputs == f"{FINAL_ANSWER_ACTION} What about Tokyo?"

        # Create Long Term Memory
        span_4 = traces[0].data.spans[4]
        assert span_4.name == "CrewAgentExecutor._create_long_term_memory_1"
        assert span_4.span_type == SpanType.RETRIEVER
        assert span_4.parent_id is span_2.span_id
        assert span_4.inputs == {
            "output": {
                "output": "What about Tokyo?",
                "text": "Final Answer: What about Tokyo?",
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
            "context": "What about Tokyo?",
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
            "raw": "What about Tokyo?",
            "summary": "Compile an in-depth guide...",
        }
        # Agent
        span_6 = traces[0].data.spans[6]
        assert span_6.name == "Agent.execute_task_2"
        assert span_6.span_type == SpanType.AGENT
        assert span_6.parent_id is span_5.span_id
        assert span_6.inputs == {
            "context": "What about Tokyo?",
            "tools": [],
        }
        assert span_6.outputs == "What about Tokyo?"
        # LLM
        span_7 = traces[0].data.spans[7]
        assert span_7.name == "LLM.call_2"
        assert span_7.span_type == SpanType.LLM
        assert span_7.parent_id is span_6.span_id
        assert span_7.inputs["messages"] is not None
        assert span_7.outputs == f"{FINAL_ANSWER_ACTION} What about Tokyo?"
        # Create Long Term Memory
        span_8 = traces[0].data.spans[8]
        assert span_8.name == "CrewAgentExecutor._create_long_term_memory_2"
        assert span_8.span_type == SpanType.RETRIEVER
        assert span_8.parent_id is span_6.span_id
        assert span_8.inputs == {
            "output": {
                "output": "What about Tokyo?",
                "text": "Final Answer: What about Tokyo?",
                "thought": "",
            }
        }
        assert span_8.outputs is None


@pytest.mark.skipif(
    Version(crewai.__version__) < Version("0.83.0"),
    reason=("Memory feature in the current style is not available before 0.83.0"),
)
def test_memory(simple_agent_1, task_1, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "000")
    with patch("litellm.completion", return_value=SIMPLE_CHAT_COMPLETION):
        with patch("openai.OpenAI") as client:
            client().embeddings.create.return_value = EMBEDDING
            mlflow.crewai.autolog()
            crew = Crew(
                agents=[
                    simple_agent_1,
                ],
                tasks=[task_1],
                memory=True,
            )

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
            assert span_0.outputs == CREW_OUTPUT
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
                "raw": "What about Tokyo?",
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
            assert span_2.outputs == "What about Tokyo?"

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
            assert span_6.outputs == f"{FINAL_ANSWER_ACTION} What about Tokyo?"

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
                "value": "Final Answer: What about Tokyo?",
            }
            assert span_7.outputs is None

            # Create Long Term Memory
            span_8 = traces[0].data.spans[8]
            assert span_8.name == "CrewAgentExecutor._create_long_term_memory"
            assert span_8.span_type == SpanType.RETRIEVER
            assert span_8.parent_id is span_2.span_id
            assert span_8.inputs == {
                "output": {
                    "output": "What about Tokyo?",
                    "text": "Final Answer: What about Tokyo?",
                    "thought": "",
                }
            }
            assert span_8.outputs is None


@pytest.mark.skipif(
    Version(crewai.__version__) < Version("0.83.0"),
    reason=("Knowledge feature in the current style is not available before 0.83.0"),
)
def test_knowledge(simple_agent_1, task_1, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "000")
    from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

    content = "Users name is John"
    string_source = StringKnowledgeSource(content=content, metadata={"preference": "personal"})
    with patch("litellm.completion", return_value=SIMPLE_CHAT_COMPLETION):
        mlflow.crewai.autolog()
        crew = Crew(
            agents=[
                simple_agent_1,
            ],
            tasks=[task_1],
            knowledge={"sources": [string_source], "metadata": {"preference": "personal"}},
        )

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
        assert span_0.outputs == CREW_OUTPUT
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
            "raw": "What about Tokyo?",
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
        assert span_2.outputs == "What about Tokyo?"

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
        assert span_4.outputs == f"{FINAL_ANSWER_ACTION} What about Tokyo?"

        # Create Long Term Memory
        span_5 = traces[0].data.spans[5]
        assert span_5.name == "CrewAgentExecutor._create_long_term_memory"
        assert span_5.span_type == SpanType.RETRIEVER
        assert span_5.parent_id is span_2.span_id
        assert span_5.inputs == {
            "output": {
                "output": "What about Tokyo?",
                "text": "Final Answer: What about Tokyo?",
                "thought": "",
            }
        }
        assert span_5.outputs is None


def test_kickoff_for_each(simple_agent_1, task_1):
    with patch("litellm.completion", return_value=SIMPLE_CHAT_COMPLETION):
        mlflow.crewai.autolog()
        crew = Crew(
            agents=[
                simple_agent_1,
            ],
            tasks=[task_1],
        )

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
        assert span_0.outputs == [CREW_OUTPUT]
        # Crew
        span_1 = traces[0].data.spans[1]
        assert span_1.name == "Crew.kickoff"
        assert span_1.span_type == SpanType.CHAIN
        assert span_1.parent_id == span_0.span_id
        assert span_1.inputs == {
            "inputs": {},
        }
        assert span_1.outputs == CREW_OUTPUT
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
            "raw": "What about Tokyo?",
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
        assert span_3.outputs == "What about Tokyo?"
        # LLM
        span_4 = traces[0].data.spans[4]
        assert span_4.name == "LLM.call"
        assert span_4.span_type == SpanType.LLM
        assert span_4.parent_id is span_3.span_id
        assert span_4.inputs["messages"] is not None
        assert span_4.outputs == f"{FINAL_ANSWER_ACTION} What about Tokyo?"
        # Create Long Term Memory
        span_5 = traces[0].data.spans[5]
        assert span_5.name == "CrewAgentExecutor._create_long_term_memory"
        assert span_5.span_type == SpanType.RETRIEVER
        assert span_5.parent_id is span_3.span_id
        assert span_5.inputs == {
            "output": {
                "output": "What about Tokyo?",
                "text": "Final Answer: What about Tokyo?",
                "thought": "",
            }
        }
        assert span_5.outputs is None


def test_flow(simple_agent_1, task_1):
    with patch("litellm.completion", return_value=SIMPLE_CHAT_COMPLETION):
        mlflow.crewai.autolog()
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
        assert span_0.outputs == CREW_OUTPUT
        # Crew
        span_1 = traces[0].data.spans[1]
        assert span_1.name == "Crew.kickoff"
        assert span_1.span_type == SpanType.CHAIN
        assert span_1.parent_id == span_0.span_id
        assert span_1.inputs == {}
        assert span_1.outputs == CREW_OUTPUT
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
            "raw": "What about Tokyo?",
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
        assert span_3.outputs == "What about Tokyo?"
        # LLM
        span_4 = traces[0].data.spans[4]
        assert span_4.name == "LLM.call"
        assert span_4.span_type == SpanType.LLM
        assert span_4.parent_id is span_3.span_id
        assert span_4.inputs["messages"] is not None
        assert span_4.outputs == f"{FINAL_ANSWER_ACTION} What about Tokyo?"
        # Create Long Term Memory
        span_5 = traces[0].data.spans[5]
        assert span_5.name == "CrewAgentExecutor._create_long_term_memory"
        assert span_5.span_type == SpanType.RETRIEVER
        assert span_5.parent_id is span_3.span_id
        assert span_5.inputs == {
            "output": {
                "output": "What about Tokyo?",
                "text": "Final Answer: What about Tokyo?",
                "thought": "",
            }
        }
        assert span_5.outputs is None
