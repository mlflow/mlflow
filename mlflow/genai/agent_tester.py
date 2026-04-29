from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import pydantic

import mlflow
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.genai.discovery.entities import DiscoverIssuesResult

_logger = logging.getLogger(__name__)


class _AgentDescription(pydantic.BaseModel):
    description: str = pydantic.Field(
        description="What the agent does — a concise summary of its purpose"
    )
    capabilities: list[str] = pydantic.Field(
        description="Tools, skills, or knowledge areas the agent has"
    )
    limitations: list[str] = pydantic.Field(
        description="Known constraints, boundaries, or things the agent cannot do"
    )

    def __str__(self) -> str:
        capabilities = "\n".join(f"- {c}" for c in self.capabilities)
        limitations = "\n".join(f"- {lim}" for lim in self.limitations)
        return (
            f"Agent description: {self.description}\n\n"
            f"Capabilities:\n{capabilities}\n\n"
            f"Limitations:\n{limitations}"
        )


class _TestCase(pydantic.BaseModel):
    goal: str = pydantic.Field(description="What the simulated user is trying to accomplish")
    persona: str = pydantic.Field(description="A short description of who the simulated user is")
    simulation_guidelines: list[str] = pydantic.Field(
        description="Instructions for how the simulated user should behave"
    )


class _TestCaseList(pydantic.BaseModel):
    test_cases: list[_TestCase] = pydantic.Field(description="List of test cases to simulate")


@experimental(version="3.13.0")
@dataclass
class AgentTestResult:
    """
    Result of :func:`test_agent`.

    Attributes:
        test_cases: Test cases that were generated and simulated.
        agent_description: Natural-language description of the agent
            produced by Step 1.
        simulation_traces: Per-test-case lists of traces produced by
            the conversation simulator.
        issues_result: Full result from the underlying issue detection call.
    """

    test_cases: list[dict[str, Any]]
    agent_description: str
    simulation_traces: list[list[Trace]]
    issues_result: DiscoverIssuesResult

    def __str__(self) -> str:
        issues = self.issues_result.issues
        lines = [self.agent_description, "", f"Issues found: {len(issues)}"]
        lines += [f"  [{issue.severity}] {issue.name}: {issue.description}" for issue in issues]
        return "\n".join(lines)


_DEFAULT_NUM_TEST_CASES = 7

_DESCRIBE_AGENT_SYSTEM_PROMPT = """\
You are an expert at analysing AI agents. Given the agent's own response to \
"describe yourself", extract a structured description."""

_DESCRIBE_AGENT_FROM_TRACES_SYSTEM_PROMPT = """\
You are an expert at analysing AI agents. Given conversation traces from an \
AI agent, extract a structured description of what the agent does, its \
capabilities, and its limitations."""

_DEFAULT_TESTING_GUIDANCE = (
    "Cover a broad mix of the agent's stated capabilities. All test cases should "
    "be realistic. Some should be challenging: ambiguous requests, multi-step "
    "tasks, or requests near the agent's stated limitations."
)

_GENERATE_TEST_CASES_SYSTEM_PROMPT = """\
You are a QA engineer for AI agents. Given a description of an agent, \
generate diverse test cases that exercise different capabilities.

Each test case needs a goal (what the user wants), a persona (who they are), \
and simulation_guidelines (a short list of behavioral instructions for the \
simulated user).

{guidance}

Example output for a weather assistant:

```json
{{
  "test_cases": [
    {{
      "goal": "Get a 7-day weather forecast for Seattle",
      "persona": "A traveler packing for a trip",
      "simulation_guidelines": ["Ask one follow-up about what to wear"]
    }},
    {{
      "goal": "Compare today's weather in Tokyo and London",
      "persona": "Someone scheduling an international call",
      "simulation_guidelines": ["Keep the conversation to 2-3 turns"]
    }}
  ]
}}
```"""


def _get_agent_response_text(predict_fn: Callable[..., Any]) -> str | None:
    """
    Call *predict_fn* with a self-description prompt and return the
    assistant's response as a plain string, or ``None`` on failure.
    """

    prompt = [
        {
            "role": "user",
            "content": (
                "What can you do? Describe your capabilities, tools, and limitations in detail."
            ),
        }
    ]

    try:
        sig = inspect.signature(predict_fn)
        params = list(sig.parameters.keys())
        result = predict_fn(messages=prompt) if "messages" in params else predict_fn(input=prompt)
    except Exception:
        _logger.debug("predict_fn raised when asked to self-describe", exc_info=True)
        return None

    if isinstance(result, str):
        return result

    # Try to extract text the same way the simulator does
    from mlflow.genai.utils.trace_utils import parse_outputs_to_str

    text = parse_outputs_to_str(result)
    if text and text.strip():
        return text

    # Last resort: check the latest trace
    try:
        if trace_id := mlflow.get_last_active_trace_id():
            from mlflow.genai.utils.trace_utils import extract_outputs_from_trace

            trace = mlflow.get_trace(trace_id)
            if outputs := extract_outputs_from_trace(trace):
                return parse_outputs_to_str(outputs)
    except Exception:
        _logger.debug("Failed to extract text from last active trace", exc_info=True)

    return None


def _describe_agent_from_response(
    response_text: str,
    model: str,
) -> _AgentDescription:
    from mlflow.genai.judges.utils import get_chat_completions_with_structured_output
    from mlflow.types.llm import ChatMessage

    messages = [
        ChatMessage(role="system", content=_DESCRIBE_AGENT_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=f"Agent's self-description:\n\n{response_text}",
        ),
    ]
    return get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_AgentDescription,
    )


def _describe_agent_from_traces(
    traces: list[Trace],
    model: str,
) -> _AgentDescription:
    from mlflow.genai.discovery.extraction import extract_execution_paths_for_session
    from mlflow.genai.discovery.utils import group_traces_by_session
    from mlflow.genai.judges.utils import get_chat_completions_with_structured_output
    from mlflow.genai.utils.trace_utils import (
        extract_available_tools_from_trace,
        resolve_conversation_from_session,
    )
    from mlflow.types.llm import ChatMessage

    sessions = group_traces_by_session(traces)
    context_parts: list[str] = []

    # Sample up to 5 sessions to keep prompt size manageable
    for session_id, session_traces in list(sessions.items())[:5]:
        if conversation := resolve_conversation_from_session(session_traces):
            formatted = "\n".join(f"  {m['role']}: {m['content']}" for m in conversation)
            context_parts.append(f"Conversation ({session_id}):\n{formatted}")

        paths = extract_execution_paths_for_session(session_traces)
        if paths and paths != "(no routing)":
            context_parts.append(f"Execution paths: {paths}")

    # Extract tools from the first trace that has them
    tools_desc = ""
    for trace in traces[:10]:
        if tools := extract_available_tools_from_trace(trace, model=model):
            tool_names = [t.function.name for t in tools if t.function]
            tools_desc = f"Available tools: {', '.join(tool_names)}"
            break

    if tools_desc:
        context_parts.append(tools_desc)

    messages = [
        ChatMessage(
            role="system",
            content=_DESCRIBE_AGENT_FROM_TRACES_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content="\n\n".join(context_parts) if context_parts else "(no traces)",
        ),
    ]
    return get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_AgentDescription,
    )


def _generate_test_cases(
    agent_desc: _AgentDescription,
    model: str,
    num_test_cases: int | None = None,
    guidance: str | None = None,
) -> list[dict[str, Any]]:
    from mlflow.genai.judges.utils import get_chat_completions_with_structured_output
    from mlflow.types.llm import ChatMessage

    if num_test_cases is not None and num_test_cases < 1:
        raise ValueError(f"num_test_cases must be >= 1, got {num_test_cases}")
    guidance = guidance or _DEFAULT_TESTING_GUIDANCE
    count = _DEFAULT_NUM_TEST_CASES if num_test_cases is None else num_test_cases

    system_prompt = _GENERATE_TEST_CASES_SYSTEM_PROMPT.format(
        guidance=guidance,
    )
    user_content = str(agent_desc) + f"\n\nGenerate {count} diverse test cases."
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_content),
    ]
    result = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_TestCaseList,
    )

    return [tc.model_dump() for tc in result.test_cases]


def _resolve_agent_description(
    predict_fn: Callable[..., Any],
    experiment_id: str | None,
    traces: list[Trace] | None,
    model: str,
) -> _AgentDescription:
    agent_desc: _AgentDescription | None = None

    if response_text := _get_agent_response_text(predict_fn):
        try:
            agent_desc = _describe_agent_from_response(response_text, model)
        except Exception:
            _logger.warning("Failed to describe agent from self-description", exc_info=True)

    if (not agent_desc or not agent_desc.capabilities) and (
        existing_traces := (traces or _load_traces(experiment_id))
    ):
        try:
            agent_desc = _describe_agent_from_traces(existing_traces, model)
        except Exception:
            _logger.warning("Failed to describe agent from traces", exc_info=True)

    return agent_desc or _AgentDescription(
        description="A conversational AI agent",
        capabilities=["general conversation"],
        limitations=["unknown"],
    )


def _load_traces(
    experiment_id: str | None,
) -> list[Trace] | None:
    if experiment_id is None:
        return None

    return mlflow.search_traces(
        locations=[experiment_id],
        max_results=50,
        return_type="list",
    )


@experimental(version="3.13.0")
def test_agent(
    predict_fn: Callable[..., Any],
    *,
    experiment_id: str | None = None,
    traces: list[Trace] | None = None,
    model: str | None = None,
    max_turns: int = 10,
    max_issues: int = 20,
    num_test_cases: int | None = None,
    guidance: str | None = None,
) -> AgentTestResult:
    """
    Automatically stress-test a conversational AI agent and discover issues.

    Runs a multi-step pipeline:

    1. **Describe** — asks the agent to describe itself (falls back to
       analysing existing traces when available).
    2. **Generate test cases** — uses an LLM to create diverse,
       targeted test scenarios from the agent description.
    3. **Simulate conversations** — runs each test case through the
       :class:`~mlflow.genai.simulators.ConversationSimulator`.
    4. **Detect issues** — analyses simulation traces with
       :func:`~mlflow.genai.discovery.pipeline.discover_issues`.

    Args:
        predict_fn: Agent function compatible with
            :class:`~mlflow.genai.simulators.ConversationSimulator`.
            Must accept either ``input`` or ``messages`` for conversation
            history.
        experiment_id: Optional experiment containing existing traces to
            help describe the agent. Ignored when ``traces`` is provided.
        traces: Optional list of existing traces to help describe the
            agent.
        model: LLM used for analysis, test generation, and simulation.
            Defaults to :func:`~mlflow.genai.simulators.utils.get_default_simulation_model`.
        max_turns: Maximum conversation turns per test case.
        max_issues: Maximum number of issues to report.
        num_test_cases: Number of test cases to generate. Defaults to
            ``7`` when ``None``.
        guidance: Optional natural-language guidance for what kinds of
            queries to test. For example,
            ``"Focus on multi-step financial workflows"``.
            When ``None``, uses a default that covers a broad,
            realistic mix of the agent's capabilities.

    Returns:
        An :class:`AgentTestResult` containing discovered issues, generated
        test cases, the agent description, simulation traces, and the
        full :class:`~mlflow.genai.discovery.entities.DiscoverIssuesResult`.

    Example:

    .. code-block:: python

        from openai import OpenAI
        import mlflow

        client = OpenAI()


        @mlflow.trace
        def agent(input: list[dict], **kwargs) -> dict:
            mlflow.update_current_trace(session_id=kwargs.get("mlflow_session_id"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful assistant."}] + input,
            )
            content = response.choices[0].message.content
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}


        result = mlflow.genai.test_agent(agent, model="openai:/gpt-4o-mini")
        print(result)
    """
    from mlflow.genai.discovery.pipeline import discover_issues
    from mlflow.genai.simulators import ConversationSimulator
    from mlflow.genai.simulators.utils import get_default_simulation_model

    if not model:
        model = get_default_simulation_model()
        _logger.info(f"Using default model: {model}")

    # ------------------------------------------------------------------
    # Step 1: Describe the agent
    # ------------------------------------------------------------------
    _logger.info("Step 1/4: Describing the agent")
    agent_desc = _resolve_agent_description(predict_fn, experiment_id, traces, model)
    _logger.info(str(agent_desc))

    # ------------------------------------------------------------------
    # Step 2: Generate test cases
    # ------------------------------------------------------------------
    _logger.info("Step 2/4: Generating test cases")
    test_cases = _generate_test_cases(agent_desc, model, num_test_cases, guidance)
    _logger.info(f"Generated {len(test_cases)} test cases")

    # ------------------------------------------------------------------
    # Step 3: Simulate conversations
    # ------------------------------------------------------------------
    _logger.info("Step 3/4: Simulating conversations")
    simulator = ConversationSimulator(
        test_cases=test_cases,
        max_turns=max_turns,
        user_model=model,
    )
    simulation_traces = simulator.simulate(predict_fn)

    # ------------------------------------------------------------------
    # Step 4: Detect issues
    # ------------------------------------------------------------------
    _logger.info("Step 4/4: Detecting issues")
    flat_traces = [t for session in simulation_traces for t in session]
    issues_result = discover_issues(
        traces=flat_traces,
        model=model,
        max_issues=max_issues,
    )

    return AgentTestResult(
        test_cases=test_cases,
        agent_description=str(agent_desc),
        simulation_traces=simulation_traces,
        issues_result=issues_result,
    )
