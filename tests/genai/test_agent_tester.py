from unittest import mock

import pydantic
import pytest

from mlflow.genai.agent_tester import (
    _DEFAULT_NUM_TEST_CASES,
    _DEFAULT_TESTING_GUIDANCE,
    AgentTestResult,
    _AgentDescription,
    _describe_agent_from_response,
    _describe_agent_from_traces,
    _generate_test_cases,
    _get_agent_response_text,
    _load_traces,
    _resolve_agent_description,
    _TestCase,
    _TestCaseList,
)
from mlflow.genai.agent_tester import test_agent as run_test_agent

_MODEL = "openai:/gpt-4o"


def _make_agent_desc(description="A helper", capabilities=("assist",), limitations=()):
    return _AgentDescription(
        description=description,
        capabilities=list(capabilities),
        limitations=list(limitations),
    )


def _make_test_case_list(n=2):
    return _TestCaseList(
        test_cases=[
            _TestCase(
                goal=f"Goal {i}",
                persona=f"Persona {i}",
                simulation_guidelines=[f"guideline {i}"],
            )
            for i in range(n)
        ]
    )


def _llm_dispatcher(agent_desc, test_case_list):
    """Return the right object based on output_schema."""

    def dispatch(**kwargs):
        schema = kwargs.get("output_schema")
        if schema is _AgentDescription:
            return agent_desc
        if schema is _TestCaseList:
            return test_case_list
        raise ValueError(f"Unexpected schema: {schema}")

    return dispatch


@pytest.fixture
def mock_llm():
    with mock.patch("mlflow.genai.agent_tester.get_chat_completions_with_structured_output") as m:
        yield m


@pytest.fixture
def mock_no_active_trace():
    with mock.patch("mlflow.get_last_active_trace_id", return_value=None):
        yield


def test_agent_description_str():
    desc = _AgentDescription(
        description="A weather assistant",
        capabilities=["forecast", "alerts"],
        limitations=["no historical data"],
    )
    result = str(desc)
    assert "A weather assistant" in result
    assert "forecast" in result
    assert "no historical data" in result


def test_agent_description_str_structure():
    desc = _AgentDescription(
        description="A coding assistant",
        capabilities=["write code", "debug"],
        limitations=["no internet access"],
    )
    result = str(desc)
    assert result.startswith("Agent description:")
    assert "Capabilities:" in result
    assert "Limitations:" in result
    assert "- write code" in result
    assert "- no internet access" in result


def test_agent_description_validation_requires_fields():
    with pytest.raises(pydantic.ValidationError, match="capabilities"):
        _AgentDescription.model_validate({})


def test_agent_test_result_str():
    issue = mock.MagicMock()
    issue.severity = "high"
    issue.name = "Issue: Confusing responses"
    issue.description = "Agent gives contradictory answers."
    issues_result = mock.MagicMock()
    issues_result.issues = [issue]
    result = AgentTestResult(
        test_cases=[],
        agent_description="A helpful assistant",
        simulation_traces=[],
        issues_result=issues_result,
    )
    text = str(result)
    assert "A helpful assistant" in text
    assert "Issues found: 1" in text
    assert "[high] Issue: Confusing responses" in text
    assert "Agent gives contradictory answers." in text


def test_test_case_requires_all_fields():
    with pytest.raises(pydantic.ValidationError, match="persona"):
        _TestCase.model_validate({"goal": "Do something"})


def test_test_case_list_empty_is_valid():
    result = _TestCaseList.model_validate({"test_cases": []})
    assert result.test_cases == []


def test_get_agent_response_text_returns_string_directly():
    def predict(messages):
        return "I am a helpful assistant."

    result = _get_agent_response_text(predict)
    assert result == "I am a helpful assistant."


def test_get_agent_response_text_dispatches_messages_kwarg():
    received = {}

    def agent(messages):
        received["messages"] = messages
        return "response"

    _get_agent_response_text(agent)
    assert "messages" in received
    assert received["messages"][0]["role"] == "user"


def test_get_agent_response_text_dispatches_input_kwarg():
    received = {}

    def agent(input):
        received["input"] = input
        return "response"

    _get_agent_response_text(agent)
    assert "input" in received


def test_get_agent_response_text_returns_none_on_exception(mock_no_active_trace):
    def predict(messages):
        raise RuntimeError("boom")

    result = _get_agent_response_text(predict)
    assert result is None


def test_get_agent_response_text_falls_back_to_trace():
    def predict(messages):
        return None

    mock_trace = mock.MagicMock()
    with (
        mock.patch("mlflow.get_last_active_trace_id", return_value="trace-123"),
        mock.patch("mlflow.get_trace", return_value=mock_trace) as mock_get_trace,
        mock.patch(
            "mlflow.genai.utils.trace_utils.extract_outputs_from_trace",
            return_value={"content": "trace text"},
        ),
        mock.patch(
            "mlflow.genai.utils.trace_utils.parse_outputs_to_str",
            side_effect=[None, "trace text"],
        ),
    ):
        result = _get_agent_response_text(predict)

    mock_get_trace.assert_called_once_with("trace-123")
    assert result == "trace text"


def test_get_agent_response_text_returns_none_when_no_output(mock_no_active_trace):
    def predict(messages):
        return None

    result = _get_agent_response_text(predict)
    assert result is None


def test_describe_agent_from_response_calls_llm(mock_llm):
    expected = _make_agent_desc()
    mock_llm.return_value = expected

    result = _describe_agent_from_response("I help with questions.", model=_MODEL)

    assert result == expected
    mock_llm.assert_called_once()
    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs["model_uri"] == _MODEL
    assert call_kwargs["output_schema"] is _AgentDescription
    assert any("I help with questions." in m.content for m in call_kwargs["messages"])


def test_describe_agent_from_traces_with_no_sessions(mock_llm):
    expected = _make_agent_desc()
    mock_llm.return_value = expected

    with (
        mock.patch("mlflow.genai.discovery.utils.group_traces_by_session", return_value={}),
        mock.patch(
            "mlflow.genai.utils.trace_utils.extract_available_tools_from_trace",
            return_value=None,
        ),
    ):
        result = _describe_agent_from_traces([], model=_MODEL)

    assert result == expected
    assert "(no traces)" in mock_llm.call_args.kwargs["messages"][-1].content


def test_describe_agent_from_traces_includes_conversation(mock_llm):
    trace = mock.MagicMock()
    expected = _make_agent_desc(description="A coding assistant", capabilities=["write code"])
    mock_llm.return_value = expected

    with (
        mock.patch(
            "mlflow.genai.discovery.utils.group_traces_by_session",
            return_value={"sess-1": [trace]},
        ),
        mock.patch(
            "mlflow.genai.utils.trace_utils.resolve_conversation_from_session",
            return_value=[{"role": "user", "content": "Write a function"}],
        ),
        mock.patch(
            "mlflow.genai.discovery.extraction.extract_execution_paths_for_session",
            return_value="(no routing)",
        ),
        mock.patch(
            "mlflow.genai.utils.trace_utils.extract_available_tools_from_trace",
            return_value=None,
        ),
    ):
        result = _describe_agent_from_traces([trace], model=_MODEL)

    assert result == expected
    assert "Write a function" in mock_llm.call_args.kwargs["messages"][-1].content


def test_describe_agent_from_traces_includes_tools(mock_llm):
    trace = mock.MagicMock()
    tool = mock.MagicMock()
    tool.function.name = "search_web"
    expected = _make_agent_desc(description="A research agent", capabilities=["search_web"])
    mock_llm.return_value = expected

    with (
        mock.patch("mlflow.genai.discovery.utils.group_traces_by_session", return_value={}),
        mock.patch(
            "mlflow.genai.utils.trace_utils.extract_available_tools_from_trace",
            return_value=[tool],
        ),
    ):
        result = _describe_agent_from_traces([trace], model=_MODEL)

    assert result == expected
    assert "search_web" in mock_llm.call_args.kwargs["messages"][-1].content


def test_generate_test_cases_uses_default_count_and_guidance(mock_llm):
    agent_desc = _make_agent_desc(capabilities=["answer questions"])
    mock_llm.return_value = _make_test_case_list(7)

    result = _generate_test_cases(agent_desc, model=_MODEL)

    assert len(result) == 7
    user_content = mock_llm.call_args.kwargs["messages"][-1].content
    system_content = mock_llm.call_args.kwargs["messages"][0].content
    assert f"Generate {_DEFAULT_NUM_TEST_CASES}" in user_content
    assert _DEFAULT_TESTING_GUIDANCE in system_content


def test_generate_test_cases_uses_provided_count_and_guidance(mock_llm):
    agent_desc = _make_agent_desc(capabilities=["answer questions"])
    mock_llm.return_value = _make_test_case_list(3)

    result = _generate_test_cases(
        agent_desc, model=_MODEL, num_test_cases=3, guidance="Focus on edge cases"
    )

    assert len(result) == 3
    assert "Generate 3" in mock_llm.call_args.kwargs["messages"][-1].content
    assert "Focus on edge cases" in mock_llm.call_args.kwargs["messages"][0].content


def test_generate_test_cases_raises_on_invalid_count(mock_llm):
    agent_desc = _make_agent_desc()
    with pytest.raises(ValueError, match="num_test_cases must be >= 1"):
        _generate_test_cases(agent_desc, model=_MODEL, num_test_cases=0)


def test_generate_test_cases_returns_dicts(mock_llm):
    agent_desc = _make_agent_desc()
    mock_llm.return_value = _make_test_case_list(1)

    result = _generate_test_cases(agent_desc, model=_MODEL)

    assert isinstance(result[0], dict)
    assert result[0]["goal"] == "Goal 0"


def test_load_traces_returns_none_when_no_experiment_id():
    result = _load_traces(experiment_id=None)
    assert result is None


def test_load_traces_searches_experiment():
    mock_traces = [mock.MagicMock(), mock.MagicMock()]

    with mock.patch("mlflow.search_traces", return_value=mock_traces) as mock_search:
        result = _load_traces(experiment_id="exp-123")

    assert result == mock_traces
    mock_search.assert_called_once_with(
        locations=["exp-123"],
        max_results=50,
        return_type="list",
    )


def test_resolve_agent_description_uses_self_description(mock_llm):
    agent_desc = _make_agent_desc(capabilities=["assist"])
    mock_llm.return_value = agent_desc

    def predict(messages):
        return "I am a helpful assistant."

    result = _resolve_agent_description(predict, None, None, _MODEL)
    assert result == agent_desc


def test_resolve_agent_description_falls_back_to_traces(mock_llm):
    traces = [mock.MagicMock()]
    agent_desc = _make_agent_desc(capabilities=["assist"])
    mock_llm.return_value = agent_desc

    def predict(messages):
        raise RuntimeError("cannot self-describe")

    with (
        mock.patch("mlflow.genai.discovery.utils.group_traces_by_session", return_value={}),
        mock.patch(
            "mlflow.genai.utils.trace_utils.extract_available_tools_from_trace",
            return_value=None,
        ),
    ):
        result = _resolve_agent_description(predict, None, traces, _MODEL)

    assert result == agent_desc


def test_resolve_agent_description_loads_traces_from_experiment(mock_llm):
    agent_desc = _make_agent_desc(capabilities=["assist"])
    loaded_traces = [mock.MagicMock()]
    mock_llm.return_value = agent_desc

    def predict(messages):
        raise RuntimeError("cannot self-describe")

    with (
        mock.patch("mlflow.search_traces", return_value=loaded_traces) as mock_search,
        mock.patch("mlflow.genai.discovery.utils.group_traces_by_session", return_value={}),
        mock.patch(
            "mlflow.genai.utils.trace_utils.extract_available_tools_from_trace",
            return_value=None,
        ),
    ):
        result = _resolve_agent_description(predict, "exp-456", None, _MODEL)

    assert result == agent_desc
    mock_search.assert_called_once_with(locations=["exp-456"], max_results=50, return_type="list")


def test_resolve_agent_description_ignores_experiment_id_when_traces_provided(mock_llm):
    traces = [mock.MagicMock()]
    agent_desc = _make_agent_desc(capabilities=["assist"])
    mock_llm.return_value = agent_desc

    def predict(messages):
        raise RuntimeError("cannot self-describe")

    with (
        mock.patch("mlflow.search_traces") as mock_search,
        mock.patch("mlflow.genai.discovery.utils.group_traces_by_session", return_value={}),
        mock.patch(
            "mlflow.genai.utils.trace_utils.extract_available_tools_from_trace",
            return_value=None,
        ),
    ):
        result = _resolve_agent_description(predict, "exp-456", traces, _MODEL)

    assert result == agent_desc
    mock_search.assert_not_called()


def test_resolve_agent_description_falls_back_when_llm_raises(mock_llm):
    traces = [mock.MagicMock()]
    agent_desc = _make_agent_desc(capabilities=["assist"])
    # First call (self-description) raises, second call (traces) succeeds
    mock_llm.side_effect = [RuntimeError("LLM error"), agent_desc]

    def predict(messages):
        return "I am a helpful assistant."

    with (
        mock.patch("mlflow.genai.discovery.utils.group_traces_by_session", return_value={}),
        mock.patch(
            "mlflow.genai.utils.trace_utils.extract_available_tools_from_trace",
            return_value=None,
        ),
    ):
        result = _resolve_agent_description(predict, None, traces, _MODEL)

    assert result == agent_desc


def test_resolve_agent_description_returns_default_when_all_fail():
    def predict(messages):
        raise RuntimeError("cannot self-describe")

    with mock.patch("mlflow.search_traces", return_value=None):
        result = _resolve_agent_description(predict, None, None, _MODEL)

    assert result.description == "A conversational AI agent"
    assert result.capabilities == ["general conversation"]


def test_test_agent_uses_default_model(mock_llm):
    agent_desc = _make_agent_desc()
    mock_llm.side_effect = _llm_dispatcher(agent_desc, _make_test_case_list())
    mock_issues = mock.MagicMock()

    def predict(messages):
        return "I am a helpful assistant."

    with (
        mock.patch(
            "mlflow.genai.simulators.utils.get_default_simulation_model",
            return_value=_MODEL,
        ) as mock_get_model,
        mock.patch("mlflow.genai.simulators.ConversationSimulator") as mock_sim_cls,
        mock.patch("mlflow.genai.discovery.pipeline.discover_issues", return_value=mock_issues),
    ):
        mock_sim_cls.return_value.simulate.return_value = [[mock.MagicMock()]]
        result = run_test_agent(predict)

    mock_get_model.assert_called_once()
    assert result.agent_description == str(agent_desc)


def test_test_agent_passes_model_to_simulator_and_discovery(mock_llm):
    agent_desc = _make_agent_desc()
    mock_llm.side_effect = _llm_dispatcher(agent_desc, _make_test_case_list())
    mock_issues = mock.MagicMock()

    def predict(messages):
        return "I am a helpful assistant."

    with (
        mock.patch("mlflow.genai.simulators.ConversationSimulator") as mock_sim_cls,
        mock.patch(
            "mlflow.genai.discovery.pipeline.discover_issues", return_value=mock_issues
        ) as mock_discover,
    ):
        mock_sim_cls.return_value.simulate.return_value = [[]]
        run_test_agent(predict, model=_MODEL)

    assert all(call.kwargs["model_uri"] == _MODEL for call in mock_llm.call_args_list)
    assert mock_sim_cls.call_args.kwargs["user_model"] == _MODEL
    assert mock_discover.call_args.kwargs["model"] == _MODEL


def test_test_agent_returns_correct_result(mock_llm):
    agent_desc = _make_agent_desc()
    test_case_list = _make_test_case_list(2)
    mock_llm.side_effect = _llm_dispatcher(agent_desc, test_case_list)
    mock_sim_traces = [[mock.MagicMock()], [mock.MagicMock()]]
    mock_issues = mock.MagicMock()

    def predict(messages):
        return "I am a helpful assistant."

    with (
        mock.patch("mlflow.genai.simulators.ConversationSimulator") as mock_sim_cls,
        mock.patch("mlflow.genai.discovery.pipeline.discover_issues", return_value=mock_issues),
    ):
        mock_sim_cls.return_value.simulate.return_value = mock_sim_traces
        result = run_test_agent(predict, model=_MODEL)

    assert len(result.test_cases) == 2
    assert result.agent_description == str(agent_desc)
    assert result.simulation_traces == mock_sim_traces
    assert result.issues_result is mock_issues


def test_test_agent_flattens_traces_for_issue_detection(mock_llm):
    agent_desc = _make_agent_desc()
    mock_llm.side_effect = _llm_dispatcher(agent_desc, _make_test_case_list(2))
    mock_issues = mock.MagicMock()
    t1 = mock.MagicMock()
    t2 = mock.MagicMock()
    t3 = mock.MagicMock()

    def predict(messages):
        return "I am a helpful assistant."

    with (
        mock.patch("mlflow.genai.simulators.ConversationSimulator") as mock_sim_cls,
        mock.patch(
            "mlflow.genai.discovery.pipeline.discover_issues", return_value=mock_issues
        ) as mock_discover,
    ):
        mock_sim_cls.return_value.simulate.return_value = [[t1, t2], [t3]]
        run_test_agent(predict, model=_MODEL)

    flat_traces = mock_discover.call_args.kwargs["traces"]
    assert flat_traces == [t1, t2, t3]


def test_test_agent_passes_max_turns_and_max_issues(mock_llm):
    agent_desc = _make_agent_desc()
    mock_llm.side_effect = _llm_dispatcher(agent_desc, _make_test_case_list(1))
    mock_issues = mock.MagicMock()

    def predict(messages):
        return "I am a helpful assistant."

    with (
        mock.patch("mlflow.genai.simulators.ConversationSimulator") as mock_sim_cls,
        mock.patch(
            "mlflow.genai.discovery.pipeline.discover_issues", return_value=mock_issues
        ) as mock_discover,
    ):
        mock_sim_cls.return_value.simulate.return_value = [[]]
        run_test_agent(predict, model=_MODEL, max_turns=5, max_issues=10)

    assert mock_sim_cls.call_args.kwargs["max_turns"] == 5
    assert mock_discover.call_args.kwargs["max_issues"] == 10
