from unittest.mock import Mock

import pytest

from mlflow.entities.evaluation_dataset import DatasetGranularity, EvaluationDataset
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from mlflow.telemetry.events import (
    AiCommandRunEvent,
    AlignJudgeEvent,
    CreateDatasetEvent,
    CreateExperimentEvent,
    CreateLoggedModelEvent,
    CreateModelVersionEvent,
    CreatePromptEvent,
    CreateRegisteredModelEvent,
    CreateRunEvent,
    DatasetToDataFrameEvent,
    EvaluateEvent,
    GatewayCreateEndpointEvent,
    GatewayCreateSecretEvent,
    GatewayListEndpointsEvent,
    GatewayListSecretsEvent,
    GatewayUpdateEndpointEvent,
    LogAssessmentEvent,
    MakeJudgeEvent,
    MergeRecordsEvent,
    OptimizePromptsJobEvent,
    PromptOptimizationEvent,
    SimulateConversationEvent,
    StartTraceEvent,
)


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {"flavor": "mlflow.pyfunc"},
            {"flavor": "pyfunc"},
        ),
        (
            {"flavor": "sklearn"},
            {"flavor": "sklearn"},
        ),
        (
            {
                "flavor": None,
            },
            None,
        ),
        ({}, None),
    ],
)
def test_logged_model_parse_params(arguments, expected_params):
    assert CreateLoggedModelEvent.name == "create_logged_model"
    assert CreateLoggedModelEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"tags": None}, {"is_prompt": False}),
        ({"tags": {}}, {"is_prompt": False}),
        ({"tags": {IS_PROMPT_TAG_KEY: "true"}}, {"is_prompt": True}),
        ({"tags": {IS_PROMPT_TAG_KEY: "false"}}, {"is_prompt": False}),
        ({}, {"is_prompt": False}),
    ],
)
def test_registered_model_parse_params(arguments, expected_params):
    assert CreateRegisteredModelEvent.name == "create_registered_model"
    assert CreateRegisteredModelEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"tags": None}, {"is_prompt": False}),
        ({"tags": {}}, {"is_prompt": False}),
        ({"tags": {IS_PROMPT_TAG_KEY: "true"}}, {"is_prompt": True}),
        ({"tags": {IS_PROMPT_TAG_KEY: "false"}}, {"is_prompt": False}),
        ({}, {"is_prompt": False}),
    ],
)
def test_create_model_version_parse_params(arguments, expected_params):
    assert CreateModelVersionEvent.name == "create_model_version"
    assert CreateModelVersionEvent.parse(arguments) == expected_params


def test_event_name():
    assert AiCommandRunEvent.name == "ai_command_run"
    assert CreatePromptEvent.name == "create_prompt"
    assert CreateLoggedModelEvent.name == "create_logged_model"
    assert CreateRegisteredModelEvent.name == "create_registered_model"
    assert CreateModelVersionEvent.name == "create_model_version"
    assert CreateRunEvent.name == "create_run"
    assert CreateExperimentEvent.name == "create_experiment"
    assert LogAssessmentEvent.name == "log_assessment"
    assert StartTraceEvent.name == "start_trace"
    assert EvaluateEvent.name == "evaluate"
    assert CreateDatasetEvent.name == "create_dataset"
    assert MergeRecordsEvent.name == "merge_records"
    assert MakeJudgeEvent.name == "make_judge"
    assert AlignJudgeEvent.name == "align_judge"
    assert PromptOptimizationEvent.name == "prompt_optimization"
    assert SimulateConversationEvent.name == "simulate_conversation"


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        # Records without 'inputs' field -> unknown dataset type
        (
            {"records": [{"test": "data"}]},
            {"record_count": 1, "input_type": "list[dict]", "dataset_type": "unknown"},
        ),
        (
            {"records": [{"a": 1}, {"b": 2}]},
            {"record_count": 2, "input_type": "list[dict]", "dataset_type": "unknown"},
        ),
        # Trace records
        (
            {"records": [{"inputs": {"question": "What is MLflow?", "context": "docs"}}]},
            {"record_count": 1, "input_type": "list[dict]", "dataset_type": "trace"},
        ),
        (
            {"records": [{"inputs": {"q": "a"}}, {"inputs": {"q": "b"}}]},
            {"record_count": 2, "input_type": "list[dict]", "dataset_type": "trace"},
        ),
        # Session records
        (
            {"records": [{"inputs": {"persona": "user", "goal": "test", "context": "info"}}]},
            {"record_count": 1, "input_type": "list[dict]", "dataset_type": "session"},
        ),
        # Edge cases
        ({"records": []}, None),
        ({"records": None}, None),
        ({}, None),
        (None, None),
        ({"records": object()}, None),
    ],
)
def test_merge_records_parse_params(arguments, expected_params):
    assert MergeRecordsEvent.parse(arguments) == expected_params


def _make_mock_dataset(granularity: DatasetGranularity) -> Mock:
    mock = Mock(spec=EvaluationDataset)
    mock._get_existing_granularity.return_value = granularity
    return mock


@pytest.mark.parametrize(
    ("granularity", "expected_dataset_type"),
    [
        (DatasetGranularity.TRACE, "trace"),
        (DatasetGranularity.SESSION, "session"),
        (DatasetGranularity.UNKNOWN, "unknown"),
    ],
)
def test_dataset_to_df_parse(granularity, expected_dataset_type):
    mock_dataset = _make_mock_dataset(granularity)
    arguments = {"self": mock_dataset}
    result = DatasetToDataFrameEvent.parse(arguments)
    assert result == {"dataset_type": expected_dataset_type, "callsite": "direct_call"}


@pytest.mark.parametrize(
    ("result", "expected_params"),
    [
        ([{"a": 1}, {"b": 2}, {"c": 3}], {"record_count": 3}),
        ([{"row": 1}], {"record_count": 1}),
        ([], {"record_count": 0}),
        (None, {"record_count": 0}),
    ],
)
def test_dataset_to_df_parse_result(result, expected_params):
    assert DatasetToDataFrameEvent.parse_result(result) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"model": "openai:/gpt-4"}, {"model_provider": "openai"}),
        ({"model": "databricks:/dbrx"}, {"model_provider": "databricks"}),
        ({"model": "custom"}, {"model_provider": None}),
        ({"model": None}, {"model_provider": None}),
        ({}, {"model_provider": None}),
    ],
)
def test_make_judge_parse_params(arguments, expected_params):
    assert MakeJudgeEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {"traces": [{}, {}], "optimizer": None},
            {"trace_count": 2, "optimizer_type": "default"},
        ),
        (
            {"traces": [{}], "optimizer": type("MockOptimizer", (), {})()},
            {"trace_count": 1, "optimizer_type": "MockOptimizer"},
        ),
        (
            {"traces": [], "optimizer": None},
            {"trace_count": 0, "optimizer_type": "default"},
        ),
        ({"traces": None, "optimizer": None}, {"optimizer_type": "default"}),
        ({}, {"optimizer_type": "default"}),
    ],
)
def test_align_judge_parse_params(arguments, expected_params):
    assert AlignJudgeEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        # Normal case with optimizer and prompt URIs
        (
            {
                "optimizer": type("MockOptimizer", (), {})(),
                "prompt_uris": ["prompts:/test/1"],
                "scorers": None,
                "aggregation": None,
            },
            {
                "optimizer_type": "MockOptimizer",
                "prompt_count": 1,
                "scorer_count": None,
                "custom_aggregation": False,
            },
        ),
        # Multiple prompt URIs with custom scorers
        (
            {
                "optimizer": type("CustomAdapter", (), {})(),
                "prompt_uris": ["prompts:/test/1", "prompts:/test/2"],
                "scorers": [Mock()],
                "aggregation": None,
            },
            {
                "optimizer_type": "CustomAdapter",
                "prompt_count": 2,
                "scorer_count": 1,
                "custom_aggregation": False,
            },
        ),
        # Custom objective with multiple scorers
        (
            {
                "optimizer": type("TestAdapter", (), {})(),
                "prompt_uris": ["prompts:/test/1"],
                "scorers": [Mock(), Mock(), Mock()],
                "aggregation": lambda scores: sum(scores.values()),
            },
            {
                "optimizer_type": "TestAdapter",
                "prompt_count": 1,
                "scorer_count": 3,
                "custom_aggregation": True,
            },
        ),
        # No optimizer provided - optimizer_type should be None
        (
            {
                "optimizer": None,
                "prompt_uris": ["prompts:/test/1"],
                "scorers": None,
                "aggregation": None,
            },
            {
                "optimizer_type": None,
                "prompt_count": 1,
                "scorer_count": None,
                "custom_aggregation": False,
            },
        ),
    ],
)
def test_prompt_optimization_parse_params(arguments, expected_params):
    assert PromptOptimizationEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("result", "expected_params"),
    [
        (
            [["t1", "t2", "t3"], ["t1"]],
            {"simulated_conversation_info": [{"turn_count": 3}, {"turn_count": 1}]},
        ),
        ([[]], {"simulated_conversation_info": [{"turn_count": 0}]}),
        ([], {"simulated_conversation_info": []}),
    ],
)
def test_simulate_conversation_parse_result(result, expected_params):
    assert SimulateConversationEvent.parse_result(result) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {
                "fallback_config": {"strategy": "FAILOVER"},
                "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
                "model_configs": [{"model_definition_id": "md-1"}, {"model_definition_id": "md-2"}],
            },
            {
                "has_fallback_config": True,
                "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
                "num_model_configs": 2,
            },
        ),
        (
            {
                "fallback_config": None,
                "routing_strategy": None,
                "model_configs": [{"model_definition_id": "md-1"}],
            },
            {
                "has_fallback_config": False,
                "routing_strategy": None,
                "num_model_configs": 1,
            },
        ),
        (
            {"fallback_config": None, "routing_strategy": None, "model_configs": []},
            {"has_fallback_config": False, "routing_strategy": None, "num_model_configs": 0},
        ),
        (
            {},
            {"has_fallback_config": False, "routing_strategy": None, "num_model_configs": 0},
        ),
    ],
)
def test_gateway_create_endpoint_parse_params(arguments, expected_params):
    assert GatewayCreateEndpointEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {
                "fallback_config": {"strategy": "FAILOVER"},
                "routing_strategy": "ROUND_ROBIN",
                "model_configs": [{"model_definition_id": "md-1"}],
            },
            {
                "has_fallback_config": True,
                "routing_strategy": "ROUND_ROBIN",
                "num_model_configs": 1,
            },
        ),
        (
            {"fallback_config": None, "routing_strategy": None, "model_configs": None},
            {"has_fallback_config": False, "routing_strategy": None, "num_model_configs": None},
        ),
        (
            {},
            {"has_fallback_config": False, "routing_strategy": None, "num_model_configs": None},
        ),
    ],
)
def test_gateway_update_endpoint_parse_params(arguments, expected_params):
    assert GatewayUpdateEndpointEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"provider": "openai"}, {"filter_by_provider": True}),
        ({"provider": "anthropic"}, {"filter_by_provider": True}),
        ({"provider": None}, {"filter_by_provider": False}),
        ({}, {"filter_by_provider": False}),
    ],
)
def test_gateway_list_endpoints_parse_params(arguments, expected_params):
    assert GatewayListEndpointsEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"provider": "openai"}, {"provider": "openai"}),
        ({"provider": "anthropic"}, {"provider": "anthropic"}),
        ({"provider": None}, {"provider": None}),
        ({}, {"provider": None}),
    ],
)
def test_gateway_create_secret_parse_params(arguments, expected_params):
    assert GatewayCreateSecretEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"provider": "openai"}, {"filter_by_provider": True}),
        ({"provider": "anthropic"}, {"filter_by_provider": True}),
        ({"provider": None}, {"filter_by_provider": False}),
        ({}, {"filter_by_provider": False}),
    ],
)
def test_gateway_list_secrets_parse_params(arguments, expected_params):
    assert GatewayListSecretsEvent.parse(arguments) == expected_params


def test_simulate_conversation_parse_params():
    result = SimulateConversationEvent.parse({})
    assert result == {"callsite": "conversation_simulator"}


def test_optimize_prompts_job_event_name():
    assert OptimizePromptsJobEvent.name == "optimize_prompts_job"


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {"optimizer_type": "gepa", "scorer_names": ["Correctness", "Safety"]},
            {"optimizer_type": "gepa", "scorer_count": 2},
        ),
        (
            {"optimizer_type": "metaprompt", "scorer_names": ["Correctness"]},
            {"optimizer_type": "metaprompt", "scorer_count": 1},
        ),
        (
            {"optimizer_type": "gepa", "scorer_names": []},
            {"optimizer_type": "gepa", "scorer_count": 0},
        ),
        ({}, None),
    ],
)
def test_optimize_prompts_job_parse_params(arguments, expected_params):
    assert OptimizePromptsJobEvent.parse(arguments) == expected_params
