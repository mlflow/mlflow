from unittest.mock import Mock

import pandas as pd
import pytest

from mlflow.entities.evaluation_dataset import DatasetGranularity, EvaluationDataset
from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
)
from mlflow.entities.issue import Issue, IssueSeverity, IssueStatus
from mlflow.genai.discovery.entities import DiscoverIssuesResult
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
    DiscoverIssuesEvent,
    EvaluateEvent,
    GatewayCreateBudgetPolicyEvent,
    GatewayCreateEndpointEvent,
    GatewayCreateGuardrailEvent,
    GatewayCreateModelDefinitionEvent,
    GatewayCreateSecretEvent,
    GatewayDeleteGuardrailEvent,
    GatewayListBudgetPoliciesEvent,
    GatewayListEndpointsEvent,
    GatewayListSecretsEvent,
    GatewayUpdateEndpointEvent,
    GatewayUpdateGuardrailEvent,
    GenAIEvaluateEvent,
    LogAssessmentEvent,
    MakeJudgeEvent,
    MergeRecordsEvent,
    OptimizePromptsJobEvent,
    PromptOptimizationEvent,
    SimulateConversationEvent,
    StartTraceEvent,
    TraceAttachmentsEvent,
    UpdateIssueEvent,
)
from mlflow.tracing.attachments import Attachment


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
            {"flavor": "mlflow.pyfunc", "serialization_format": "cloudpickle"},
            {"flavor": "pyfunc", "serialization_format": "cloudpickle"},
        ),
        (
            {"serialization_format": "cloudpickle"},
            {"serialization_format": "cloudpickle"},
        ),
        (
            {
                "flavor": None,
            },
            None,
        ),
        ({}, None),
        (
            {"flavor": "mlflow.pyfunc", "uses_uv": True},
            {"flavor": "pyfunc", "uses_uv": True},
        ),
        (
            {"flavor": "mlflow.pyfunc", "uses_uv": False},
            {"flavor": "pyfunc"},
        ),
        (
            {"uses_uv": True},
            {"uses_uv": True},
        ),
        (
            {"flavor": "sklearn", "serialization_format": "cloudpickle", "uses_uv": True},
            {"flavor": "sklearn", "serialization_format": "cloudpickle", "uses_uv": True},
        ),
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
    assert DiscoverIssuesEvent.name == "discover_issues"
    assert UpdateIssueEvent.name == "update_issue"
    assert GatewayCreateGuardrailEvent.name == "gateway_create_guardrail"
    assert GatewayUpdateGuardrailEvent.name == "gateway_update_guardrail"
    assert GatewayDeleteGuardrailEvent.name == "gateway_delete_guardrail"
    assert GatewayCreateModelDefinitionEvent.name == "gateway_create_model_definition"


def test_start_trace_parse_format_native():
    result = StartTraceEvent.parse({})
    assert result["format"] == "native"


def test_start_trace_parse_format_genai_semconv(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_OTEL_GENAI_SEMCONV", "true")
    result = StartTraceEvent.parse({})
    assert result["format"] == "genai_semconv"


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
                "usage_tracking": True,
            },
            {
                "has_fallback_config": True,
                "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
                "num_model_configs": 2,
                "usage_tracking": True,
            },
        ),
        (
            {
                "fallback_config": None,
                "routing_strategy": None,
                "model_configs": [{"model_definition_id": "md-1"}],
                "usage_tracking": False,
            },
            {
                "has_fallback_config": False,
                "routing_strategy": None,
                "num_model_configs": 1,
                "usage_tracking": False,
            },
        ),
        (
            {"fallback_config": None, "routing_strategy": None, "model_configs": []},
            {
                "has_fallback_config": False,
                "routing_strategy": None,
                "num_model_configs": 0,
                "usage_tracking": None,
            },
        ),
        (
            {},
            {
                "has_fallback_config": False,
                "routing_strategy": None,
                "num_model_configs": 0,
                "usage_tracking": None,
            },
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
                "usage_tracking": True,
            },
            {
                "has_fallback_config": True,
                "routing_strategy": "ROUND_ROBIN",
                "num_model_configs": 1,
                "usage_tracking": True,
            },
        ),
        (
            {
                "fallback_config": None,
                "routing_strategy": None,
                "model_configs": None,
                "usage_tracking": None,
            },
            {
                "has_fallback_config": False,
                "routing_strategy": None,
                "num_model_configs": None,
                "usage_tracking": None,
            },
        ),
        (
            {},
            {
                "has_fallback_config": False,
                "routing_strategy": None,
                "num_model_configs": None,
                "usage_tracking": None,
            },
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
        (
            {"model_name": "gpt-4o", "provider": "openai"},
            {"model_name": "gpt-4o", "provider": "openai"},
        ),
        (
            {"model_name": "claude-3-5-sonnet", "provider": "anthropic"},
            {"model_name": "claude-3-5-sonnet", "provider": "anthropic"},
        ),
        ({"model_name": None, "provider": None}, {"model_name": None, "provider": None}),
        ({}, {"model_name": None, "provider": None}),
    ],
)
def test_gateway_create_model_definition_parse_params(arguments, expected_params):
    assert GatewayCreateModelDefinitionEvent.parse(arguments) == expected_params


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


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {
                "budget_unit": "USD",
                "duration": BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
                "target_scope": "GLOBAL",
                "budget_action": "ALERT",
            },
            {
                "budget_unit": "USD",
                "duration_unit": "DAYS",
                "target_scope": "GLOBAL",
                "budget_action": "ALERT",
            },
        ),
        (
            {
                "budget_unit": BudgetUnit.USD,
                "duration": BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=1),
                "target_scope": BudgetTargetScope.WORKSPACE,
                "budget_action": BudgetAction.REJECT,
            },
            {
                "budget_unit": "USD",
                "duration_unit": "MONTHS",
                "target_scope": "WORKSPACE",
                "budget_action": "REJECT",
            },
        ),
        (
            {},
            {
                "budget_unit": None,
                "duration_unit": None,
                "target_scope": None,
                "budget_action": None,
            },
        ),
    ],
)
def test_gateway_create_budget_policy_parse_params(arguments, expected_params):
    assert GatewayCreateBudgetPolicyEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {"stage": "BEFORE", "action": "VALIDATION"},
            {"stage": "BEFORE", "action": "VALIDATION"},
        ),
        (
            {"stage": "AFTER", "action": "SANITIZATION", "action_endpoint_id": "e-123"},
            {"stage": "AFTER", "action": "SANITIZATION"},
        ),
        (
            {},
            {"stage": None, "action": None},
        ),
    ],
)
def test_gateway_create_guardrail_parse_params(arguments, expected_params):
    assert GatewayCreateGuardrailEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {"stage": "BEFORE", "action": "VALIDATION"},
            {"stage": "BEFORE", "action": "VALIDATION"},
        ),
        (
            {"stage": "AFTER", "action": "SANITIZATION", "execution_order": 2},
            {"stage": "AFTER", "action": "SANITIZATION"},
        ),
        ({}, {"stage": None, "action": None}),
    ],
)
def test_gateway_update_guardrail_parse_params(arguments, expected_params):
    assert GatewayUpdateGuardrailEvent.parse(arguments) == expected_params


def test_gateway_list_budget_policies_parse_params():
    assert GatewayListBudgetPoliciesEvent.parse({}) is None


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


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {
                "model": "openai:/gpt-4",
                "traces": [Mock(), Mock()],
                "categories": ["hallucination"],
                "run_id": "run-1",
            },
            {
                "model": "openai:/gpt-4",
                "trace_count": 2,
                "categories": ["hallucination"],
                "source_run_id": "run-1",
            },
        ),
        (
            {"model": "databricks:/dbrx", "traces": [Mock()], "categories": None},
            {
                "model": "databricks:/dbrx",
                "trace_count": 1,
                "categories": None,
                "source_run_id": None,
            },
        ),
        (
            {"model": None, "traces": [], "categories": ["accuracy", "safety"]},
            {
                "model": None,
                "trace_count": 0,
                "categories": ["accuracy", "safety"],
                "source_run_id": None,
            },
        ),
        (
            {"traces": None, "categories": []},
            {"model": None, "trace_count": 0, "categories": [], "source_run_id": None},
        ),
        ({}, {"model": None, "trace_count": 0, "categories": None, "source_run_id": None}),
    ],
)
def test_discover_issues_parse_params(arguments, expected_params):
    assert DiscoverIssuesEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("result", "expected_params"),
    [
        (
            DiscoverIssuesResult(
                issues=[
                    Issue(
                        issue_id="1",
                        experiment_id="exp",
                        name="issue1",
                        description="desc",
                        status=IssueStatus.PENDING,
                        created_timestamp=0,
                        last_updated_timestamp=0,
                    ),
                    Issue(
                        issue_id="2",
                        experiment_id="exp",
                        name="issue2",
                        description="desc",
                        status=IssueStatus.PENDING,
                        created_timestamp=0,
                        last_updated_timestamp=0,
                    ),
                    Issue(
                        issue_id="3",
                        experiment_id="exp",
                        name="issue3",
                        description="desc",
                        status=IssueStatus.PENDING,
                        created_timestamp=0,
                        last_updated_timestamp=0,
                    ),
                ],
                triage_run_id="run",
                summary="summary",
                total_traces_analyzed=100,
                total_cost_usd=2.5,
            ),
            {
                "issue_count": 3,
                "total_traces_analyzed": 100,
                "total_cost_usd": 2.5,
                "triage_run_id": "run",
            },
        ),
        (
            DiscoverIssuesResult(
                issues=[
                    Issue(
                        issue_id="1",
                        experiment_id="exp",
                        name="issue1",
                        description="desc",
                        status=IssueStatus.PENDING,
                        created_timestamp=0,
                        last_updated_timestamp=0,
                    )
                ],
                triage_run_id="run",
                summary="summary",
                total_traces_analyzed=50,
                total_cost_usd=1.0,
            ),
            {
                "issue_count": 1,
                "total_traces_analyzed": 50,
                "total_cost_usd": 1.0,
                "triage_run_id": "run",
            },
        ),
        (
            DiscoverIssuesResult(
                issues=[],
                triage_run_id="run",
                summary="summary",
                total_traces_analyzed=10,
                total_cost_usd=0.0,
            ),
            {
                "issue_count": 0,
                "total_traces_analyzed": 10,
                "total_cost_usd": 0.0,
                "triage_run_id": "run",
            },
        ),
        (
            DiscoverIssuesResult(
                issues=[],
                triage_run_id=None,
                summary="summary",
                total_traces_analyzed=0,
                total_cost_usd=None,
            ),
            {
                "issue_count": 0,
                "total_traces_analyzed": 0,
                "total_cost_usd": None,
                "triage_run_id": None,
            },
        ),
    ],
)
def test_discover_issues_parse_result(result, expected_params):
    assert DiscoverIssuesEvent.parse_result(result) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        # String values pass through; name/description tracked as booleans
        (
            {"status": "pending", "name": "Test Issue", "description": "Desc", "severity": "high"},
            {"status": "pending", "has_name": True, "has_description": True, "severity": "high"},
        ),
        # Enum values are converted to their string value
        (
            {
                "status": IssueStatus.RESOLVED,
                "name": "Issue",
                "description": "Desc",
                "severity": IssueSeverity.MEDIUM,
            },
            {"status": "resolved", "has_name": True, "has_description": True, "severity": "medium"},
        ),
        # Missing fields: status/severity None, has_name/has_description False
        (
            {},
            {"status": None, "has_name": False, "has_description": False, "severity": None},
        ),
    ],
)
def test_update_issue_parse_params(arguments, expected_params):
    assert UpdateIssueEvent.name == "update_issue"
    assert UpdateIssueEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("source_run_id", "expected_params"),
    [
        ("run-123", {"source_run_id": "run-123"}),
        (None, {"source_run_id": None}),
    ],
)
def test_update_issue_parse_result(source_run_id, expected_params):
    mock_issue = Mock()
    mock_issue.source_run_id = source_run_id
    assert UpdateIssueEvent.parse_result(mock_issue) == expected_params


def test_update_issue_parse_result_none():
    assert UpdateIssueEvent.parse_result(None) == {}


@pytest.mark.parametrize(
    ("arguments", "expected_eval_data_type"),
    [
        ({"data": [{"inputs": {"q": "a"}}]}, "list[dict]"),
        ({"data": pd.DataFrame([{"inputs": {"q": "a"}}])}, "pd.DataFrame"),
        ({"data": "unexpected_type"}, "unknown"),
        ({"data": None}, None),
        ({}, None),
    ],
)
def test_genai_evaluate_event_parse_eval_data_type(arguments, expected_eval_data_type):
    result = GenAIEvaluateEvent.parse(arguments)
    assert result.get("eval_data_type") == expected_eval_data_type


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            {
                "attachments": {
                    "a": Attachment(content_type="image/png", content_bytes=b"img1"),
                    "b": Attachment(content_type="audio/wav", content_bytes=b"audio"),
                    "c": Attachment(content_type="image/png", content_bytes=b"img2"),
                }
            },
            {"content_types": {"image/png": 2, "audio/wav": 1}},
        ),
        ({"attachments": {}}, None),
        ({}, None),
    ],
)
def test_trace_attachments_event_parse(arguments, expected):
    assert TraceAttachmentsEvent.parse(arguments) == expected
