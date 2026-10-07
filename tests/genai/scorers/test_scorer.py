import json
import os
import sys
import threading
import time
import types
from collections import defaultdict
from dataclasses import asdict
from unittest.mock import call, patch

import pandas as pd
import pytest

import mlflow
from mlflow.entities import Assessment, AssessmentSource, AssessmentSourceType, Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.environment_variables import _MLFLOW_IN_JOB_EXECUTOR
from mlflow.exceptions import MlflowException
from mlflow.genai import Scorer, scorer
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    QualityThreshold,
    RetrievalGroundedness,
    make_scorer_ensemble,
)
from mlflow.genai.scorers.base import (
    ScorerSamplingConfig,
    ScorerStatus,
    SerializedScorer,
    _is_tracking_server_process,
    _job_executor_scorer_context,
    _serialized_scorer_is_custom_code,
    _UnexecutedDecoratorScorer,
)
from mlflow.genai.scorers.registry import get_scorer, list_scorer_versions, list_scorers
from mlflow.utils.timeout import MlflowTimeoutError


@pytest.fixture(autouse=True)
def increase_db_pool_size(monkeypatch):
    # Set larger pool size for tests to handle concurrent trace creation
    # test_extra_traces_from_customer_scorer_should_be_cleaned_up test requires this
    # to reduce flakiness
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOL_SIZE", "20")
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW", "40")
    return


def _create_test_trace(name, inputs, outputs):
    with mlflow.start_span(name=name) as span:
        span.set_inputs(inputs)
        span.set_outputs(outputs)
    return mlflow.get_trace(span.trace_id)


def always_yes(inputs, outputs, expectations, trace):
    return "yes"


class AlwaysYesScorer(Scorer):
    def __call__(self, inputs, outputs, expectations, trace):
        return "yes"


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "inputs": [
            {"message": [{"role": "user", "content": "What is Spark??"}]},
            {
                "messages": [
                    {"role": "user", "content": "How can you minimize data shuffling in Spark?"}
                ]
            },
        ],
        "outputs": [
            {"choices": [{"message": {"content": "actual response for first question"}}]},
            {"choices": [{"message": {"content": "actual response for second question"}}]},
        ],
        "expectations": [
            {"expected_response": "expected response for first question"},
            {"expected_response": "expected response for second question"},
        ],
    })


@pytest.mark.parametrize("dummy_scorer", [AlwaysYesScorer(name="always_yes"), scorer(always_yes)])
def test_scorer_existence_in_metrics(sample_data, dummy_scorer, is_in_databricks):
    result = mlflow.genai.evaluate(data=sample_data, scorers=[dummy_scorer])
    assert any("always_yes" in metric for metric in result.metrics.keys())


@pytest.mark.parametrize(
    "dummy_scorer", [AlwaysYesScorer(name="always_no"), scorer(name="always_no")(always_yes)]
)
def test_scorer_name_works(sample_data, dummy_scorer, is_in_databricks):
    _SCORER_NAME = "always_no"
    result = mlflow.genai.evaluate(data=sample_data, scorers=[dummy_scorer])
    assert any(_SCORER_NAME in metric for metric in result.metrics.keys())


def test_trace_passed_to_builtin_scorers_correctly(
    sample_rag_trace, is_in_databricks, monkeypatch: pytest.MonkeyPatch
):
    if not is_in_databricks:
        pytest.skip("OSS GenAI evaluator doesn't support passing traces yet")

    # Disable logging traces to MLflow to avoid calling mlflow APIs which need to be mocked
    monkeypatch.setenv("AGENT_EVAL_LOG_TRACES_TO_MLFLOW_ENABLED", "false")

    # Remove expected_facts from trace to avoid validation error (can only have one)
    sample_rag_trace.info.assessments = [
        a for a in sample_rag_trace.info.assessments if a.name != "expected_facts"
    ]

    with (
        patch(
            "databricks.agents.evals.judges.correctness",
            return_value=Feedback(name="correctness", value=CategoricalRating.YES),
        ) as mock_correctness,
        patch(
            "databricks.agents.evals.judges.guidelines",
            return_value=Feedback(name="guidelines", value=CategoricalRating.YES),
        ) as mock_guidelines,
        patch(
            "databricks.agents.evals.judges.groundedness",
            return_value=Feedback(name="groundedness", value=CategoricalRating.YES),
        ) as mock_groundedness,
    ):
        mlflow.genai.evaluate(
            data=pd.DataFrame({"trace": [sample_rag_trace]}),
            scorers=[
                RetrievalGroundedness(name="retrieval_groundedness"),
                Correctness(name="correctness"),
                Guidelines(name="english", guidelines=["write in english"]),
            ],
        )

    assert mock_correctness.call_count == 1
    assert mock_guidelines.call_count == 1
    assert mock_groundedness.call_count == 2  # Called per retriever span

    mock_correctness.assert_called_once_with(
        request="{'question': 'query'}",
        response="answer",
        expected_facts=None,
        expected_response="expected answer",
        assessment_name="correctness",
    )
    mock_guidelines.assert_called_once_with(
        guidelines=["write in english"],
        context={"request": "{'question': 'query'}", "response": "answer"},
        assessment_name="english",
    )
    mock_groundedness.assert_has_calls([
        call(
            request="{'question': 'query'}",
            response="answer",
            retrieved_context=[
                {"content": "content_1", "doc_uri": "url_1"},
                {"content": "content_2", "doc_uri": "url_2"},
            ],
            assessment_name="retrieval_groundedness",
        ),
        call(
            request="{'question': 'query'}",
            response="answer",
            retrieved_context=[
                {"content": "content_3"},
            ],
            assessment_name="retrieval_groundedness",
        ),
    ])


def test_trace_passed_to_custom_scorer_correctly(sample_data, is_in_databricks):
    if not is_in_databricks:
        pytest.skip("OSS GenAI evaluator doesn't support passing traces yet")

    actual_call_args_list = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations, trace) -> float:
        actual_call_args_list.append({
            "inputs": inputs,
            "outputs": outputs,
            "expectations": expectations,
        })
        return 0.0

    mlflow.genai.evaluate(data=sample_data, scorers=[dummy_scorer])

    assert len(actual_call_args_list) == len(sample_data)

    # Prepare expected arguments, keyed by expected_response for matching
    sample_data_set = defaultdict(set)
    for i in range(len(sample_data)):
        sample_data_set["inputs"].add(str(sample_data["inputs"][i]))
        sample_data_set["outputs"].add(str(sample_data["outputs"][i]))
        sample_data_set["expectations"].add(
            str(sample_data["expectations"][i]["expected_response"])
        )

    for actual_args in actual_call_args_list:
        # do any check since actual passed input could be reformatted and larger than sample input
        assert any(
            sample_data_input in str(actual_args["inputs"])
            for sample_data_input in sample_data_set["inputs"]
        )
        assert str(actual_args["outputs"]) in sample_data_set["outputs"]
        assert (
            str(actual_args["expectations"]["expected_response"]) in sample_data_set["expectations"]
        )


def test_trace_passed_correctly(is_in_databricks):
    if not is_in_databricks:
        pytest.skip("OSS GenAI evaluator doesn't support passing traces yet")

    @mlflow.trace
    def predict_fn(question):
        return "output: " + str(question)

    actual_call_args_list = []

    @scorer
    def dummy_scorer(inputs, outputs, trace):
        actual_call_args_list.append({
            "inputs": inputs,
            "outputs": outputs,
            "trace": trace,
        })
        return 0.0

    data = [
        {"inputs": {"question": "input1"}},
        {"inputs": {"question": "input2"}},
    ]
    mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=data,
        scorers=[dummy_scorer],
    )

    assert len(actual_call_args_list) == len(data)
    for actual_args in actual_call_args_list:
        assert actual_args["trace"] is not None
        trace = actual_args["trace"]
        # check if the input is present in the trace
        assert any(
            str(data[i]["inputs"]["question"]) in str(trace.data.request) for i in range(len(data))
        )
        # check if predict_fn was run by making output it starts with "output:"
        assert "output:" in str(trace.data.response)[:10]


@pytest.mark.parametrize(
    "scorer_return",
    [
        "yes",
        42,
        42.0,
        # Feedback object.
        Feedback(name="big_question", value=42, rationale="It's the answer to everything"),
        # List of Feedback objects.
        [
            Feedback(name="big_question", value=42, rationale="It's the answer to everything"),
            Feedback(name="small_question", value=1, rationale="Not sure, just a guess"),
        ],
    ],
)
def test_scorer_on_genai_evaluate(sample_data, scorer_return, is_in_databricks):
    @scorer
    def dummy_scorer(inputs, outputs):
        return scorer_return

    results = mlflow.genai.evaluate(
        data=sample_data,
        scorers=[dummy_scorer],
    )
    if isinstance(scorer_return, Assessment):
        assert any(scorer_return.name in metric for metric in results.metrics.keys())
    elif isinstance(scorer_return, list) and all(
        isinstance(item, Assessment) for item in scorer_return
    ):
        assert any(
            item.name in metric for item in scorer_return for metric in results.metrics.keys()
        )
    else:
        assert any("dummy_scorer" in metric for metric in results.metrics.keys())


def test_custom_scorer_allow_none_return():
    @scorer
    def dummy_scorer(inputs, outputs):
        return None

    assert dummy_scorer.run(inputs={"question": "query"}, outputs="answer") is None


def test_scorer_returns_feedback_with_error(sample_data, is_in_databricks):
    @scorer
    def dummy_scorer(inputs):
        return Feedback(
            name="feedback_with_error",
            error=AssessmentError(error_code="500", error_message="This is an error"),
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt"),
            metadata={"index": 0},
        )

    results = mlflow.genai.evaluate(
        data=sample_data,
        scorers=[dummy_scorer],
    )

    # Scorer should not be in result when it returns an error
    assert all("dummy_scorer" not in metric for metric in results.metrics.keys())


@pytest.mark.parametrize(
    ("scorer_return", "expected_feedback_name"),
    [
        # Single feedback object with default name -> should be renamed to "my_scorer"
        (Feedback(value=42, rationale="rationale"), "my_scorer"),
        # Single feedback object with custom name -> should NOT be renamed to "my_scorer"
        (Feedback(name="custom_name", value=42, rationale="rationale"), "custom_name"),
    ],
)
def test_custom_scorer_overwrites_default_feedback_name(scorer_return, expected_feedback_name):
    @scorer
    def my_scorer(inputs, outputs):
        return scorer_return

    feedback = my_scorer.run(
        inputs={"question": "What is the capital of France?"},
        outputs="The capital of France is Paris.",
    )
    assert feedback.name == expected_feedback_name
    assert feedback.value == 42


def test_custom_scorer_does_not_overwrite_feedback_name_when_returning_list():
    @scorer
    def my_scorer(inputs, outputs):
        return [
            Feedback(name="big_question", value=42, rationale="It's the answer to everything"),
            Feedback(name="small_question", value=1, rationale="Not sure, just a guess"),
        ]

    feedbacks = my_scorer.run(
        inputs={"question": "What is the capital of France?"},
        outputs="The capital of France is Paris.",
    )
    assert feedbacks[0].name == "big_question"
    assert feedbacks[1].name == "small_question"


def test_custom_scorer_registration_blocked_for_non_databricks_uri():
    experiment_id = mlflow.create_experiment("test_security_experiment")

    @scorer
    def test_custom_scorer(outputs) -> bool:
        return len(outputs) > 0

    with pytest.raises(
        mlflow.exceptions.MlflowException,
        match="Custom scorer registration.*disabled by default outside of Databricks tracking",
    ):
        test_custom_scorer.register(experiment_id=experiment_id, name="test_scorer")

    mlflow.delete_experiment(experiment_id)


def test_custom_scorer_loading_blocked_for_non_databricks_uri():
    serialized = SerializedScorer(
        name="malicious_scorer",
        is_session_level_scorer=False,
        call_source="import os\nos.system('echo hacked')\nreturn True",
        call_signature="(outputs)",
        original_func_name="malicious_scorer",
    )

    with pytest.raises(
        mlflow.exceptions.MlflowException, match="Custom scorer registration.*disabled by default"
    ):
        Scorer._reconstruct_decorator_scorer(serialized)


def test_custom_scorer_loading_allowed_for_databricks_remote_access():
    serialized = SerializedScorer(
        name="test_scorer",
        is_session_level_scorer=False,
        call_source="return len(outputs) > 0",
        call_signature="(outputs)",
        original_func_name="test_scorer",
    )

    with patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True):
        result = Scorer._reconstruct_decorator_scorer(serialized)
        assert result.name == "test_scorer"


def test_custom_scorer_loading_allowed_when_flag_enabled(monkeypatch):
    serialized = SerializedScorer(
        name="test_scorer",
        is_session_level_scorer=False,
        call_source="return len(outputs) > 0",
        call_signature="(outputs)",
        original_func_name="test_scorer",
    )

    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    result = Scorer._reconstruct_decorator_scorer(serialized)
    assert result.name == "test_scorer"


def _decorator_serialized(name="test_scorer", is_session_level=False):
    return SerializedScorer(
        name=name,
        is_session_level_scorer=is_session_level,
        call_source="return len(outputs) > 0",
        call_signature="(outputs)",
        original_func_name=name,
    )


def test_serialized_scorer_is_custom_code():
    decorator = asdict(_decorator_serialized())
    assert _serialized_scorer_is_custom_code(decorator) is True
    assert _serialized_scorer_is_custom_code(json.dumps(decorator)) is True
    # A SerializedScorer object (what ScorerVersion.serialized_scorer returns) is handled too.
    assert _serialized_scorer_is_custom_code(_decorator_serialized()) is True
    assert (
        _serialized_scorer_is_custom_code(
            SerializedScorer(name="safety", builtin_scorer_class="Safety")
        )
        is False
    )
    # A built-in scorer's serialized form carries no decorator source.
    assert (
        _serialized_scorer_is_custom_code({"name": "safety", "builtin_scorer_class": "Safety"})
        is False
    )
    with pytest.raises(MlflowException, match="Malformed serialized scorer"):
        _serialized_scorer_is_custom_code("not valid json")
    # An ensemble embedding a custom @scorer counts as custom code (checked recursively).
    ensemble_with_custom = {
        "ensemble_scorer_data": {
            "ensemble_fn": "majority",
            "scorers": [asdict(_decorator_serialized())],
        }
    }
    assert _serialized_scorer_is_custom_code(ensemble_with_custom) is True
    # An ensemble of only built-in scorers does not.
    ensemble_builtin = {
        "ensemble_scorer_data": {
            "ensemble_fn": "majority",
            "scorers": [{"name": "safety", "builtin_scorer_class": "Safety"}],
        }
    }
    assert _serialized_scorer_is_custom_code(ensemble_builtin) is False


def test_server_process_returns_non_executing_scorer(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "boot")
    monkeypatch.delenv("_MLFLOW_IN_JOB_EXECUTOR", raising=False)

    with patch("mlflow.genai.scorers.scorer_utils.recreate_function") as recreate:
        loaded = Scorer.model_validate(_decorator_serialized(is_session_level=True))

    # The server never executes custom scorer source.
    recreate.assert_not_called()
    assert isinstance(loaded, _UnexecutedDecoratorScorer)
    assert loaded.name == "test_scorer"
    assert loaded.is_session_level_scorer is True
    # It still re-serializes so the server can forward it to the executor.
    assert loaded.model_dump()["call_source"] == "return len(outputs) > 0"
    # It cannot be run in the server.
    with pytest.raises(MlflowException, match="runs only inside the job executor"):
        loaded(outputs="abc")


def test_job_executor_reconstructs_scorer(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "boot")
    monkeypatch.setenv("_MLFLOW_IN_JOB_EXECUTOR", "true")

    loaded = Scorer.model_validate(_decorator_serialized())

    assert not isinstance(loaded, _UnexecutedDecoratorScorer)
    assert loaded(outputs="abc") is True


def test_client_process_reconstructs_scorer(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    monkeypatch.delenv("_MLFLOW_SERVER_BOOT_ID", raising=False)
    monkeypatch.delenv("_MLFLOW_IN_JOB_EXECUTOR", raising=False)

    loaded = Scorer.model_validate(_decorator_serialized())

    assert not isinstance(loaded, _UnexecutedDecoratorScorer)
    assert loaded(outputs="abc") is True


def test_server_process_detected_for_direct_app_launch(monkeypatch):
    # A server started by importing the app directly (e.g. `gunicorn mlflow.server:app`) has no
    # boot id, but mlflow.server.is_running_as_server is True. Such a process must be treated as a
    # server (and so must not reconstruct custom scorer code).
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    monkeypatch.delenv("_MLFLOW_SERVER_BOOT_ID", raising=False)
    monkeypatch.delenv("_MLFLOW_IN_JOB_EXECUTOR", raising=False)

    fake_server = types.ModuleType("mlflow.server")
    fake_server.is_running_as_server = True
    monkeypatch.setitem(sys.modules, "mlflow.server", fake_server)

    assert _is_tracking_server_process() is True
    loaded = Scorer.model_validate(_decorator_serialized())
    assert isinstance(loaded, _UnexecutedDecoratorScorer)


def test_reconstruct_decorator_scorer_blocked_in_server_process(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "boot")
    monkeypatch.delenv("_MLFLOW_IN_JOB_EXECUTOR", raising=False)

    with pytest.raises(MlflowException, match="cannot be reconstructed in the MLflow"):
        Scorer._reconstruct_decorator_scorer(_decorator_serialized())


@pytest.mark.parametrize("preset", [None, "false", "true"])
def test_job_executor_scorer_context_sets_and_restores_marker(monkeypatch, preset):
    if preset is None:
        monkeypatch.delenv("_MLFLOW_IN_JOB_EXECUTOR", raising=False)
    else:
        monkeypatch.setenv("_MLFLOW_IN_JOB_EXECUTOR", preset)
    before = os.environ.get("_MLFLOW_IN_JOB_EXECUTOR")

    with _job_executor_scorer_context():
        assert _MLFLOW_IN_JOB_EXECUTOR.get() is True

    # The prior value is restored exactly, whether it was absent or a specific string.
    assert os.environ.get("_MLFLOW_IN_JOB_EXECUTOR") == before


def test_custom_scorer_registration_allowed_when_flag_enabled(monkeypatch):
    @scorer
    def flagged_scorer(outputs) -> bool:
        return len(outputs) > 0

    # Off by default: registration is rejected on a non-Databricks URI.
    monkeypatch.delenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", raising=False)
    with pytest.raises(
        mlflow.exceptions.MlflowException,
        match="Custom scorer registration.*disabled by default outside of Databricks tracking",
    ):
        flagged_scorer._check_can_be_registered()

    # Opted in: the decorator-scorer guard no longer blocks registration.
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    flagged_scorer._check_can_be_registered()


def test_custom_scorer_registration_deferred_to_remote_server(monkeypatch):
    @scorer
    def remote_scorer(outputs) -> bool:
        return len(outputs) > 0

    # Against a remote HTTP server the server's own handler enforces the flag, so the
    # client-side guard must not block (and must not require a remote client to set the
    # server variable) even with the flag unset locally.
    monkeypatch.delenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", raising=False)
    monkeypatch.setattr(
        "mlflow.genai.scorers.base.get_tracking_uri", lambda: "http://localhost:5000"
    )
    remote_scorer._check_can_be_registered()


def test_custom_scorer_error_message_renders_code_snippet_legibly():
    serialized = SerializedScorer(
        name="complex_scorer",
        is_session_level_scorer=False,
        call_source=(
            "if not outputs:\n"
            "    return 0\n"
            "score = 0\n"
            "for word in outputs.split():\n"
            "    if word.isupper():\n"
            "        score += 2\n"
            "    else:\n"
            "        score += 1\n"
            "return score"
        ),
        call_signature="(outputs)",
        original_func_name="complex_scorer",
    )

    with pytest.raises(
        mlflow.exceptions.MlflowException, match="is disabled by default outside of"
    ) as exc_info:
        Scorer._reconstruct_decorator_scorer(serialized)

    error_msg = str(exc_info.value)

    assert "Registered scorer code:" in error_msg
    assert "from mlflow.genai import scorer" in error_msg
    assert "@scorer" in error_msg
    assert "def complex_scorer(outputs):" in error_msg

    expected_code = """
from mlflow.genai import scorer

@scorer
def complex_scorer(outputs):
    if not outputs:
        return 0
    score = 0
    for word in outputs.split():
        if word.isupper():
            score += 2
        else:
            score += 1
    return score"""

    assert expected_code.strip() in error_msg


def test_session_level_scorer_auto_detected_from_session_param():
    @scorer
    def session_scorer(session):
        return len(session)

    assert session_scorer.is_session_level_scorer is True


def test_session_level_scorer_with_expectations():
    @scorer
    def session_scorer(session, expectations):
        return len(session)

    assert session_scorer.is_session_level_scorer is True


def test_single_turn_scorer_not_session_level():
    @scorer
    def single_turn(inputs, outputs):
        return True

    assert single_turn.is_session_level_scorer is False


def test_session_param_with_single_turn_params_raises():
    def with_trace(session, trace):
        pass

    def with_inputs(session, inputs):
        pass

    def with_outputs(session, outputs):
        pass

    def with_all(session, inputs, outputs, trace):
        pass

    for func in [with_trace, with_inputs, with_outputs, with_all]:
        with pytest.raises(
            mlflow.exceptions.MlflowException,
            match="Session-level scorers.*cannot also accept",
        ):
            scorer(func)


def test_session_level_scorer_serialization_roundtrip(is_in_databricks):
    @scorer
    def session_scorer(session):
        return len(session)

    dumped = session_scorer.model_dump()
    assert dumped["is_session_level_scorer"] is True

    with patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True):
        loaded = Scorer.model_validate(dumped)
        assert loaded.name == "session_scorer"
        assert loaded.is_session_level_scorer is True


def test_session_level_scorer_invocation_with_traces():
    @scorer
    def session_scorer(session) -> Feedback:
        total = len(session)
        errors = sum(1 for t in session if t.info.state == "ERROR")
        return Feedback(value=errors == 0, rationale=f"{total} turns, {errors} errors")

    traces = [
        _create_test_trace("turn_1", {"question": "What is MLflow?"}, "An ML platform."),
        _create_test_trace("turn_2", {"question": "How do I track?"}, "Use mlflow.log_param()."),
        _create_test_trace("turn_3", {"question": "Thanks!"}, "You're welcome!"),
    ]

    result = session_scorer.run(session=traces)
    assert result.value is True
    assert "3 turns" in result.rationale
    assert "0 errors" in result.rationale


def test_make_judge_scorer_works_without_databricks_uri():
    experiment_id = mlflow.create_experiment("test_make_judge_experiment")

    judge_scorer = make_judge(
        instructions="Evaluate if the {{outputs}} is helpful and relevant",
        name="helpfulness_judge",
        feedback_value_type=str,
    )

    registered_scorer = judge_scorer.register(experiment_id=experiment_id, name="helpfulness_judge")

    assert registered_scorer is not None
    assert registered_scorer.name == "helpfulness_judge"

    retrieved_scorer = get_scorer(name="helpfulness_judge", experiment_id=experiment_id)
    assert retrieved_scorer is not None
    assert retrieved_scorer.name == "helpfulness_judge"

    scorers = list_scorers(experiment_id=experiment_id)
    assert len(scorers) == 1
    assert scorers[0].name == "helpfulness_judge"

    mlflow.delete_experiment(experiment_id)


def test_scorer_pass_if_is_exposed():
    @scorer(pass_if=lambda v: v >= 0.5)
    def my_score(outputs):
        return 0.6

    assert my_score.pass_if is not None
    assert my_score.pass_if(0.6) is True

    @scorer
    def plain(outputs):
        return True

    assert plain.pass_if is None


def test_scorer_default_timeout_is_none_sentinel():
    @scorer
    def s(outputs) -> bool:
        return True

    assert s.timeout is None


def test_scorer_default_timeout_uses_constant(monkeypatch):
    # A scorer with no explicit timeout is bounded by DEFAULT_SCORER_TIMEOUT.
    monkeypatch.setattr("mlflow.genai.scorers.base.DEFAULT_SCORER_TIMEOUT", 0.2)
    release = threading.Event()

    @scorer
    def slow(outputs) -> bool:
        release.wait(10)
        return True

    try:
        with pytest.raises(MlflowTimeoutError, match="timed out after 0.2 seconds"):
            slow.run(outputs="x")
    finally:
        release.set()


@pytest.mark.parametrize("timeout", [None, 0, 30, 5.5])
def test_scorer_timeout_value_preserved(timeout):
    @scorer(timeout=timeout)
    def s(outputs) -> bool:
        return True

    assert s.timeout == timeout


@pytest.mark.parametrize("bad", [-1, True, False, "5", float("inf"), float("nan")])
def test_scorer_rejects_invalid_timeout(bad):
    with pytest.raises(MlflowException, match="must be a non-negative"):

        @scorer(timeout=bad)
        def s(outputs) -> bool:
            return True


def test_scorer_run_times_out_slow_scorer():
    # Event (released below) instead of a bare sleep so the abandoned daemon thread doesn't linger.
    release = threading.Event()

    @scorer(timeout=0.2)
    def slow(outputs) -> bool:
        release.wait(10)
        return True

    start = time.time()
    try:
        with pytest.raises(MlflowTimeoutError, match="timed out after 0.2 seconds"):
            slow.run(outputs="x")
        # Abandoned well before the 10s the scorer would otherwise take.
        assert time.time() - start < 3
    finally:
        release.set()


def test_scorer_run_within_timeout_returns_normally():
    @scorer(timeout=5)
    def ok(outputs) -> bool:
        return outputs == "good"

    assert ok.run(outputs="good") is True
    assert ok.run(outputs="bad") is False


def test_scorer_run_propagates_non_timeout_exception():
    @scorer(timeout=5)
    def boom(outputs) -> bool:
        raise ValueError("boom in scorer")

    with pytest.raises(ValueError, match="boom in scorer"):
        boom.run(outputs="x")


def test_scorer_timeout_zero_runs_unbounded(monkeypatch):
    # `timeout=0` disables the timeout even when the default is tiny.
    monkeypatch.setattr("mlflow.genai.scorers.base.DEFAULT_SCORER_TIMEOUT", 0.05)

    @scorer(timeout=0)
    def slow(outputs) -> bool:
        time.sleep(0.2)
        return True

    assert slow.run(outputs="x") is True


def test_scorer_timeout_becomes_error_feedback_in_evaluate(sample_data, is_in_databricks):
    release = threading.Event()

    @scorer(timeout=0.2)
    def slow_scorer(inputs) -> bool:
        release.wait(10)
        return True

    @scorer
    def fast_scorer(inputs) -> bool:
        return True

    try:
        results = mlflow.genai.evaluate(data=sample_data, scorers=[slow_scorer, fast_scorer])
    finally:
        release.set()

    # The fast scorer yields its metric; the timed-out one errors and is excluded. The control
    # scorer makes the absence specifically a timeout, not scorers being dropped wholesale.
    metrics = results.metrics.keys()
    assert any("fast_scorer" in metric for metric in metrics)
    assert all("slow_scorer" not in metric for metric in metrics)


@pytest.mark.parametrize(
    "kwargs",
    [{"at_least": 0.9}, {"at_most": 2}, {"at_least": 0.5, "aggregation": "p90"}],
)
def test_quality_threshold_accepts_valid_bounds(kwargs):
    assert QualityThreshold(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one of `at_least` or `at_most`"),
        ({"at_least": 0.1, "at_most": 0.9}, "exactly one of `at_least` or `at_most`"),
        ({"at_least": True}, "must be a finite number"),
        ({"at_least": "0.9"}, "must be a finite number"),
        ({"at_most": float("nan")}, "must be a finite number"),
        ({"at_least": float("inf")}, "must be a finite number"),
        ({"at_least": 0.9, "aggregation": "sum"}, "`aggregation` must be one of"),
    ],
)
def test_quality_threshold_rejects_invalid_bounds(kwargs, match):
    with pytest.raises(MlflowException, match=match):
        QualityThreshold(**kwargs)


def _threshold_scorers():
    @scorer
    def decorated(outputs) -> float:
        return 1.0

    judge = make_judge(
        name="tone", instructions="Is {{ outputs }} polite?", feedback_value_type=bool
    )
    ensemble = make_scorer_ensemble(name="ensemble", scorers=[Correctness()], ensemble_fn="agg_all")
    return [decorated, Correctness(), judge, ensemble]


@pytest.mark.parametrize("value", [0.9, 1, QualityThreshold(at_most=2.0, aggregation="p90")])
@pytest.mark.parametrize("original", _threshold_scorers(), ids=lambda s: s.name)
def test_with_quality_threshold_returns_copy_with_threshold(original, value):
    copy = original.with_quality_threshold(value)

    assert type(copy) is type(original)
    assert copy.quality_threshold == value
    assert original.quality_threshold is None
    assert copy.with_quality_threshold(None).quality_threshold is None


def test_with_quality_threshold_keeps_registration_metadata():
    sampling_config = ScorerSamplingConfig(sample_rate=0.5)
    registered = Correctness()._set_registration_metadata(
        backend="tracking", experiment_id="123", sampling_config=sampling_config, scorer_version=2
    )

    copy = registered.with_quality_threshold(0.9)

    assert copy.scorer_version == 2
    assert copy.status == ScorerStatus.STARTED
    assert copy._experiment_id == "123"
    assert copy._sampling_config == sampling_config


def test_scorer_quality_threshold_defaults_to_none():
    @scorer
    def s(outputs) -> bool:
        return True

    assert s.quality_threshold is None
    assert Correctness().quality_threshold is None


@pytest.mark.parametrize("bad", [True, "0.9", float("nan")])
def test_with_quality_threshold_rejects_invalid_value(bad):
    with pytest.raises(MlflowException, match="must be a finite number"):
        Correctness().with_quality_threshold(bad)


@pytest.mark.parametrize("factory", [scorer, make_judge, make_scorer_ensemble])
def test_scorer_factories_do_not_accept_quality_threshold(factory):
    with pytest.raises(TypeError, match="quality_threshold"):
        factory(quality_threshold=0.9)


def test_scorer_copy_preserves_quality_threshold():
    threshold = QualityThreshold(at_most=0.2)
    ensemble = make_scorer_ensemble(
        name="ensemble", scorers=[Correctness()], ensemble_fn="agg_all"
    ).with_quality_threshold(threshold)

    assert ensemble._create_copy().quality_threshold == threshold
    assert Correctness().with_quality_threshold(0.9)._create_copy().quality_threshold == 0.9


def test_registered_versions_keep_their_own_quality_threshold():
    experiment_id = mlflow.create_experiment("test_quality_threshold_versions")
    tone = make_judge(
        name="tone", instructions="Is {{ outputs }} polite?", feedback_value_type=bool
    )
    v2_threshold = QualityThreshold(at_least=0.8, aggregation="min")

    with patch("mlflow.genai.scorers.base._logger.warning") as mock_warning:
        tone.with_quality_threshold(0.6).register(experiment_id=experiment_id)
        tone.with_quality_threshold(v2_threshold).register(experiment_id=experiment_id)

    assert [c[0][0] for c in mock_warning.call_args_list] == [
        "`quality_threshold` (at least 0.6 on the mean) is saved with version 1 of scorer "
        "'tone'. A new threshold creates a new scorer version.",
        "`quality_threshold` (at least 0.8 on the min) is saved with version 2 of scorer "
        "'tone'. A new threshold creates a new scorer version.",
    ]
    v1 = get_scorer(name="tone", experiment_id=experiment_id, version=1)
    v2 = get_scorer(name="tone", experiment_id=experiment_id, version=2)
    assert (v1.scorer_version, v1.quality_threshold) == (1, 0.6)
    assert (v2.scorer_version, v2.quality_threshold) == (2, v2_threshold)
    assert [
        (version, s.quality_threshold)
        for s, version in list_scorer_versions(name="tone", experiment_id=experiment_id)
    ] == [(1, 0.6), (2, v2_threshold)]
    [latest] = list_scorers(experiment_id=experiment_id)
    assert latest.quality_threshold == v2_threshold


def test_register_warns_that_decorator_quality_threshold_is_not_saved(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    experiment_id = mlflow.create_experiment("test_quality_threshold_decorator")

    @scorer
    def is_concise(outputs) -> bool:
        return len(outputs) < 100

    with patch("mlflow.genai.scorers.base._logger.warning") as mock_warning:
        is_concise.with_quality_threshold(0.9).register(experiment_id=experiment_id)

    mock_warning.assert_called_once()
    assert "is not saved with the registered scorer" in mock_warning.call_args[0][0]
    assert get_scorer(name="is_concise", experiment_id=experiment_id).quality_threshold is None


def test_databricks_register_sends_definition_that_changes_only_with_threshold():
    sent = []

    def upsert(experiment_id, config):
        sent.append(json.loads(config.serialized_scorer))
        return [config]

    with (
        patch(
            "mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks"
        ),
        patch(
            "mlflow.genai.scorers.registry.DatabricksStore._upsert_registered_scorer_config",
            side_effect=upsert,
        ),
        patch("mlflow.genai.scorers.registry.DatabricksStore._resolve_experiment_id"),
    ):
        Correctness().register()
        for threshold in [0.6, 0.6, 0.8]:
            Correctness().with_quality_threshold(threshold).register()

    without, first, unchanged, changed = sent
    assert "quality_threshold" not in json.dumps(without)
    assert unchanged == first
    assert changed != first
    assert changed["builtin_scorer_pydantic_data"]["quality_threshold"] == 0.8
