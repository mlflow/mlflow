from collections import defaultdict
from unittest.mock import call, patch

import pandas as pd
import pytest

import mlflow
from mlflow.entities import Assessment, AssessmentSource, AssessmentSourceType, Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.genai import Scorer, scorer
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import Correctness, Guidelines, RetrievalGroundedness
from mlflow.genai.scorers.base import SerializedScorer
from mlflow.genai.scorers.registry import get_scorer, list_scorers


@pytest.fixture(autouse=True)
def increase_db_pool_size(monkeypatch):
    # Set larger pool size for tests to handle concurrent trace creation
    # test_extra_traces_from_customer_scorer_should_be_cleaned_up test requires this
    # to reduce flakiness
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOL_SIZE", "20")
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW", "40")
    return


def always_yes(inputs, outputs, expectations, trace):
    return "yes"


class AlwaysYesScorer(Scorer):
    def __call__(self, inputs, outputs, expectations, trace):
        return "yes"


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
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
        }
    )


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
        expected_facts=["fact1", "fact2"],
        expected_response="expected answer",
        assessment_name="correctness",
    )
    mock_guidelines.assert_called_once_with(
        guidelines=["write in english"],
        context={"request": "{'question': 'query'}", "response": "answer"},
        assessment_name="english",
    )
    mock_groundedness.assert_has_calls(
        [
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
        ]
    )


def test_trace_passed_to_custom_scorer_correctly(sample_data, is_in_databricks):
    if not is_in_databricks:
        pytest.skip("OSS GenAI evaluator doesn't support passing traces yet")

    actual_call_args_list = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations, trace) -> float:
        actual_call_args_list.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "expectations": expectations,
            }
        )
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
        actual_call_args_list.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "trace": trace,
            }
        )
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
        match="Custom scorer registration.*not supported outside of Databricks tracking",
    ):
        test_custom_scorer.register(experiment_id=experiment_id, name="test_scorer")

    mlflow.delete_experiment(experiment_id)


def test_custom_scorer_loading_blocked_for_non_databricks_uri():
    serialized = SerializedScorer(
        name="malicious_scorer",
        call_source="import os\nos.system('echo hacked')\nreturn True",
        call_signature="(outputs)",
        original_func_name="malicious_scorer",
    )

    with pytest.raises(
        mlflow.exceptions.MlflowException, match="Loading custom scorer.*not supported"
    ):
        Scorer._reconstruct_decorator_scorer(serialized)


def test_custom_scorer_loading_blocked_for_databricks_remote_access():
    serialized = SerializedScorer(
        name="malicious_scorer",
        call_source="import os\nos.system('echo hacked')\nreturn True",
        call_signature="(outputs)",
        original_func_name="malicious_scorer",
    )

    with (
        patch("mlflow.genai.scorers.base.is_in_databricks_runtime", return_value=False),
        patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True),
    ):
        with pytest.raises(
            mlflow.exceptions.MlflowException, match="via remote access is not supported"
        ):
            Scorer._reconstruct_decorator_scorer(serialized)


def test_custom_scorer_error_message_renders_code_snippet_legibly():
    serialized = SerializedScorer(
        name="complex_scorer",
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
        mlflow.exceptions.MlflowException, match="is not supported outside of"
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
