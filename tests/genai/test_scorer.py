import importlib
from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pytest
from packaging.version import Version

import mlflow
from mlflow.entities import Assessment
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai import Scorer, scorer
from mlflow.genai.evaluation.utils import _convert_scorer_to_legacy_metric
from mlflow.models import Model
from mlflow.models.evaluation.base import _get_model_from_function
from mlflow.pyfunc import PyFuncModel

if importlib.util.find_spec("databricks.agents") is None:
    pytest.skip(reason="databricks-agents is not installed", allow_module_level=True)

agent_sdk_version = Version(importlib.import_module("databricks.agents").__version__)


def mock_init_auth(config_instance):
    config_instance.host = "https://databricks.com/"
    config_instance._header_factory = lambda: {}


def always_yes(inputs, outputs, expectations, trace):
    return "yes"


class AlwaysYesScorer(Scorer):
    def __call__(self, inputs, outputs, expectations, trace):
        return "yes"


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "request": [
                "What is the difference between reduceByKey and groupByKey in Spark?",
                {
                    "messages": [
                        {"role": "user", "content": "How can you minimize data shuffling in Spark?"}
                    ]
                },
            ],
            "response": [
                {"choices": [{"message": {"content": "actual response for first question"}}]},
                {"choices": [{"message": {"content": "actual response for second question"}}]},
            ],
            "expected_response": [
                "expected response for first question",
                "expected response for second question",
            ],
        }
    )


@pytest.fixture
def sample_new_data():
    # sample data for new eval dataset format for mlflow.genai.evaluate()
    return pd.DataFrame(
        {
            "inputs": [
                "What is the difference between reduceByKey and groupByKey in Spark?",
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
                "expected response for first question",
                "expected response for second question",
            ],
        }
    )


@pytest.mark.parametrize("dummy_scorer", [AlwaysYesScorer(name="always_yes"), scorer(always_yes)])
def test_scorer_existence_in_metrics(sample_data, dummy_scorer):
    legacy_metric = _convert_scorer_to_legacy_metric(dummy_scorer)

    # patch is needed for databricks-agent tests
    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        result = mlflow.evaluate(
            data=sample_data,
            model_type="databricks-agent",
            extra_metrics=[legacy_metric],
        )

    assert any("always_yes" in metric for metric in result.metrics.keys())


@pytest.mark.parametrize(
    "dummy_scorer", [AlwaysYesScorer(name="always_no"), scorer(name="always_no")(always_yes)]
)
def test_scorer_name_works(sample_data, dummy_scorer):
    _SCORER_NAME = "always_no"

    legacy_metric = _convert_scorer_to_legacy_metric(dummy_scorer)

    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        result = mlflow.evaluate(
            data=sample_data,
            model_type="databricks-agent",
            extra_metrics=[legacy_metric],
        )

    assert any(_SCORER_NAME in metric for metric in result.metrics.keys())


def test_scorer_is_called_with_correct_arguments(sample_data):
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

    legacy_metric = _convert_scorer_to_legacy_metric(dummy_scorer)

    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        mlflow.evaluate(
            data=sample_data,
            model_type="databricks-agent",
            extra_metrics=[legacy_metric],
        )

    assert len(actual_call_args_list) == len(sample_data)

    # Prepare expected arguments, keyed by expected_response for matching
    sample_data_set = defaultdict(set)
    for i in range(len(sample_data)):
        sample_data_set["inputs"].add(str(sample_data["request"][i]))
        sample_data_set["outputs"].add(str(sample_data["response"][i]))
        sample_data_set["expectations"].add(str(sample_data["expected_response"][i]))

    for actual_args in actual_call_args_list:
        # do any check since actual passed input could be reformatted and larger than sample input
        assert any(
            sample_data_input in str(actual_args["inputs"])
            for sample_data_input in sample_data_set["inputs"]
        )
        assert str(actual_args["outputs"]) in sample_data_set["outputs"]
        assert str(actual_args["expectations"]) in sample_data_set["expectations"]


def test_trace_passed_correctly():
    @mlflow.trace
    def predict_fn(inputs):
        return "output: " + str(inputs)

    actual_call_args_list = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations, trace):
        actual_call_args_list.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "trace": trace,
            }
        )
        return 0.0

    legacy_metric = _convert_scorer_to_legacy_metric(dummy_scorer)

    pyfunc_model = PyFuncModel(
        model_meta=Model(),
        model_impl=_get_model_from_function(predict_fn),
    )

    data = pd.DataFrame({"request": ["input1", "input2", "input3"]})

    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        mlflow.evaluate(
            model=pyfunc_model,
            data=data,
            extra_metrics=[legacy_metric],
            model_type="databricks-agent",
        )

    assert len(actual_call_args_list) == len(data)
    for actual_args in actual_call_args_list:
        assert actual_args["trace"] is not None
        trace = actual_args["trace"]
        # check if the input is present in the trace
        assert any(str(data["request"][i]) in str(trace.data.request) for i in range(len(data)))
        # check if predict_fn was run by making output it starts with "output:"
        assert "output:" in str(trace.data.response)[:10]


@pytest.mark.parametrize(
    "scorer_return",
    [
        "yes",
        42,
        42.0,
        Assessment(
            name="big_question",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="123"),
            feedback=Feedback(value=42),
            rationale="It's the answer to everything",
        ),
        [
            Assessment(
                name="big_question",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id="judge_1"
                ),
                feedback=Feedback(value=42),
                rationale="It's the answer to everything",
            ),
            Assessment(
                name="small_question",
                feedback=Feedback(value=1),
                rationale="Not sure, just a guess",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id="judge_2"
                ),
            ),
        ],
    ],
)
def test_scorer_on_genai_evaluate(sample_new_data, scorer_return):
    # Skip if `databricks-agents` SDK is not 1.x. It doesn't
    # support the `mlflow.entities.Assessment` type.
    is_return_assessment = isinstance(scorer_return, Assessment) or (
        isinstance(scorer_return, list) and isinstance(scorer_return[0], Assessment)
    )
    if is_return_assessment and agent_sdk_version.major < 1:
        pytest.skip("Skipping test for assessment return type")

    @scorer
    def dummy_scorer(inputs, outputs):
        return scorer_return

    with patch("mlflow.get_tracking_uri", return_value="databricks"):
        with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
            results = mlflow.genai.evaluate(
                data=sample_new_data,
                scorers=[dummy_scorer],
            )

            assert any("metric/dummy_scorer" in metric for metric in results.metrics.keys())

            dummy_scorer_cols = [
                col for col in results.result_df.keys() if "dummy_scorer" in col and "value" in col
            ]
            dummy_scorer_values = set()
            for col in dummy_scorer_cols:
                for _val in results.result_df[col]:
                    dummy_scorer_values.add(_val)

            scorer_return_values = set()
            if isinstance(scorer_return, list):
                for _assessment in scorer_return:
                    scorer_return_values.add(_assessment.feedback.value)
            elif isinstance(scorer_return, Assessment):
                scorer_return_values.add(scorer_return.feedback.value)
            else:
                scorer_return_values.add(scorer_return)

            assert dummy_scorer_values == scorer_return_values
