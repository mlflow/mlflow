from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pytest

import mlflow
from mlflow.genai.evaluation.utils import _convert_scorer_to_legacy_metric
from mlflow.genai.scorers import Scorer, scorer
from mlflow.models import Model
from mlflow.models.evaluation.base import _get_model_from_function
from mlflow.pyfunc import PyFuncModel


def mock_init_auth(config_instance):
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
