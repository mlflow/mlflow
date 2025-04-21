from unittest.mock import patch

import pandas as pd
import pytest

import mlflow
from mlflow.genai.evaluation.utils import (
    _convert_scorer_to_legacy_metric,
    _convert_to_legacy_eval_set,
)
from mlflow.genai.scorers import scorer


def mock_init_auth(config_instance):
    config_instance.host = "https://databricks.com/"
    config_instance._header_factory = lambda: {}


@pytest.fixture(scope="module")
def spark():
    try:
        from pyspark.sql import SparkSession

        with SparkSession.builder.getOrCreate() as spark:
            yield spark
    except RuntimeError:
        pytest.skip("Can't create a Spark session")


@pytest.fixture
def sample_dict_data():
    return {
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


@pytest.fixture
def sample_pd_data(sample_dict_data):
    """Returns a pandas DataFrame with sample data"""
    return pd.DataFrame(sample_dict_data)


@pytest.fixture
def sample_dict_list_data(sample_pd_data):
    """Returns a pandas DataFrame with sample data"""
    return sample_pd_data.to_dict(orient="records")


@pytest.fixture
def sample_spark_data(sample_pd_data, spark):
    """Convert pandas DataFrame to PySpark DataFrame"""
    return spark.createDataFrame(sample_pd_data)


@pytest.mark.parametrize(
    "data_fixture",
    ["sample_dict_data", "sample_pd_data", "sample_spark_data", "sample_dict_list_data"],
)
def test_convert_to_legacy_eval_set_has_no_errors(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        transformed_data = _convert_to_legacy_eval_set(sample_data)

        assert "request" in transformed_data.columns
        assert "response" in transformed_data.columns
        assert "expected_response" in transformed_data.columns

        mlflow.evaluate(
            data=transformed_data,
            model_type="databricks-agent",
        )


@pytest.mark.parametrize(
    "data_fixture",
    ["sample_dict_data", "sample_pd_data", "sample_spark_data", "sample_dict_list_data"],
)
def test_scorer_receives_correct_data(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    received_args = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations, trace):
        received_args.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "expectations": expectations,
                "trace": trace,
            }
        )

    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        transformed_data = _convert_to_legacy_eval_set(sample_data)
        legacy_metric = _convert_scorer_to_legacy_metric(dummy_scorer)
        mlflow.evaluate(
            data=transformed_data,
            model_type="databricks-agent",
            extra_metrics=[legacy_metric],
        )

        expected_len = (
            len(sample_data["inputs"]) if isinstance(sample_data, dict) else len(sample_data)
        )
        assert len(received_args) == expected_len
        assert set(received_args[0].keys()) == set({"inputs", "outputs", "expectations", "trace"})

        all_inputs, all_outputs, all_expectations, all_traces = [], [], [], []
        for arg in received_args:
            all_inputs.append(arg["inputs"]["messages"][0]["content"])
            all_outputs.append(arg["outputs"]["choices"][0]["message"]["content"])
            all_expectations.append(arg["expectations"])
            all_traces.extend(arg["trace"])

        expected_inputs = [
            "What is the difference between reduceByKey and groupByKey in Spark?",
            "How can you minimize data shuffling in Spark?",
        ]
        expected_outputs = [
            "actual response for first question",
            "actual response for second question",
        ]
        expected_expectations = [
            "expected response for first question",
            "expected response for second question",
        ]

        assert set(all_inputs) == set(expected_inputs)
        assert set(all_outputs) == set(expected_outputs)
        assert set(all_expectations) == set(expected_expectations)
        for trace in all_traces:
            assert any(
                expected_input in trace.info.request_preview for expected_input in expected_inputs
            )
            assert any(
                expected_output in trace.info.response_preview
                for expected_output in expected_outputs
            )


@pytest.mark.parametrize(
    "data_fixture",
    ["sample_dict_data", "sample_pd_data", "sample_spark_data", "sample_dict_list_data"],
)
def test_predict_fn_receives_correct_data(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    received_args = []

    def predict_fn(inputs):
        received_args.append(inputs)
        return inputs

    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        transformed_data = _convert_to_legacy_eval_set(sample_data)
        mlflow.evaluate(
            model=predict_fn,
            data=transformed_data,
            model_type="databricks-agent",
        )
        expected_len = (
            len(sample_data["inputs"]) if isinstance(sample_data, dict) else len(sample_data)
        )
        assert len(received_args) == expected_len
        assert set(
            {received_args[0]["messages"][0]["content"], received_args[1]["messages"][0]["content"]}
        ) == set(
            {
                "What is the difference between reduceByKey and groupByKey in Spark?",
                "How can you minimize data shuffling in Spark?",
            }
        )
