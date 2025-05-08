import importlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai import scorer
from mlflow.genai.evaluation.utils import _convert_to_legacy_eval_set
from mlflow.genai.scorers.builtin_scorers import safety

if importlib.util.find_spec("databricks.agents") is None:
    pytest.skip(reason="databricks-agents is not installed", allow_module_level=True)


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


@pytest.fixture(autouse=True)
def spoof_tracking_uri_check():
    # NB: The mlflow.genai.evaluate() API is only runnable when the tracking URI is set
    # to Databricks. However, we cannot test against real Databricks server in CI, so
    # we spoof the check by patching the is_databricks_uri() function.
    with patch("mlflow.genai.evaluation.base.is_databricks_uri", return_value=True):
        yield


@pytest.fixture
def sample_dict_data_single():
    return [
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "actual response for first question",
            "expectations": "expected response for first question",
        },
    ]


@pytest.fixture
def sample_dict_data_multiple():
    return [
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "actual response for first question",
            "expectations": "expected response for first question",
            # Additional columns required by the judges
            "retrieved_context": [
                {
                    "content": "doc content 1",
                    "doc_uri": "doc_uri_2_1",
                },
                {
                    "content": "doc content 2.",
                    "doc_uri": "doc_uri_6_extra",
                },
            ],
        },
        {
            "inputs": {"question": "How can you minimize data shuffling in Spark?"},
            "outputs": "actual response for second question",
            "expectations": "expected response for second question",
            "retrieved_context": [],
        },
    ]


@pytest.fixture
def sample_pd_data(sample_dict_data_multiple):
    """Returns a pandas DataFrame with sample data"""
    return pd.DataFrame(sample_dict_data_multiple)


@pytest.fixture
def sample_spark_data(sample_pd_data, spark):
    """Convert pandas DataFrame to PySpark DataFrame"""
    return spark.createDataFrame(sample_pd_data)


@pytest.mark.parametrize(
    "data_fixture",
    ["sample_dict_data_single", "sample_dict_data_multiple", "sample_pd_data", "sample_spark_data"],
)
def test_convert_to_legacy_eval_set_has_no_errors(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        transformed_data = _convert_to_legacy_eval_set(sample_data)

    assert "request" in transformed_data.columns
    assert "response" in transformed_data.columns
    assert "expected_response" in transformed_data.columns


def test_convert_to_legacy_eval_set_invalid_inputs():
    with pytest.raises(MlflowException, match="The 'inputs' column must be a dictionary"):
        _convert_to_legacy_eval_set([{"inputs": "not a dict"}])


@pytest.mark.parametrize(
    "data_fixture",
    ["sample_dict_data_single", "sample_dict_data_multiple", "sample_pd_data", "sample_spark_data"],
)
def test_scorer_receives_correct_data(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    received_args = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations):
        received_args.append((inputs["question"], outputs, expectations))
        return 0

    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        mlflow.genai.evaluate(
            data=sample_data,
            scorers=[dummy_scorer],
        )

        all_inputs, all_outputs, all_expectations = zip(*received_args)
        expected_inputs = [
            "What is Spark?",
            "How can you minimize data shuffling in Spark?",
        ][: len(sample_data)]
        expected_outputs = [
            "actual response for first question",
            "actual response for second question",
        ][: len(sample_data)]
        expected_expectations = [
            "expected response for first question",
            "expected response for second question",
        ][: len(sample_data)]

        assert set(all_inputs) == set(expected_inputs)
        assert set(all_outputs) == set(expected_outputs)
        assert set(all_expectations) == set(expected_expectations)


def test_input_is_required_if_trace_is_not_provided():
    with patch("mlflow.evaluate") as mock_evaluate:
        with pytest.raises(MlflowException, match="inputs.*required"):
            mlflow.genai.evaluate(
                data=pd.DataFrame({"outputs": ["Paris"]}),
                scorers=[safety()],
            )

        mock_evaluate.assert_not_called()

        mlflow.genai.evaluate(
            data=pd.DataFrame({"inputs": ["What is the capital of France?"], "outputs": ["Paris"]}),
            scorers=[safety()],
        )
        mock_evaluate.assert_called_once()


def test_input_is_optional_if_trace_is_provided():
    with patch("mlflow.evaluate") as mock_evaluate:
        mlflow.genai.evaluate(
            data=pd.DataFrame({"outputs": ["Paris"], "trace": [MagicMock()]}),
            scorers=[safety()],
        )

        mock_evaluate.assert_called_once()


@pytest.mark.parametrize(
    "data_fixture",
    ["sample_dict_data_single", "sample_dict_data_multiple", "sample_pd_data", "sample_spark_data"],
)
def test_predict_fn_receives_correct_data(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    received_args = []

    def predict_fn(question: str):
        received_args.append(question)
        return question

    with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        mlflow.genai.evaluate(
            predict_fn=predict_fn,
            data=sample_data,
            scorers=[safety()],
        )
        received_args.pop(0)  # Remove the one-time prediction to check if a model is traced
        assert len(received_args) == len(sample_data)
        expected_contents = ["What is Spark?", "How can you minimize data shuffling in Spark?"]
        # Using set because eval harness runs predict_fn in parallel
        assert set(received_args) == set(expected_contents[: len(sample_data)])
