import pytest

from mlflow.genai.scorers import Scorer, scorer
from mlflow.genai.scorers.builtin_scorers import Correctness, Safety, get_all_scorers
from mlflow.pyfunc.model import PythonModel
from mlflow.telemetry.parser import (
    AutologParser,
    GenaiEvaluateParser,
    LogModelParser,
)
from mlflow.telemetry.schemas import AutologParams, GenaiEvaluateParams, LogModelParams, ModelType


@pytest.mark.parametrize(
    ("func_name", "arguments", "expected_params"),
    [
        (
            "mlflow.langchain.autolog",
            {"disable": True},
            AutologParams(flavor="langchain", disable=True, log_traces=False, log_models=False),
        ),
        (
            "mlflow.langchain.autolog",
            {"disable": False, "log_traces": True},
            AutologParams(flavor="langchain", disable=False, log_traces=True, log_models=False),
        ),
        (
            "mlflow.sklearn.autolog",
            {"log_models": True},
            AutologParams(flavor="sklearn", disable=False, log_traces=False, log_models=True),
        ),
    ],
)
def test_autolog_parser(func_name, arguments, expected_params):
    assert AutologParser.extract_params(func_name, arguments) == expected_params


@pytest.mark.parametrize(
    ("func_name", "arguments", "expected_params"),
    [
        (
            "mlflow.langchain.log_model",
            {"model": "model_path", "extra_pip_requirements": ["pandas", "numpy"]},
            LogModelParams(
                flavor="langchain",
                model=ModelType.MODEL_PATH.value,
                is_pip_requirements_set=False,
                is_extra_pip_requirements_set=True,
                is_code_paths_set=False,
                is_params_set=False,
                is_metadata_set=False,
            ),
        ),
        (
            "mlflow.pyfunc.log_model",
            {"model": lambda x: x, "pip_requirements": ["pandas"], "code_paths": ["/path/to/code"]},
            LogModelParams(
                flavor="pyfunc",
                model=ModelType.PYTHON_FUNCTION.value,
                is_pip_requirements_set=True,
                is_extra_pip_requirements_set=False,
                is_code_paths_set=True,
                is_params_set=False,
                is_metadata_set=False,
            ),
        ),
        (
            "mlflow.pyfunc.log_model",
            {"model": PythonModel(), "metadata": {"key": "value"}},
            LogModelParams(
                flavor="pyfunc",
                model=ModelType.PYTHON_MODEL.value,
                is_pip_requirements_set=False,
                is_extra_pip_requirements_set=False,
                is_code_paths_set=False,
                is_params_set=False,
                is_metadata_set=True,
            ),
        ),
        (
            "mlflow.sklearn.log_model",
            {"model": object(), "params": {"key": "value"}},
            LogModelParams(
                flavor="sklearn",
                model=ModelType.MODEL_OBJECT.value,
                is_pip_requirements_set=False,
                is_extra_pip_requirements_set=False,
                is_code_paths_set=False,
                is_params_set=True,
                is_metadata_set=False,
            ),
        ),
    ],
)
def test_log_model_parser(func_name, arguments, expected_params):
    assert LogModelParser.extract_params(func_name, arguments) == expected_params


@scorer
def not_empty(outputs) -> bool:
    return outputs != ""


def test_sanitize_scorer_name():
    parser = GenaiEvaluateParser()
    built_in_scorers = get_all_scorers()
    for built_in_scorer in built_in_scorers:
        assert parser.sanitize_scorer_name(built_in_scorer) == built_in_scorer.name

    custom_scorer = Scorer(name="test_scorer")
    assert parser.sanitize_scorer_name(custom_scorer) == "CustomScorer"
    assert parser.sanitize_scorer_name(not_empty) == "CustomScorer"


@pytest.mark.parametrize(
    ("func_name", "arguments", "expected_params"),
    [
        (
            "mlflow.genai.evaluate",
            {"data": ["a", "b", "c"], "scorers": [Safety(), not_empty]},
            GenaiEvaluateParams(scorers=["safety", "CustomScorer"], is_predict_fn_set=False),
        ),
        (
            "mlflow.genai.evaluate",
            {"scorers": [Safety(), Correctness()], "predict_fn": lambda x: x},
            GenaiEvaluateParams(scorers=["safety", "correctness"], is_predict_fn_set=True),
        ),
        (
            "mlflow.genai.evaluate",
            {
                "data": ["a", "b", "c"],
                "scorers": [Scorer(name="test_scorer"), not_empty],
                "predict_fn": lambda x: x,
            },
            GenaiEvaluateParams(scorers=["CustomScorer", "CustomScorer"], is_predict_fn_set=True),
        ),
    ],
)
def test_genai_evaluate_parser(func_name, arguments, expected_params):
    assert GenaiEvaluateParser.extract_params(func_name, arguments) == expected_params
