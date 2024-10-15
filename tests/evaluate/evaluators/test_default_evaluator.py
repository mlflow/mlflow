from __future__ import annotations

import os
from typing import Dict, List
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
    MetricValue,
    flesch_kincaid_grade_level,
    make_metric,
    toxicity,
)
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.genai_metric import (
    _GENAI_CUSTOM_METRICS_FILE_NAME,
    make_genai_metric_from_prompt,
    retrieve_custom_metrics,
)
from mlflow.metrics.genai.metric_definitions import answer_similarity
from mlflow.models.evaluation.evaluators.default import _extract_output_and_other_columns


def language_model(inputs: list[str]) -> list[str]:
    return inputs


def validate_question_answering_logged_data(
    logged_data, with_targets=True, predictions_name="outputs"
):
    columns = {
        "question",
        predictions_name,
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "token_count",
    }
    if with_targets:
        columns.update({"answer"})

    assert set(logged_data.columns.tolist()) == columns

    assert logged_data["question"].tolist() == ["words random", "This is a sentence."]
    assert logged_data[predictions_name].tolist() == ["words random", "This is a sentence."]
    assert logged_data["toxicity/v1/score"][0] < 0.5
    assert logged_data["toxicity/v1/score"][1] < 0.5
    assert all(
        isinstance(grade, float) for grade in logged_data["flesch_kincaid_grade_level/v1/score"]
    )
    assert all(isinstance(grade, float) for grade in logged_data["ari_grade_level/v1/score"])
    assert all(isinstance(grade, int) for grade in logged_data["token_count"])

    if with_targets:
        assert logged_data["answer"].tolist() == ["words random", "This is a sentence."]


def test_missing_args_raises_exception():
    def dummy_fn1(param_1, param_2, targets, metrics):
        pass

    def dummy_fn2(param_3, param_4, builtin_metrics):
        pass

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"question": ["a", "b"], "answer": ["a", "b"]})

    metric_1 = make_metric(name="metric_1", eval_fn=dummy_fn1, greater_is_better=True)
    metric_2 = make_metric(name="metric_2", eval_fn=dummy_fn2, greater_is_better=True)

    error_message = (
        r"Error: Metric calculation failed for the following metrics:\n"
        r"Metric 'metric_1' requires the following:\n"
        r"- the 'targets' parameter needs to be specified\n"
        r"- missing columns \['param_1', 'param_2'\] need to be defined or mapped\n"
        r"Metric 'metric_2' requires the following:\n"
        r"- missing columns \['param_3', 'builtin_metrics'\] need to be defined or mapped\n\n"
        r"Below are the existing column names for the input/output data:\n"
        r"Input Columns: \['question', 'answer'\]\n"
        r"Output Columns: \['predictions'\]\n\n"
    )

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                model_info.model_uri,
                data,
                evaluators="default",
                model_type="question-answering",
                extra_metrics=[metric_1, metric_2],
                evaluator_config={"col_mapping": {"param_4": "question"}},
            )


def test_evaluate_question_answering_with_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "answer": ["words random", "This is a sentence."],
            }
        )
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="answer",
            model_type="question-answering",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_question_answering_logged_data(logged_data)
    assert set(results.metrics.keys()) == set(
        get_question_answering_metrics_keys(with_targets=True)
    )
    assert results.metrics["exact_match/v1"] == 1.0


def test_evaluate_question_answering_on_static_dataset_with_targets():
    with mlflow.start_run() as run:
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "answer": ["words random", "This is a sentence."],
                "pred": ["words random", "This is a sentence."],
            }
        )
        results = mlflow.evaluate(
            data=data,
            targets="answer",
            predictions="pred",
            model_type="question-answering",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_question_answering_logged_data(logged_data, predictions_name="pred")
    assert set(results.metrics.keys()) == {
        "toxicity/v1/variance",
        "toxicity/v1/ratio",
        "toxicity/v1/mean",
        "flesch_kincaid_grade_level/v1/variance",
        "ari_grade_level/v1/p90",
        "flesch_kincaid_grade_level/v1/p90",
        "flesch_kincaid_grade_level/v1/mean",
        "ari_grade_level/v1/mean",
        "ari_grade_level/v1/variance",
        "exact_match/v1",
        "toxicity/v1/p90",
    }
    assert results.metrics["exact_match/v1"] == 1.0
    assert results.metrics["toxicity/v1/ratio"] == 0.0


def question_classifier(inputs):
    return inputs["question"].map({"a": 0, "b": 1})


def test_evaluate_question_answering_with_numerical_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=question_classifier,
            input_example=pd.DataFrame({"question": ["a", "b"]}),
        )
        data = pd.DataFrame({"question": ["a", "b"], "answer": [0, 1]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="answer",
            model_type="question-answering",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    pd.testing.assert_frame_equal(
        logged_data.drop("token_count", axis=1),
        data.assign(outputs=[0, 1]),
    )
    assert results.metrics == {"exact_match/v1": 1.0}


def test_evaluate_question_answering_without_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"question": ["words random", "This is a sentence."]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="question-answering",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_question_answering_logged_data(logged_data, False)
    assert set(results.metrics.keys()) == set(
        get_question_answering_metrics_keys(with_targets=False)
    )


def validate_text_summarization_logged_data(logged_data, with_targets=True):
    columns = {
        "text",
        "outputs",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "token_count",
    }
    if with_targets:
        columns.update(
            {
                "summary",
                "rouge1/v1/score",
                "rouge2/v1/score",
                "rougeL/v1/score",
                "rougeLsum/v1/score",
            }
        )

    assert set(logged_data.columns.tolist()) == columns

    assert logged_data["text"].tolist() == ["a", "b"]
    assert logged_data["outputs"].tolist() == ["a", "b"]
    assert logged_data["toxicity/v1/score"][0] < 0.5
    assert logged_data["toxicity/v1/score"][1] < 0.5
    assert all(
        isinstance(grade, float) for grade in logged_data["flesch_kincaid_grade_level/v1/score"]
    )
    assert all(isinstance(grade, float) for grade in logged_data["ari_grade_level/v1/score"])
    assert all(isinstance(grade, int) for grade in logged_data["token_count"])

    if with_targets:
        assert logged_data["summary"].tolist() == ["a", "b"]
        assert logged_data["rouge1/v1/score"].tolist() == [1.0, 1.0]
        assert logged_data["rouge2/v1/score"].tolist() == [0.0, 0.0]
        assert logged_data["rougeL/v1/score"].tolist() == [1.0, 1.0]
        assert logged_data["rougeLsum/v1/score"].tolist() == [1.0, 1.0]


def get_text_metrics_keys():
    metric_names = ["toxicity", "flesch_kincaid_grade_level", "ari_grade_level"]
    standard_aggregations = ["mean", "variance", "p90"]
    version = "v1"

    metrics_keys = [
        f"{metric}/{version}/{agg}" for metric in metric_names for agg in standard_aggregations
    ]
    additional_aggregations = ["toxicity/v1/ratio"]
    return metrics_keys + additional_aggregations


def get_text_summarization_metrics_keys(with_targets=False):
    if with_targets:
        metric_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        standard_aggregations = ["mean", "variance", "p90"]
        version = "v1"

        metrics_keys = [
            f"{metric}/{version}/{agg}" for metric in metric_names for agg in standard_aggregations
        ]
    else:
        metrics_keys = []
    return get_text_metrics_keys() + metrics_keys


def get_question_answering_metrics_keys(with_targets=False):
    metrics_keys = ["exact_match/v1"] if with_targets else []
    return get_text_metrics_keys() + metrics_keys


def test_evaluate_text_summarization_with_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["a", "b"], "summary": ["a", "b"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="summary",
            model_type="text-summarization",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_text_summarization_logged_data(logged_data)

    metrics = results.metrics
    assert set(metrics.keys()) == set(get_text_summarization_metrics_keys(with_targets=True))


def test_evaluate_text_summarization_with_targets_no_type_hints():
    def another_language_model(x):
        x.rename(columns={"text": "outputs"}, inplace=True)
        return x

    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=another_language_model,
            input_example=pd.DataFrame({"text": ["a", "b"]}),
        )
        data = pd.DataFrame({"text": ["a", "b"], "summary": ["a", "b"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="summary",
            model_type="text-summarization",
            evaluators="default",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_text_summarization_logged_data(logged_data)

    metrics = results.metrics
    assert set(metrics.keys()) == set(get_text_summarization_metrics_keys(with_targets=True))


def test_evaluate_text_summarization_without_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["a", "b"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="text-summarization",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_text_summarization_logged_data(logged_data, with_targets=False)

    assert set(results.metrics.keys()) == set(
        get_text_summarization_metrics_keys(with_targets=False)
    )


def test_evaluate_text_summarization_fails_to_load_evaluate_metrics():
    from mlflow.metrics.metric_definitions import _cached_evaluate_load

    _cached_evaluate_load.cache_clear()

    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )

        data = pd.DataFrame({"text": ["a", "b"], "summary": ["a", "b"]})
        with mock.patch(
            "mlflow.metrics.metric_definitions._cached_evaluate_load",
            side_effect=ImportError("mocked error"),
        ) as mock_load:
            results = mlflow.evaluate(
                model_info.model_uri,
                data,
                targets="summary",
                model_type="text-summarization",
            )
            mock_load.assert_any_call("rouge")
            mock_load.assert_any_call("toxicity", module_type="measurement")

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "summary",
        "outputs",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "token_count",
    }
    assert logged_data["text"].tolist() == ["a", "b"]
    assert logged_data["summary"].tolist() == ["a", "b"]
    assert logged_data["outputs"].tolist() == ["a", "b"]


def test_evaluate_text_and_text_metrics():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["sentence not", "All women are bad."]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="text",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "outputs",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "token_count",
    }
    assert logged_data["text"].tolist() == ["sentence not", "All women are bad."]
    assert logged_data["outputs"].tolist() == ["sentence not", "All women are bad."]
    # Hateful sentiments should be marked as toxic
    assert logged_data["toxicity/v1/score"][0] < 0.5
    assert logged_data["toxicity/v1/score"][1] > 0.5
    # Simple sentences should have a low grade level.
    assert logged_data["flesch_kincaid_grade_level/v1/score"][1] < 4
    assert logged_data["ari_grade_level/v1/score"][1] < 4
    assert set(results.metrics.keys()) == set(get_text_metrics_keys())


def very_toxic(predictions, targets=None, metrics=None):
    new_scores = [1.0 if score > 0.9 else 0.0 for score in metrics["toxicity/v1"].scores]
    return MetricValue(
        scores=new_scores,
        justifications=["toxic" if score == 1.0 else "not toxic" for score in new_scores],
        aggregate_results={"ratio": sum(new_scores) / len(new_scores)},
    )


def per_row_metric(predictions, targets=None, metrics=None):
    return MetricValue(scores=[1] * len(predictions))


def test_evaluate_text_custom_metrics():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["a", "b"], "target": ["a", "b"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="target",
            model_type="text",
            extra_metrics=[
                make_metric(eval_fn=very_toxic, greater_is_better=True, version="v2"),
                make_metric(eval_fn=per_row_metric, greater_is_better=False, name="no_version"),
            ],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)

    assert "very_toxic/v2/score" in logged_data.columns.tolist()
    assert "very_toxic/v2/justification" in logged_data.columns.tolist()
    assert all(isinstance(score, float) for score in logged_data["very_toxic/v2/score"])
    assert all(
        isinstance(justification, str)
        for justification in logged_data["very_toxic/v2/justification"]
    )
    assert "very_toxic/v2/ratio" in set(results.metrics.keys())
    assert "no_version/score" in logged_data.columns.tolist()


@pytest.mark.parametrize("metric_prefix", ["train_", None])
def test_eval_results_table_json_can_be_prefixed_with_metric_prefix(metric_prefix):
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["a", "b"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="text",
            evaluators="default",
            evaluator_config={
                "metric_prefix": metric_prefix,
            },
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]

    if metric_prefix is None:
        metric_prefix = ""

    assert f"{metric_prefix}eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "outputs",
        f"{metric_prefix}toxicity/v1/score",
        f"{metric_prefix}flesch_kincaid_grade_level/v1/score",
        f"{metric_prefix}ari_grade_level/v1/score",
        f"{metric_prefix}token_count",
    }


def test_extracting_output_and_other_columns():
    data_dict = {
        "text": ["text_a", "text_b"],
        "target": ["target_a", "target_b"],
        "other": ["other_a", "other_b"],
    }
    data_df = pd.DataFrame(data_dict)
    data_list_dict = [
        {
            "text": "text_a",
            "target": "target_a",
            "other": "other_a",
        },
        {
            "text": "text_b",
            "target": "target_b",
            "other": "other_b",
        },
    ]
    data_list = (["data_a", "data_b"],)
    data_dict_text = {
        "text": ["text_a", "text_b"],
    }

    output1, other1, prediction_col1 = _extract_output_and_other_columns(data_dict, "target")
    output2, other2, prediction_col2 = _extract_output_and_other_columns(data_df, "target")
    output3, other3, prediction_col3 = _extract_output_and_other_columns(data_list_dict, "target")
    output4, other4, prediction_col4 = _extract_output_and_other_columns(data_list, None)
    output5, other5, prediction_col5 = _extract_output_and_other_columns(pd.Series(data_list), None)
    output6, other6, prediction_col6 = _extract_output_and_other_columns(data_dict_text, None)
    output7, other7, prediction_col7 = _extract_output_and_other_columns(
        pd.DataFrame(data_dict_text), None
    )

    assert output1.equals(data_df["target"])
    assert other1.equals(data_df.drop(columns=["target"]))
    assert prediction_col1 == "target"
    assert output2.equals(data_df["target"])
    assert other2.equals(data_df.drop(columns=["target"]))
    assert prediction_col2 == "target"
    assert output3.equals(data_df["target"])
    assert other3.equals(data_df.drop(columns=["target"]))
    assert prediction_col3 == "target"
    assert output4 == data_list
    assert other4 is None
    assert prediction_col4 is None
    assert output5.equals(pd.Series(data_list))
    assert other5 is None
    assert prediction_col5 is None
    assert output6.equals(pd.Series(data_dict_text["text"]))
    assert other6 is None
    assert prediction_col6 == "text"
    assert output7.equals(pd.Series(data_dict_text["text"]))
    assert other7 is None
    assert prediction_col7 == "text"


def language_model_with_context(inputs: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "context": f"context_{input}",
            "output": input,
        }
        for input in inputs
    ]


def test_constructing_eval_df_for_custom_metrics():
    test_eval_df_value = pd.DataFrame(
        {
            "predictions": ["text_a", "text_b"],
            "targets": ["target_a", "target_b"],
            "inputs": ["text_a", "text_b"],
            "truth": ["truth_a", "truth_b"],
            "context": ["context_text_a", "context_text_b"],
        }
    )

    def example_custom_artifact(_, __, ___):
        return {"test_json_artifact": {"a": 2, "b": [1, 2]}}

    def test_eval_df(predictions, targets, metrics, inputs, truth, context):
        global eval_df_value
        eval_df_value = pd.DataFrame(
            {
                "predictions": predictions,
                "targets": targets,
                "inputs": inputs,
                "truth": truth,
                "context": context,
            }
        )
        return predictions.eq(targets).mean()

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=language_model_with_context,
            input_example=["a", "b"],
        )
        data = pd.DataFrame(
            {
                "text": ["text_a", "text_b"],
                "truth": ["truth_a", "truth_b"],
                "targets": ["target_a", "target_b"],
            }
        )
        eval_results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="targets",
            predictions="output",
            model_type="text",
            extra_metrics=[make_metric(eval_fn=test_eval_df, greater_is_better=True)],
            custom_artifacts=[example_custom_artifact],
            evaluators="default",
            evaluator_config={"col_mapping": {"inputs": "text"}},
        )

    assert eval_df_value.equals(test_eval_df_value)
    assert len(eval_results.artifacts) == 2
    assert len(eval_results.tables) == 1
    assert eval_results.tables["eval_results_table"].columns.tolist() == [
        "text",
        "truth",
        "targets",
        "output",
        "context",
        "token_count",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
    ]


def test_evaluate_no_model_or_predictions_specified():
    data = pd.DataFrame(
        {
            "question": ["words random", "This is a sentence."],
            "truth": ["words random", "This is a sentence."],
        }
    )

    with pytest.raises(
        MlflowException,
        match=(
            "Either a model or set of predictions must be specified in order to use the"
            " default evaluator"
        ),
    ):
        mlflow.evaluate(
            data=data,
            targets="truth",
            model_type="question-answering",
            evaluators="default",
        )


def test_evaluate_no_model_and_predictions_specified_with_unsupported_data_type():
    X = np.random.random((5, 5))
    y = np.random.random(5)

    with pytest.raises(
        MlflowException,
        match="If predictions is specified, data must be one of the following types",
    ):
        mlflow.evaluate(
            data=X,
            targets=y,
            predictions="model_output",
            model_type="question-answering",
            evaluators="default",
        )


def test_evaluate_no_model_type():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        with pytest.raises(
            MlflowException,
            match="The extra_metrics argument must be specified model_type is None.",
        ):
            mlflow.evaluate(
                model_info.model_uri,
                data,
            )


def test_evaluate_no_model_type_with_builtin_metric():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            extra_metrics=[mlflow.metrics.toxicity()],
        )
        assert results.metrics.keys() == {
            "toxicity/v1/mean",
            "toxicity/v1/variance",
            "toxicity/v1/p90",
            "toxicity/v1/ratio",
        }
        assert len(results.tables) == 1
        assert results.tables["eval_results_table"].columns.tolist() == [
            "text",
            "outputs",
            "toxicity/v1/score",
        ]


def test_evaluate_no_model_type_with_custom_metric():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        from mlflow.metrics import make_metric
        from mlflow.metrics.base import standard_aggregations

        def word_count_eval(predictions, targets=None, metrics=None):
            scores = []
            for prediction in predictions:
                scores.append(len(prediction.split(" ")))
            return MetricValue(
                scores=scores,
                aggregate_results=standard_aggregations(scores),
            )

        word_count = make_metric(eval_fn=word_count_eval, greater_is_better=True, name="word_count")

        results = mlflow.evaluate(model_info.model_uri, data, extra_metrics=[word_count])
        assert results.metrics.keys() == {
            "word_count/mean",
            "word_count/variance",
            "word_count/p90",
        }
        assert results.metrics["word_count/mean"] == 3.0
        assert len(results.tables) == 1
        assert results.tables["eval_results_table"].columns.tolist() == [
            "text",
            "outputs",
            "word_count/score",
        ]


def multi_output_model(inputs):
    return pd.DataFrame(
        {
            "answer": ["words random", "This is a sentence."],
            "source": ["words random", "This is a sentence."],
        }
    )


def test_default_metrics_as_extra_metrics():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=multi_output_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "truth": ["words random", "This is a sentence."],
            }
        )
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="truth",
            predictions="answer",
            model_type="question-answering",
            extra_metrics=[
                mlflow.metrics.exact_match(),
            ],
            evaluators="default",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    assert "exact_match/v1" in results.metrics.keys()


def test_default_metrics_as_extra_metrics_static_dataset():
    with mlflow.start_run() as run:
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "truth": ["words random", "This is a sentence."],
                "answer": ["words random", "This is a sentence."],
                "source": ["words random", "This is a sentence."],
            }
        )
        results = mlflow.evaluate(
            data=data,
            targets="truth",
            predictions="answer",
            model_type="question-answering",
            extra_metrics=[
                mlflow.metrics.flesch_kincaid_grade_level(),
                mlflow.metrics.ari_grade_level(),
                mlflow.metrics.toxicity(),
                mlflow.metrics.exact_match(),
            ],
            evaluators="default",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    for metric in ["toxicity", "ari_grade_level", "flesch_kincaid_grade_level"]:
        for measure in ["mean", "p90", "variance"]:
            assert f"{metric}/v1/{measure}" in results.metrics.keys()
    assert "exact_match/v1" in results.metrics.keys()


def test_derived_metrics_basic_dependency_graph():
    def metric_1(predictions, targets, metrics):
        return MetricValue(
            scores=[0, 1],
            justifications=["first justification", "second justification"],
            aggregate_results={"aggregate": 0.5},
        )

    def metric_2(predictions, targets, metrics, metric_1):
        return MetricValue(
            scores=[score * 5 for score in metric_1.scores],
            justifications=[
                "metric_2: " + justification for justification in metric_1.justifications
            ],
            aggregate_results={
                **metric_1.aggregate_results,
                **metrics["toxicity/v1"].aggregate_results,
            },
        )

    def metric_3(predictions, targets, metric_1, metric_2):
        return MetricValue(
            scores=[score - 1 for score in metric_2.scores],
            justifications=metric_1.justifications,
            aggregate_results=metric_2.aggregate_results,
        )

    with mlflow.start_run():
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "truth": ["words random", "This is a sentence."],
                "answer": ["words random", "This is a sentence."],
            }
        )
        results = mlflow.evaluate(
            data=data,
            targets="truth",
            predictions="answer",
            model_type="text",
            extra_metrics=[
                make_metric(eval_fn=metric_1, greater_is_better=True, version="v1"),
                make_metric(eval_fn=metric_2, greater_is_better=True, version="v2"),
                make_metric(eval_fn=metric_3, greater_is_better=True),
            ],
            evaluators="default",
        )

    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "question",
        "truth",
        "answer",
        "token_count",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "metric_1/v1/score",
        "metric_2/v2/score",
        "metric_3/score",
        "metric_1/v1/justification",
        "metric_2/v2/justification",
        "metric_3/justification",
    }

    assert logged_data["metric_1/v1/score"].tolist() == [0, 1]
    assert logged_data["metric_2/v2/score"].tolist() == [0, 5]
    assert logged_data["metric_3/score"].tolist() == [-1, 4]
    assert logged_data["metric_1/v1/justification"].tolist() == [
        "first justification",
        "second justification",
    ]
    assert logged_data["metric_2/v2/justification"].tolist() == [
        "metric_2: first justification",
        "metric_2: second justification",
    ]
    assert logged_data["metric_3/justification"].tolist() == [
        "first justification",
        "second justification",
    ]

    assert results.metrics["metric_1/v1/aggregate"] == 0.5
    assert results.metrics["metric_2/v2/aggregate"] == 0.5
    assert results.metrics["metric_3/aggregate"] == 0.5
    assert "metric_2/v2/mean" in results.metrics.keys()
    assert "metric_2/v2/variance" in results.metrics.keys()
    assert "metric_2/v2/p90" in results.metrics.keys()
    assert "metric_3/mean" in results.metrics.keys()
    assert "metric_3/variance" in results.metrics.keys()
    assert "metric_3/p90" in results.metrics.keys()


def test_derived_metrics_complicated_dependency_graph():
    def metric_1(predictions, targets, metric_2, metric_3, metric_6):
        assert metric_2.scores == [2, 3]
        assert metric_3.scores == [3, 4]
        assert metric_6.scores == [6, 7]
        return MetricValue(scores=[1, 2])

    def metric_2(predictions, targets):
        return MetricValue(scores=[2, 3])

    def metric_3(predictions, targets, metric_4, metric_5):
        assert metric_4.scores == [4, 5]
        assert metric_5.scores == [5, 6]
        return MetricValue(scores=[3, 4])

    def metric_4(predictions, targets, metric_6):
        assert metric_6.scores == [6, 7]
        return MetricValue(scores=[4, 5])

    def metric_5(predictions, targets, metric_4, metric_6):
        assert metric_4.scores == [4, 5]
        assert metric_6.scores == [6, 7]
        return MetricValue(scores=[5, 6])

    def metric_6(predictions, targets, metric_2):
        assert metric_2.scores == [2, 3]
        return MetricValue(scores=[6, 7])

    data = pd.DataFrame(
        {
            "question": ["words random", "This is a sentence."],
            "truth": ["words random", "This is a sentence."],
            "answer": ["words random", "This is a sentence."],
        }
    )

    with mlflow.start_run():
        results = mlflow.evaluate(
            data=data,
            predictions="answer",
            targets="truth",
            extra_metrics=[
                make_metric(eval_fn=metric_1, greater_is_better=True, version="v1"),
                make_metric(eval_fn=metric_2, greater_is_better=True, version="v2"),
                make_metric(eval_fn=metric_3, greater_is_better=True),
                make_metric(eval_fn=metric_4, greater_is_better=True),
                make_metric(eval_fn=metric_5, greater_is_better=True, version="v1"),
                make_metric(eval_fn=metric_6, greater_is_better=True, version="v3"),
            ],
            evaluators="default",
        )

    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "question",
        "truth",
        "answer",
        "metric_1/v1/score",
        "metric_2/v2/score",
        "metric_3/score",
        "metric_4/score",
        "metric_5/v1/score",
        "metric_6/v3/score",
    }

    assert logged_data["metric_1/v1/score"].tolist() == [1, 2]
    assert logged_data["metric_2/v2/score"].tolist() == [2, 3]
    assert logged_data["metric_3/score"].tolist() == [3, 4]
    assert logged_data["metric_4/score"].tolist() == [4, 5]
    assert logged_data["metric_5/v1/score"].tolist() == [5, 6]
    assert logged_data["metric_6/v3/score"].tolist() == [6, 7]

    def metric_7(predictions, targets, metric_8, metric_9):
        return MetricValue(scores=[7, 8])

    def metric_8(predictions, targets, metric_11):
        return MetricValue(scores=[8, 9])

    def metric_9(predictions, targets):
        return MetricValue(scores=[9, 10])

    def metric_10(predictions, targets, metric_9):
        return MetricValue(scores=[10, 11])

    def metric_11(predictions, targets, metric_7, metric_10):
        return MetricValue(scores=[11, 12])

    error_message = r"Error: Metric calculation failed for the following metrics:\n"

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                data=data,
                predictions="answer",
                targets="truth",
                model_type="text",
                extra_metrics=[
                    make_metric(eval_fn=metric_7, greater_is_better=True),
                    make_metric(eval_fn=metric_8, greater_is_better=True),
                    make_metric(eval_fn=metric_9, greater_is_better=True),
                    make_metric(eval_fn=metric_10, greater_is_better=True),
                    make_metric(eval_fn=metric_11, greater_is_better=True),
                ],
                evaluators="default",
            )


def test_derived_metrics_circular_dependencies_raises_exception():
    def metric_1(predictions, targets, metric_2):
        return 0

    def metric_2(predictions, targets, metric_3):
        return 0

    def metric_3(predictions, targets, metric_1):
        return 0

    error_message = r"Error: Metric calculation failed for the following metrics:\n"

    data = pd.DataFrame(
        {
            "question": ["words random", "This is a sentence."],
            "answer": ["words random", "This is a sentence."],
        }
    )

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                data=data,
                predictions="answer",
                model_type="text",
                extra_metrics=[
                    make_metric(eval_fn=metric_1, greater_is_better=True),
                    make_metric(eval_fn=metric_2, greater_is_better=True),
                    make_metric(eval_fn=metric_3, greater_is_better=True),
                ],
                evaluators="default",
            )


def test_derived_metrics_missing_dependencies_raises_exception():
    def metric_1(predictions, targets, metric_2):
        return 0

    def metric_2(predictions, targets, metric_3):
        return 0

    error_message = r"Error: Metric calculation failed for the following metrics:\n"

    data = pd.DataFrame(
        {
            "question": ["words random", "This is a sentence."],
            "answer": ["words random", "This is a sentence."],
        }
    )

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                data=data,
                predictions="answer",
                model_type="text",
                extra_metrics=[
                    make_metric(eval_fn=metric_1, greater_is_better=True),
                    make_metric(eval_fn=metric_2, greater_is_better=True),
                ],
                evaluators="default",
            )


def test_multi_output_model_error_handling():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=multi_output_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "truth": ["words random", "This is a sentence."],
            }
        )
        with pytest.raises(
            MlflowException,
            match="Output column name is not specified for the multi-output model.",
        ):
            mlflow.evaluate(
                model_info.model_uri,
                data,
                targets="truth",
                model_type="question-answering",
                extra_metrics=[
                    mlflow.metrics.flesch_kincaid_grade_level(),
                    mlflow.metrics.ari_grade_level(),
                    mlflow.metrics.toxicity(),
                    mlflow.metrics.exact_match(),
                ],
                evaluators="default",
            )


def test_invalid_extra_metrics():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        with pytest.raises(
            MlflowException,
            match="Please ensure that all extra metrics are instances of "
            "mlflow.metrics.EvaluationMetric.",
        ):
            mlflow.evaluate(
                model_info.model_uri,
                data,
                model_type="text",
                evaluators="default",
                extra_metrics=[mlflow.metrics.latency],
            )


def test_evaluate_with_latency():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["sentence not", "Hello world."]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="text",
            evaluators="default",
            extra_metrics=[mlflow.metrics.latency()],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "outputs",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "latency",
        "token_count",
    }
    assert all(isinstance(grade, float) for grade in logged_data["latency"])


def test_evaluate_with_latency_and_pd_series():
    with mlflow.start_run() as run:

        def pd_series_model(inputs: list[str]) -> pd.Series:
            return pd.Series(inputs)

        model_info = mlflow.pyfunc.log_model(
            "model", python_model=pd_series_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["input text", "random text"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="text",
            evaluators="default",
            extra_metrics=[mlflow.metrics.latency()],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "outputs",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "latency",
        "token_count",
    }


def test_evaluate_with_latency_static_dataset():
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=language_model, input_example=["a", "b"])
        data = pd.DataFrame(
            {
                "text": ["foo", "bar"],
                "model_output": ["FOO", "BAR"],
            }
        )
        results = mlflow.evaluate(
            data=data,
            model_type="text",
            evaluators="default",
            predictions="model_output",
            extra_metrics=[mlflow.metrics.latency()],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "outputs",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "latency",
        "token_count",
    }
    assert all(isinstance(grade, float) for grade in logged_data["latency"])
    assert all(grade == 0.0 for grade in logged_data["latency"])


properly_formatted_openai_response1 = """\
{
  "score": 3,
  "justification": "justification"
}"""


def test_evaluate_with_correctness():
    metric = mlflow.metrics.genai.make_genai_metric(
        name="correctness",
        definition=(
            "Correctness refers to how well the generated output matches "
            "or aligns with the reference or ground truth text that is considered "
            "accurate and appropriate for the given input. The ground truth serves as "
            "a benchmark against which the provided output is compared to determine the "
            "level of accuracy and fidelity."
        ),
        grading_prompt=(
            "Correctness: If the answer correctly answer the question, below "
            "are the details for different scores: "
            "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
            "the question or is completely contrary to the correct answer. "
            "- Score 1: the answer provides some relevance to the question and answer "
            "one aspect of the question correctly. "
            "- Score 2: the answer mostly answer the question but is missing or hallucinating "
            "on one critical aspect. "
            "- Score 4: the answer correctly answer the question and not missing any "
            "major aspect"
        ),
        examples=[],
        version="v1",
        model="openai:/gpt-4o-mini",
        grading_context_columns=["ground_truth"],
        parameters={"temperature": 0.0},
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ):
        with mlflow.start_run():
            eval_df = pd.DataFrame(
                {
                    "inputs": [
                        "What is MLflow?",
                        "What is Spark?",
                        "What is Python?",
                    ],
                    "ground_truth": [
                        "MLflow is an open-source platform",
                        "Apache Spark is an open-source, distributed computing system",
                        "Python is a high-level programming language",
                    ],
                    "prediction": [
                        "MLflow is an open-source platform",
                        "Apache Spark is an open-source, distributed computing system",
                        "Python is a high-level programming language",
                    ],
                }
            )
            results = mlflow.evaluate(
                data=eval_df,
                evaluators="default",
                targets="ground_truth",
                predictions="prediction",
                extra_metrics=[metric],
            )

            assert results.metrics == {
                "correctness/v1/mean": 3.0,
                "correctness/v1/variance": 0.0,
                "correctness/v1/p90": 3.0,
            }


def test_evaluate_custom_metrics_string_values():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            extra_metrics=[
                make_metric(
                    eval_fn=lambda predictions, metrics, eval_config: MetricValue(
                        aggregate_results={"eval_config_value_average": eval_config}
                    ),
                    name="cm",
                    greater_is_better=True,
                    long_name="custom_metric",
                )
            ],
            evaluators="default",
            evaluator_config={"eval_config": 3},
        )
        assert results.metrics["cm/eval_config_value_average"] == 3


def validate_retriever_logged_data(logged_data, k=3):
    columns = {
        "question",
        "retrieved_context",
        f"precision_at_{k}/score",
        f"recall_at_{k}/score",
        f"ndcg_at_{k}/score",
        "ground_truth",
    }

    assert set(logged_data.columns.tolist()) == columns

    assert logged_data["question"].tolist() == ["q1?", "q1?", "q1?"]
    assert logged_data["retrieved_context"].tolist() == [["doc1", "doc3", "doc2"]] * 3
    assert (logged_data[f"precision_at_{k}/score"] <= 1).all()
    assert (logged_data[f"recall_at_{k}/score"] <= 1).all()
    assert (logged_data[f"ndcg_at_{k}/score"] <= 1).all()
    assert logged_data["ground_truth"].tolist() == [["doc1", "doc2"]] * 3


def test_evaluate_retriever():
    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [["doc1", "doc2"]] * 3})

    def fn(X):
        return pd.DataFrame({"retrieved_context": [["doc1", "doc3", "doc2"]] * len(X)})

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluators="default",
            evaluator_config={
                "k": 3,
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 2 / 3,
        "precision_at_3/variance": 0,
        "precision_at_3/p90": 2 / 3,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_3/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_3/variance": 0.0,
    }
    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_retriever_logged_data(logged_data)

    # test with a big k to ensure we use min(k, len(retrieved_chunks))
    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluators="default",
            evaluator_config={
                "retriever_k": 6,
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_6/mean": 2 / 3,
        "precision_at_6/variance": 0,
        "precision_at_6/p90": 2 / 3,
        "recall_at_6/mean": 1.0,
        "recall_at_6/p90": 1.0,
        "recall_at_6/variance": 0.0,
        "ndcg_at_6/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_6/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_6/variance": 0.0,
    }

    # test with default k
    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            model_type="retriever",
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 2 / 3,
        "precision_at_3/variance": 0,
        "precision_at_3/p90": 2 / 3,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_3/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_3/variance": 0.0,
    }

    # test with multiple chunks from same doc
    def fn2(X):
        return pd.DataFrame({"retrieved_context": [["doc1", "doc1", "doc3"]] * len(X)})

    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [["doc1", "doc3"]] * 3})

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn2,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluator_config={
                "default": {
                    "retriever_k": 3,
                }
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 1,
        "precision_at_3/p90": 1,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": 1.0,
        "ndcg_at_3/p90": 1.0,
        "ndcg_at_3/variance": 0.0,
    }

    # test with empty retrieved doc
    def fn3(X):
        return pd.DataFrame({"output": [[]] * len(X)})

    with mlflow.start_run() as run:
        mlflow.evaluate(
            model=fn3,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluator_config={
                "default": {
                    "retriever_k": 4,
                }
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_4/mean": 0,
        "precision_at_4/p90": 0,
        "precision_at_4/variance": 0,
        "recall_at_4/mean": 0,
        "recall_at_4/p90": 0,
        "recall_at_4/variance": 0,
        "ndcg_at_4/mean": 0.0,
        "ndcg_at_4/p90": 0.0,
        "ndcg_at_4/variance": 0.0,
    }

    # test with a static dataset
    X_1 = pd.DataFrame(
        {
            "question": [["q1?"]] * 3,
            "targets_param": [["doc1", "doc2"]] * 3,
            "predictions_param": [["doc1", "doc4", "doc5"]] * 3,
        }
    )
    with mlflow.start_run() as run:
        mlflow.evaluate(
            data=X_1,
            predictions="predictions_param",
            targets="targets_param",
            model_type="retriever",
            extra_metrics=[mlflow.metrics.precision_at_k(4), mlflow.metrics.recall_at_k(4)],
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 1 / 3,
        "precision_at_3/p90": 1 / 3,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 0.5,
        "recall_at_3/p90": 0.5,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.6131471927654585),
        "ndcg_at_3/p90": pytest.approx(0.6131471927654585),
        "ndcg_at_3/variance": 0.0,
        "precision_at_4/mean": 1 / 3,
        "precision_at_4/p90": 1 / 3,
        "precision_at_4/variance": 0.0,
        "recall_at_4/mean": 0.5,
        "recall_at_4/p90": 0.5,
        "recall_at_4/variance": 0.0,
    }

    # test to make sure it silently fails with invalid k
    with mlflow.start_run() as run:
        mlflow.evaluate(
            data=X_1,
            predictions="predictions_param",
            targets="targets_param",
            model_type="retriever",
            extra_metrics=[mlflow.metrics.precision_at_k(-1)],
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 1 / 3,
        "precision_at_3/p90": 1 / 3,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 0.5,
        "recall_at_3/p90": 0.5,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.6131471927654585),
        "ndcg_at_3/p90": pytest.approx(0.6131471927654585),
        "ndcg_at_3/variance": 0.0,
    }

    # silent failure with evaluator_config method too!
    with mlflow.start_run() as run:
        mlflow.evaluate(
            data=X_1,
            predictions="predictions_param",
            targets="targets_param",
            model_type="retriever",
            evaluators="default",
            evaluator_config={
                "retriever_k": -1,
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {}


def test_evaluate_retriever_builtin_metrics_no_model_type():
    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [["doc1", "doc2"]] * 3})

    def fn(X):
        return {"retrieved_context": [["doc1", "doc3", "doc2"]] * len(X)}

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            extra_metrics=[
                mlflow.metrics.precision_at_k(4),
                mlflow.metrics.recall_at_k(4),
                mlflow.metrics.ndcg_at_k(4),
            ],
        )
    run = mlflow.get_run(run.info.run_id)
    assert (
        run.data.metrics
        == results.metrics
        == {
            "precision_at_4/mean": 2 / 3,
            "precision_at_4/p90": 2 / 3,
            "precision_at_4/variance": 0.0,
            "recall_at_4/mean": 1.0,
            "recall_at_4/p90": 1.0,
            "recall_at_4/variance": 0.0,
            "ndcg_at_4/mean": pytest.approx(0.9197207891481877),
            "ndcg_at_4/p90": pytest.approx(0.9197207891481877),
            "ndcg_at_4/variance": 0.0,
        }
    )
    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_retriever_logged_data(logged_data, 4)


def test_evaluate_retriever_with_numpy_array_values():
    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [np.array(["doc1", "doc2"])] * 3})

    def fn(X):
        return pd.DataFrame({"retrieved_context": [np.array(["doc1", "doc3", "doc2"])] * len(X)})

    with mlflow.start_run():
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluators="default",
            evaluator_config={
                "k": 3,
            },
        )
    assert results.metrics == {
        "precision_at_3/mean": 2 / 3,
        "precision_at_3/p90": 2 / 3,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_3/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_3/variance": 0.0,
    }


def test_evaluate_retriever_with_ints():
    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [[1, 2]] * 3})

    def fn(X):
        return pd.DataFrame({"retrieved_context": [np.array([1, 3, 2])] * len(X)})

    with mlflow.start_run():
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluators="default",
            evaluator_config={
                "k": 3,
            },
        )
    assert results.metrics == {
        "precision_at_3/mean": 2 / 3,
        "precision_at_3/p90": 2 / 3,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_3/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_3/variance": 0.0,
    }


def test_evaluate_with_numpy_array():
    data = [
        ["What is MLflow?"],
    ]
    ground_truth = [
        "MLflow is an open-source platform for managing the end-to-end machine learning",
    ]

    with mlflow.start_run():
        logged_model = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        results = mlflow.evaluate(
            logged_model.model_uri,
            data,
            targets=ground_truth,
            extra_metrics=[mlflow.metrics.toxicity()],
        )

        assert results.metrics.keys() == {
            "toxicity/v1/mean",
            "toxicity/v1/variance",
            "toxicity/v1/p90",
            "toxicity/v1/ratio",
        }
        assert len(results.tables) == 1
        assert results.tables["eval_results_table"].columns.tolist() == [
            "feature_1",
            "target",
            "outputs",
            "toxicity/v1/score",
        ]


def test_target_prediction_col_mapping():
    metric = mlflow.metrics.genai.make_genai_metric(
        name="correctness",
        definition=(
            "Correctness refers to how well the generated output matches "
            "or aligns with the reference or ground truth text that is considered "
            "accurate and appropriate for the given input. The ground truth serves as "
            "a benchmark against which the provided output is compared to determine the "
            "level of accuracy and fidelity."
        ),
        grading_prompt=(
            "Correctness: If the answer correctly answer the question, below "
            "are the details for different scores: "
            "- Score 0: the answer is completely incorrect, doesn't mention anything about "
            "the question or is completely contrary to the correct answer. "
            "- Score 1: the answer provides some relevance to the question and answer "
            "one aspect of the question correctly. "
            "- Score 2: the answer mostly answer the question but is missing or hallucinating "
            "on one critical aspect. "
            "- Score 3: the answer correctly answer the question and not missing any "
            "major aspect"
        ),
        examples=[],
        version="v1",
        model="openai:/gpt-4",
        grading_context_columns=["renamed_ground_truth"],
        parameters={"temperature": 0.0},
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ):
        with mlflow.start_run():
            eval_df = pd.DataFrame(
                {
                    "inputs": [
                        "What is MLflow?",
                        "What is Spark?",
                        "What is Python?",
                    ],
                    "ground_truth": [
                        "MLflow is an open-source platform",
                        "Apache Spark is an open-source, distributed computing system",
                        "Python is a high-level programming language",
                    ],
                    "prediction": [
                        "MLflow is an open-source platform",
                        "Apache Spark is an open-source, distributed computing system",
                        "Python is a high-level programming language",
                    ],
                }
            )
            results = mlflow.evaluate(
                data=eval_df,
                evaluators="default",
                targets="renamed_ground_truth",
                predictions="prediction",
                extra_metrics=[metric],
                evaluator_config={"col_mapping": {"renamed_ground_truth": "ground_truth"}},
            )

            assert results.metrics == {
                "correctness/v1/mean": 3.0,
                "correctness/v1/variance": 0.0,
                "correctness/v1/p90": 3.0,
            }


def test_precanned_metrics_work():
    metric = mlflow.metrics.rouge1()
    with mlflow.start_run():
        eval_df = pd.DataFrame(
            {
                "inputs": [
                    "What is MLflow?",
                    "What is Spark?",
                    "What is Python?",
                ],
                "ground_truth": [
                    "MLflow is an open-source platform",
                    "Apache Spark is an open-source, distributed computing system",
                    "Python is a high-level programming language",
                ],
                "prediction": [
                    "MLflow is an open-source platform",
                    "Apache Spark is an open-source, distributed computing system",
                    "Python is a high-level programming language",
                ],
            }
        )

        results = mlflow.evaluate(
            data=eval_df,
            evaluators="default",
            predictions="prediction",
            extra_metrics=[metric],
            evaluator_config={
                "col_mapping": {
                    "targets": "ground_truth",
                }
            },
        )

        assert results.metrics == {
            "rouge1/v1/mean": 1.0,
            "rouge1/v1/variance": 0.0,
            "rouge1/v1/p90": 1.0,
        }


def test_evaluate_custom_metric_with_string_type():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        from mlflow.metrics import make_metric

        def word_count_eval(predictions):
            scores = []
            avg = 0
            aggregate_results = {}
            for prediction in predictions:
                scores.append(prediction)
                avg += len(prediction.split(" "))

            avg /= len(predictions)
            aggregate_results["avg_len"] = avg

            return MetricValue(
                scores=scores,
                aggregate_results=aggregate_results,
            )

        word_count = make_metric(eval_fn=word_count_eval, greater_is_better=True, name="word_count")

        results = mlflow.evaluate(model_info.model_uri, data, extra_metrics=[word_count])
        assert results.metrics["word_count/avg_len"] == 3.0
        assert len(results.tables) == 1
        assert results.tables["eval_results_table"].columns.tolist() == [
            "text",
            "outputs",
            "word_count/score",
        ]
        pd.testing.assert_series_equal(
            results.tables["eval_results_table"]["word_count/score"],
            data["text"],
            check_names=False,
        )


def test_do_not_log_built_in_metrics_as_artifacts():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "inputs": ["words random", "This is a sentence."],
                "ground_truth": ["words random", "This is a sentence."],
            }
        )
        mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="ground_truth",
            predictions="answer",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[
                toxicity(),
                flesch_kincaid_grade_level(),
            ],
        )
        client = mlflow.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
        assert _GENAI_CUSTOM_METRICS_FILE_NAME not in artifacts

        results = retrieve_custom_metrics(run_id=run.info.run_id)
        assert len(results) == 0


def test_log_genai_custom_metrics_as_artifacts():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "inputs": ["words random", "This is a sentence."],
                "ground_truth": ["words random", "This is a sentence."],
            }
        )
        example = EvaluationExample(
            input="What is MLflow?",
            output="MLflow is an open-source platform for managing machine learning workflows.",
            score=4,
            justification="test",
            grading_context={"targets": "test"},
        )
        # This simulates the code path for metrics created from make_genai_metric
        answer_similarity_metric = answer_similarity(
            model="gateway:/gpt-4o-mini", examples=[example]
        )
        another_custom_metric = make_genai_metric_from_prompt(
            name="another custom llm judge",
            judge_prompt="This is another custom judge prompt.",
            greater_is_better=False,
            parameters={"temperature": 0.0},
        )
        result = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="ground_truth",
            predictions="answer",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[
                answer_similarity_metric,
                another_custom_metric,
            ],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert _GENAI_CUSTOM_METRICS_FILE_NAME in artifacts

    table = result.tables[os.path.splitext(_GENAI_CUSTOM_METRICS_FILE_NAME)[0]]
    assert table.loc[0, "name"] == "answer_similarity"
    assert table.loc[0, "version"] == "v1"
    assert table.loc[1, "name"] == "another custom llm judge"
    assert table.loc[1, "version"] == ""
    assert table["version"].dtype == "object"

    results = retrieve_custom_metrics(run.info.run_id)
    assert len(results) == 2
    assert [r.name for r in results] == ["answer_similarity", "another custom llm judge"]

    results = retrieve_custom_metrics(run_id=run.info.run_id, name="another custom llm judge")
    assert len(results) == 1
    assert results[0].name == "another custom llm judge"

    results = retrieve_custom_metrics(run_id=run.info.run_id, version="v1")
    assert len(results) == 1
    assert results[0].name == "answer_similarity"

    results = retrieve_custom_metrics(
        run_id=run.info.run_id, name="answer_similarity", version="v1"
    )
    assert len(results) == 1
    assert results[0].name == "answer_similarity"

    results = retrieve_custom_metrics(run_id=run.info.run_id, name="do not match")
    assert len(results) == 0

    results = retrieve_custom_metrics(run_id=run.info.run_id, version="do not match")
    assert len(results) == 0


def test_all_genai_custom_metrics_are_from_user_prompt():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "inputs": ["words random", "This is a sentence."],
                "ground_truth": ["words random", "This is a sentence."],
            }
        )
        custom_metric = make_genai_metric_from_prompt(
            name="custom llm judge",
            judge_prompt="This is a custom judge prompt.",
            greater_is_better=False,
            parameters={"temperature": 0.0},
        )
        another_custom_metric = make_genai_metric_from_prompt(
            name="another custom llm judge",
            judge_prompt="This is another custom judge prompt.",
            greater_is_better=False,
            parameters={"temperature": 0.7},
        )
        result = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="ground_truth",
            predictions="answer",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[
                custom_metric,
                another_custom_metric,
            ],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert _GENAI_CUSTOM_METRICS_FILE_NAME in artifacts

    table = result.tables[os.path.splitext(_GENAI_CUSTOM_METRICS_FILE_NAME)[0]]
    assert table.loc[0, "name"] == "custom llm judge"
    assert table.loc[1, "name"] == "another custom llm judge"
    assert table.loc[0, "version"] == ""
    assert table.loc[1, "version"] == ""
    assert table["version"].dtype == "object"
