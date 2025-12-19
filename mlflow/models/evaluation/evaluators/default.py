import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

import mlflow
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
    MetricValue,
    ari_grade_level,
    exact_match,
    flesch_kincaid_grade_level,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge1,
    rouge2,
    rougeL,
    rougeLsum,
    token_count,
    toxicity,
)
from mlflow.metrics.genai.genai_metric import _GENAI_CUSTOM_METRICS_FILE_NAME
from mlflow.models.evaluation.artifacts import JsonEvaluationArtifact
from mlflow.models.evaluation.base import EvaluationMetric, EvaluationResult, _ModelType
from mlflow.models.evaluation.default_evaluator import (
    _LATENCY_METRIC_NAME,
    BuiltInEvaluator,
    _extract_output_and_other_columns,
    _extract_predict_fn,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)


class DefaultEvaluator(BuiltInEvaluator):
    """
    The default built-in evaluator for any models that cannot be evaluated
    by other built-in evaluators, such as question-answering.
    """

    name = "default"

    @classmethod
    def can_evaluate(cls, *, model_type, evaluator_config, **kwargs):
        return model_type in _ModelType.values() or model_type is None

    def _evaluate(
        self,
        model: Optional["mlflow.pyfunc.PyFuncModel"],
        extra_metrics: list[EvaluationMetric],
        custom_artifacts=None,
        **kwargs,
    ) -> EvaluationResult | None:
        compute_latency = False
        for extra_metric in extra_metrics:
            # If latency metric is specified, we will compute latency for the model
            # during prediction, and we will remove the metric from the list of extra
            # metrics to be computed after prediction.
            if extra_metric.name == _LATENCY_METRIC_NAME:
                compute_latency = True
                extra_metrics.remove(extra_metric)
        self._log_genai_custom_metrics(extra_metrics)

        # Generate model predictions and evaluate metrics
        y_pred, other_model_outputs, self.predictions = self._generate_model_predictions(
            model, input_df=self.X.copy_to_avoid_mutation(), compute_latency=compute_latency
        )
        y_true = self.dataset.labels_data

        metrics = self._builtin_metrics() + extra_metrics
        self.evaluate_metrics(
            metrics,
            prediction=y_pred,
            target=self.dataset.labels_data,
            other_output_df=other_model_outputs,
        )
        self.evaluate_and_log_custom_artifacts(custom_artifacts, prediction=y_pred, target=y_true)

        # Log metrics and artifacts
        self.log_metrics()
        self.log_eval_table(y_pred, other_model_outputs)
        return EvaluationResult(
            metrics=self.aggregate_metrics, artifacts=self.artifacts, run_id=self.run_id
        )

    def _builtin_metrics(self) -> list[Metric]:
        """
        Get a list of builtin metrics for the model type.
        """
        if self.model_type is None:
            return []

        text_metrics = [
            token_count(),
            toxicity(),
            flesch_kincaid_grade_level(),
            ari_grade_level(),
        ]
        builtin_metrics = []

        # NB: Classifier and Regressor are handled by dedicated built-in evaluators,
        if self.model_type == _ModelType.QUESTION_ANSWERING:
            builtin_metrics = [*text_metrics, exact_match()]
        elif self.model_type == _ModelType.TEXT_SUMMARIZATION:
            builtin_metrics = [
                *text_metrics,
                rouge1(),
                rouge2(),
                rougeL(),
                rougeLsum(),
            ]
        elif self.model_type == _ModelType.TEXT:
            builtin_metrics = text_metrics
        elif self.model_type == _ModelType.RETRIEVER:
            # default k to 3 if not specified
            retriever_k = self.evaluator_config.pop("retriever_k", 3)
            builtin_metrics = [
                precision_at_k(retriever_k),
                recall_at_k(retriever_k),
                ndcg_at_k(retriever_k),
            ]

        return builtin_metrics

    def _generate_model_predictions(
        self,
        model: Optional["mlflow.pyfunc.PyFuncModel"],
        input_df: pd.DataFrame,
        compute_latency=False,
    ):
        """
        Helper method for generating model predictions
        """
        predict_fn = _extract_predict_fn(model)

        def predict_with_latency(X_copy):
            y_pred_list = []
            pred_latencies = []
            if len(X_copy) == 0:
                raise ValueError("Empty input data")

            is_dataframe = isinstance(X_copy, pd.DataFrame)

            for row in X_copy.iterrows() if is_dataframe else enumerate(X_copy):
                i, row_data = row
                single_input = row_data.to_frame().T if is_dataframe else row_data
                start_time = time.time()
                y_pred = predict_fn(single_input)
                end_time = time.time()
                pred_latencies.append(end_time - start_time)
                y_pred_list.append(y_pred)

            # Update latency metric
            self.metrics_values.update({_LATENCY_METRIC_NAME: MetricValue(scores=pred_latencies)})

            # Aggregate all predictions into model_predictions
            sample_pred = y_pred_list[0]
            if isinstance(sample_pred, pd.DataFrame):
                return pd.concat(y_pred_list)
            elif isinstance(sample_pred, np.ndarray):
                return np.concatenate(y_pred_list, axis=0)
            elif isinstance(sample_pred, list):
                return sum(y_pred_list, [])
            elif isinstance(sample_pred, pd.Series):
                return pd.concat(y_pred_list, ignore_index=True)
            elif isinstance(sample_pred, str):
                return y_pred_list
            else:
                raise MlflowException(
                    message=f"Unsupported prediction type {type(sample_pred)} for model type "
                    f"{self.model_type}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        if model is not None:
            _logger.info("Computing model predictions.")

            if compute_latency:
                model_predictions = predict_with_latency(input_df)
            else:
                model_predictions = predict_fn(input_df)
        else:
            if compute_latency:
                _logger.warning(
                    "Setting the latency to 0 for all entries because the model is not provided."
                )
                self.metrics_values.update(
                    {_LATENCY_METRIC_NAME: MetricValue(scores=[0.0] * len(input_df))}
                )
            model_predictions = self.dataset.predictions_data

        output_column_name = self.predictions
        (
            y_pred,
            other_output_df,
            predictions_column_name,
        ) = _extract_output_and_other_columns(model_predictions, output_column_name)

        return y_pred, other_output_df, predictions_column_name

    def _log_genai_custom_metrics(self, extra_metrics: list[EvaluationMetric]):
        genai_custom_metrics = [
            extra_metric.genai_metric_args
            for extra_metric in extra_metrics
            # When the field is present, the metric is created from either make_genai_metric
            # or make_genai_metric_from_prompt. We will log the metric definition.
            if extra_metric.genai_metric_args is not None
        ]

        if len(genai_custom_metrics) == 0:
            return

        names = []
        versions = []
        metric_args_list = []

        for metric_args in genai_custom_metrics:
            names.append(metric_args["name"])
            # Custom metrics created from make_genai_metric_from_prompt don't have version
            versions.append(metric_args.get("version", ""))
            metric_args_list.append(metric_args)

        data = {"name": names, "version": versions, "metric_args": metric_args_list}

        mlflow.log_table(data, artifact_file=_GENAI_CUSTOM_METRICS_FILE_NAME)

        artifact_name = os.path.splitext(_GENAI_CUSTOM_METRICS_FILE_NAME)[0]
        self.artifacts[artifact_name] = JsonEvaluationArtifact(
            uri=mlflow.get_artifact_uri(_GENAI_CUSTOM_METRICS_FILE_NAME)
        )
