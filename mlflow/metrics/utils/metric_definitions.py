import logging

import numpy as np
import pandas as pd

from mlflow.metrics.base import MetricValue

_logger = logging.getLogger(__name__)


def standard_aggregations(scores):
    return {
        "mean": np.mean(scores),
        "variance": np.var(scores),
        "p90": np.percentile(scores, 90),
    }


def _validate_text_predictions(predictions, metric_name):
    if len(predictions) == 0:
        return False

    if any(not isinstance(prediction, str) for prediction in predictions):
        _logger.warning(
            f"Cannot calculate {metric_name} for non-string inputs, skipping metric logging."
        )
        return False

    return True


def _toxicity_eval_fn(eval_df, metrics):
    y_pred = eval_df["prediction"]
    predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred

    if not _validate_text_predictions(predictions, "toxicity"):
        return

    try:
        _logger.info("Loading toxicity metric:")
        import evaluate

        toxicity = evaluate.load("toxicity", module_type="measurement")
    except Exception as e:
        _logger.warning(
            f"Failed to load 'toxicity' metric (error: {e!r}), skipping metric logging."
        )
        return

    _logger.info("Computing toxicity metric:")
    scores = toxicity.compute(predictions=predictions)["toxicity"]
    toxicity_ratio = toxicity.compute(predictions=predictions, aggregation="ratio")[
        "toxicity_ratio"
    ]
    return MetricValue(
        scores=scores,
        aggregate_results={
            **standard_aggregations(scores),
            "ratio": toxicity_ratio,
        },
    )


def _perplexity_eval_fn(eval_df, metrics):
    y_pred = eval_df["prediction"]
    predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred

    if not _validate_text_predictions(predictions, "perplexity"):
        return

    try:
        _logger.info("Loading perplexity metric:")
        import evaluate

        perplexity = evaluate.load("perplexity", module_type="metric")
    except Exception as e:
        _logger.warning(
            f"Failed to load 'perplexity' metric (error: {e!r}), skipping metric logging."
        )
        return

    _logger.info("Computing perplexity metric:")
    scores = perplexity.compute(predictions=predictions, model_id="gpt2")["perplexities"]
    return MetricValue(
        scores=scores,
        aggregate_results=standard_aggregations(scores),
    )


def _flesch_kincaid_eval_fn(eval_df, metrics):
    y_pred = eval_df["prediction"]
    predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred

    if not _validate_text_predictions(predictions, "flesch_kincaid"):
        return

    try:
        import textstat
    except ImportError:
        _logger.warning("Failed to load flesch kincaid metric, skipping metric logging.")
        return

    _logger.info("Computing flesch kincaid metric:")
    scores = [textstat.flesch_kincaid_grade(prediction) for prediction in predictions]
    return MetricValue(
        scores=scores,
        aggregate_results=standard_aggregations(scores),
    )


def _ari_eval_fn(eval_df, metrics):
    y_pred = eval_df["prediction"]
    predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred

    if not _validate_text_predictions(predictions, "ari"):
        return

    try:
        import textstat
    except ImportError:
        _logger.warning(
            "Failed to load automated readability index metric, skipping metric logging."
        )
        return

    _logger.info("Computing automated readability index metric:")
    scores = [textstat.automated_readability_index(prediction) for prediction in predictions]
    return MetricValue(
        scores=scores,
        aggregate_results=standard_aggregations(scores),
    )


def _accuracy_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        from sklearn.metrics import accuracy_score

        acc = accuracy_score(y_true=eval_df["target"], y_pred=eval_df["prediction"])
        return MetricValue(aggregate_results={"": acc})


def _rouge1_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        y_pred = eval_df["prediction"]
        predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred
        references = eval_df["target"]

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge1"],
            use_aggregator=False,
        )["rouge1"]
        return MetricValue(
            scores=scores,
            aggregate_results=standard_aggregations(scores),
        )


def _rouge2_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        y_pred = eval_df["prediction"]
        predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred
        references = eval_df["target"]

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge2"],
            use_aggregator=False,
        )["rouge2"]
        return MetricValue(
            scores=scores,
            aggregate_results=standard_aggregations(scores),
        )


def _rougeL_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        y_pred = eval_df["prediction"]
        predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred
        references = eval_df["target"]

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeL"],
            use_aggregator=False,
        )["rougeL"]
        return MetricValue(
            scores=scores,
            aggregate_results=standard_aggregations(scores),
        )


def _rougeLsum_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        y_pred = eval_df["prediction"]
        predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred
        references = eval_df["target"]

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeLsum"],
            use_aggregator=False,
        )["rougeLsum"]
        return MetricValue(
            scores=scores,
            aggregate_results=standard_aggregations(scores),
        )
