import logging

import numpy as np

from mlflow.metrics.base import MetricValue

_logger = logging.getLogger(__name__)


def standard_aggregations(scores):
    return {
        "mean": np.mean(scores),
        "variance": np.var(scores),
        "p90": np.percentile(scores, 90),
    }


def _validate_text_data(data, metric_name, column_name):
    """Validates that the data is text and is non-empty"""
    if len(data) == 0:
        return False

    for row, line in enumerate(data):
        if not isinstance(line, str):
            _logger.warning(
                f"Cannot calculate {metric_name} for non-string inputs. "
                + f"Non-string found for {column_name} on row {row}. skipping metric logging."
            )
            return False

    return True


def _toxicity_eval_fn(predictions, targets, metrics):
    if not _validate_text_data(predictions, "toxicity", "predictions"):
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


def _perplexity_eval_fn(predictions, targets, metrics):
    if not _validate_text_data(predictions, "perplexity", "predictions"):
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


def _flesch_kincaid_eval_fn(predictions, targets, metrics):
    if not _validate_text_data(predictions, "flesch_kincaid", "predictions"):
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


def _ari_eval_fn(predictions, targets, metrics):
    if not _validate_text_data(predictions, "ari", "predictions"):
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


def _accuracy_eval_fn(predictions, targets, metrics, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import accuracy_score

        acc = accuracy_score(y_true=targets, y_pred=predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"exact_match": acc})


def _rouge1_eval_fn(predictions, targets, metrics):
    if targets is not None and len(targets) != 0:
        if not _validate_text_data(targets, "rouge1", "targets") or not _validate_text_data(
            predictions, "rouge1", "predictions"
        ):
            return

        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        scores = rouge.compute(
            predictions=predictions,
            references=targets,
            rouge_types=["rouge1"],
            use_aggregator=False,
        )["rouge1"]
        return MetricValue(
            scores=scores,
            aggregate_results=standard_aggregations(scores),
        )


def _rouge2_eval_fn(predictions, targets, metrics):
    if targets is not None and len(targets) != 0:
        if not _validate_text_data(targets, "rouge2", "targets") or not _validate_text_data(
            predictions, "rouge2", "predictions"
        ):
            return

        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        scores = rouge.compute(
            predictions=predictions,
            references=targets,
            rouge_types=["rouge2"],
            use_aggregator=False,
        )["rouge2"]
        return MetricValue(
            scores=scores,
            aggregate_results=standard_aggregations(scores),
        )


def _rougeL_eval_fn(predictions, targets, metrics):
    if targets is not None and len(targets) != 0:
        if not _validate_text_data(targets, "rougeL", "targets") or not _validate_text_data(
            predictions, "rougeL", "predictions"
        ):
            return

        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        scores = rouge.compute(
            predictions=predictions,
            references=targets,
            rouge_types=["rougeL"],
            use_aggregator=False,
        )["rougeL"]
        return MetricValue(
            scores=scores,
            aggregate_results=standard_aggregations(scores),
        )


def _rougeLsum_eval_fn(predictions, targets, metrics):
    if targets is not None and len(targets) != 0:
        if not _validate_text_data(targets, "rougeLsum", "targets") or not _validate_text_data(
            predictions, "rougeLsum", "predictions"
        ):
            return

        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        scores = rouge.compute(
            predictions=predictions,
            references=targets,
            rouge_types=["rougeLsum"],
            use_aggregator=False,
        )["rougeLsum"]
        return MetricValue(
            scores=scores,
            aggregate_results=standard_aggregations(scores),
        )


def _mae_eval_fn(predictions, targets, metrics, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_absolute_error

        mae = mean_absolute_error(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"mean_absolute_error": mae})


def _mse_eval_fn(predictions, targets, metrics, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_squared_error

        mse = mean_squared_error(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"mean_squared_error": mse})


def _rmse_eval_fn(predictions, targets, metrics, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_squared_error

        rmse = mean_squared_error(targets, predictions, squared=False, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"root_mean_squared_error": rmse})


def _r2_score_eval_fn(predictions, targets, metrics, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import r2_score

        r2 = r2_score(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"r2_score": r2})


def _max_error_eval_fn(predictions, targets, metrics):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import max_error

        error = max_error(targets, predictions)
        return MetricValue(aggregate_results={"max_error": error})


def _mape_eval_fn(predictions, targets, metrics, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_absolute_percentage_error

        mape = mean_absolute_percentage_error(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"mean_absolute_percentage_error": mape})


def _recall_eval_fn(
    predictions, targets, metrics, pos_label=1, average="binary", sample_weight=None
):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import recall_score

        recall = recall_score(
            targets, predictions, pos_label=pos_label, average=average, sample_weight=sample_weight
        )
        return MetricValue(aggregate_results={"recall_score": recall})


def _precision_eval_fn(
    predictions, targets, metrics, pos_label=1, average="binary", sample_weight=None
):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import precision_score

        precision = precision_score(
            targets,
            predictions,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
        )
        return MetricValue(aggregate_results={"precision_score": precision})


def _f1_score_eval_fn(
    predictions, targets, metrics, pos_label=1, average="binary", sample_weight=None
):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import f1_score

        f1 = f1_score(
            targets,
            predictions,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
        )
        return MetricValue(aggregate_results={"f1_score": f1})
