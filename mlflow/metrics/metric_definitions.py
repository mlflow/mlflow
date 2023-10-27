import functools
import logging
import os

import numpy as np

from mlflow.metrics.base import MetricValue

_logger = logging.getLogger(__name__)


# used to silently fail with invalid metric params
def noop(*args, **kwargs):
    return None


targets_col_specifier = "the column specified by the `targets` parameter"
predictions_col_specifier = (
    "the column specified by the `predictions` parameter or the model output column"
)


def standard_aggregations(scores):
    return {
        "mean": np.mean(scores),
        "variance": np.var(scores),
        "p90": np.percentile(scores, 90),
    }


def _validate_text_data(data, metric_name, col_specifier):
    """Validates that the data is a list of strs and is non-empty"""
    if data is None or len(data) == 0:
        return False

    for row, line in enumerate(data):
        if not isinstance(line, str):
            _logger.warning(
                f"Cannot calculate {metric_name} for non-string inputs. "
                f"Non-string found for {col_specifier} on row {row}. Skipping metric logging."
            )
            return False

    return True


def _validate_list_str_data(data, metric_name, col_specifier):
    """Validates that the data is a list of lists of strings and is non-empty"""
    if data is None or len(data) == 0:
        return False

    for index, value in data.items():
        if not isinstance(value, list) or not all(isinstance(val, str) for val in value):
            _logger.warning(
                f"Cannot calculate metric '{metric_name}' for non-list of string inputs. "
                f"Non-list of strings found for {col_specifier} on row {index}. Skipping metric "
                f"logging."
            )
            return False

    return True


def _token_count_eval_fn(predictions, targets=None, metrics=None):
    import tiktoken

    # ref: https://github.com/openai/tiktoken/issues/75
    os.environ["TIKTOKEN_CACHE_DIR"] = ""
    encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = []
    for prediction in predictions:
        if isinstance(prediction, str):
            num_tokens.append(len(encoding.encode(prediction)))
        else:
            num_tokens.append(None)

    return MetricValue(
        scores=num_tokens,
    )


@functools.lru_cache(maxsize=8)
def _cached_evaluate_load(path, module_type=None):
    import evaluate

    return evaluate.load(path, module_type=module_type)


def _toxicity_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(predictions, "toxicity", predictions_col_specifier):
        return
    try:
        toxicity = _cached_evaluate_load("toxicity", module_type="measurement")
    except Exception as e:
        _logger.warning(
            f"Failed to load 'toxicity' metric (error: {e!r}), skipping metric logging."
        )
        return

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


def _flesch_kincaid_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(predictions, "flesch_kincaid", predictions_col_specifier):
        return

    try:
        import textstat
    except ImportError:
        _logger.warning("Failed to load flesch kincaid metric, skipping metric logging.")
        return

    scores = [textstat.flesch_kincaid_grade(prediction) for prediction in predictions]
    return MetricValue(
        scores=scores,
        aggregate_results=standard_aggregations(scores),
    )


def _ari_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(predictions, "ari", predictions_col_specifier):
        return

    try:
        import textstat
    except ImportError:
        _logger.warning(
            "Failed to load automated readability index metric, skipping metric logging."
        )
        return

    scores = [textstat.automated_readability_index(prediction) for prediction in predictions]
    return MetricValue(
        scores=scores,
        aggregate_results=standard_aggregations(scores),
    )


def _accuracy_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import accuracy_score

        acc = accuracy_score(y_true=targets, y_pred=predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"exact_match": acc})


def _rouge1_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(targets, "rouge1", targets_col_specifier) or not _validate_text_data(
        predictions, "rouge1", predictions_col_specifier
    ):
        return

    try:
        rouge = _cached_evaluate_load("rouge")
    except Exception as e:
        _logger.warning(f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging.")
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


def _rouge2_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(targets, "rouge2", targets_col_specifier) or not _validate_text_data(
        predictions, "rouge2", predictions_col_specifier
    ):
        return

    try:
        rouge = _cached_evaluate_load("rouge")
    except Exception as e:
        _logger.warning(f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging.")
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


def _rougeL_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(targets, "rougeL", targets_col_specifier) or not _validate_text_data(
        predictions, "rougeL", predictions_col_specifier
    ):
        return

    try:
        rouge = _cached_evaluate_load("rouge")
    except Exception as e:
        _logger.warning(f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging.")
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


def _rougeLsum_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(
        targets, "rougeLsum", targets_col_specifier
    ) or not _validate_text_data(predictions, "rougeLsum", predictions_col_specifier):
        return

    try:
        rouge = _cached_evaluate_load("rouge")
    except Exception as e:
        _logger.warning(f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging.")
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


def _mae_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_absolute_error

        mae = mean_absolute_error(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"mean_absolute_error": mae})


def _mse_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_squared_error

        mse = mean_squared_error(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"mean_squared_error": mse})


def _rmse_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_squared_error

        rmse = mean_squared_error(targets, predictions, squared=False, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"root_mean_squared_error": rmse})


def _r2_score_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import r2_score

        r2 = r2_score(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"r2_score": r2})


def _max_error_eval_fn(predictions, targets=None, metrics=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import max_error

        error = max_error(targets, predictions)
        return MetricValue(aggregate_results={"max_error": error})


def _mape_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_absolute_percentage_error

        mape = mean_absolute_percentage_error(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={"mean_absolute_percentage_error": mape})


def _recall_eval_fn(
    predictions, targets=None, metrics=None, pos_label=1, average="binary", sample_weight=None
):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import recall_score

        recall = recall_score(
            targets, predictions, pos_label=pos_label, average=average, sample_weight=sample_weight
        )
        return MetricValue(aggregate_results={"recall_score": recall})


def _precision_eval_fn(
    predictions, targets=None, metrics=None, pos_label=1, average="binary", sample_weight=None
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
    predictions, targets=None, metrics=None, pos_label=1, average="binary", sample_weight=None
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


def _precision_at_k_eval_fn(k):
    if not (isinstance(k, int) and k > 0):
        _logger.warning(
            f"Cannot calculate 'precision_at_k' for invalid parameter 'k'. "
            f"'k' should be a positive integer; found: {k}. Skipping metric logging."
        )
        return noop

    def _fn(predictions, targets):
        if not _validate_list_str_data(
            predictions, "precision_at_k", predictions_col_specifier
        ) or not _validate_list_str_data(targets, "precision_at_k", targets_col_specifier):
            return

        scores = []
        for target, prediction in zip(targets, predictions):
            # only include the top k retrieved chunks
            ground_truth, retrieved = set(target), prediction[:k]
            relevant_doc_count = sum(1 for doc in retrieved if doc in ground_truth)
            if len(retrieved) > 0:
                scores.append(relevant_doc_count / len(retrieved))
            else:
                # when no documents are retrieved, precision is 0
                scores.append(0)

        return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))

    return _fn


def _recall_at_k_eval_fn(k):
    if not (isinstance(k, int) and k > 0):
        _logger.warning(
            f"Cannot calculate 'precision_at_k' for invalid parameter 'k'. "
            f"'k' should be a positive integer; found: {k}. Skipping metric logging."
        )
        return noop

    def _fn(predictions, targets):
        if not _validate_list_str_data(
            predictions, "precision_at_k", predictions_col_specifier
        ) or not _validate_list_str_data(targets, "precision_at_k", targets_col_specifier):
            return

        scores = []
        for target, prediction in zip(targets, predictions):
            # only include the top k retrieved chunks
            ground_truth, retrieved = set(target), set(prediction[:k])
            relevant_doc_count = len(ground_truth.intersection(retrieved))
            if len(ground_truth) > 0:
                scores.append(relevant_doc_count / len(ground_truth))
            elif len(retrieved) == 0:
                # there are 0 retrieved and ground truth docs, so reward for the match
                scores.append(1)
            else:
                # there are > 0 retrieved, but 0 ground truth, so penalize
                scores.append(0)

        return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))

    return _fn
