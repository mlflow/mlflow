import functools
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.metrics.base import MetricValue, standard_aggregations

_logger = logging.getLogger(__name__)


# used to silently fail with invalid metric params
def noop(*args, **kwargs):
    return None


targets_col_specifier = "the column specified by the `targets` parameter"
predictions_col_specifier = (
    "the column specified by the `predictions` parameter or the model output column"
)


def _validate_text_data(data, metric_name, col_specifier):
    """Validates that the data is a list of strs and is non-empty"""
    if data is None or len(data) == 0:
        _logger.warning(
            f"Cannot calculate {metric_name} for empty inputs: "
            f"{col_specifier} is empty or the parameter is not specified. Skipping metric logging."
        )
        return False

    for row, line in enumerate(data):
        if not isinstance(line, str):
            _logger.warning(
                f"Cannot calculate {metric_name} for non-string inputs. "
                f"Non-string found for {col_specifier} on row {row}. Skipping metric logging."
            )
            return False

    return True


def _validate_array_like_id_data(data, metric_name, col_specifier):
    """Validates that the data is a list of lists/np.ndarrays of strings/ints and is non-empty"""
    if data is None or len(data) == 0:
        return False

    for index, value in data.items():
        if not (
            (isinstance(value, list) and all(isinstance(val, (str, int)) for val in value))
            or (
                isinstance(value, np.ndarray)
                and (np.issubdtype(value.dtype, str) or np.issubdtype(value.dtype, int))
            )
        ):
            _logger.warning(
                f"Cannot calculate metric '{metric_name}' for non-arraylike of string or int "
                f"inputs. Non-arraylike of strings/ints found for {col_specifier} on row "
                f"{index}, value {value}. Skipping metric logging."
            )
            return False

    return True


def _load_from_github(path: str, module_type: str = "metric"):
    import evaluate

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/huggingface/evaluate.git",
                tmpdir,
            ]
        )
        path = f"{module_type}s/{path}"
        subprocess.check_call(["git", "sparse-checkout", "set", path], cwd=tmpdir)
        subprocess.check_call(["git", "checkout"], cwd=tmpdir)
        return evaluate.load(str(tmpdir / path))


@functools.lru_cache(maxsize=8)
def _cached_evaluate_load(path: str, module_type: str = "metric"):
    import evaluate

    try:
        return evaluate.load(path, module_type=module_type)
    except (FileNotFoundError, OSError):
        if _MLFLOW_TESTING.get():
            # `evaluate.load` is highly unstable and often fails due to a network error or
            # huggingface hub being down. In testing, we want to avoid this instability, so we
            # load the metric from the evaluate repository on GitHub.
            return _load_from_github(path, module_type=module_type)
        raise


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
        _logger.warning(
            "Failed to import textstat for flesch kincaid metric, skipping metric logging. "
            "Please install textstat using 'pip install textstat'."
        )
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
            "Failed to import textstat for automated readability index metric, "
            "skipping metric logging. "
            "Please install textstat using 'pip install textstat'."
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


def _root_mean_squared_error(*, y_true, y_pred, sample_weight):
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        # If root_mean_squared_error is unavailable, fall back to
        # `mean_squared_error(..., squared=False)`, which is deprecated in scikit-learn >= 1.4.
        from sklearn.metrics import mean_squared_error

        return mean_squared_error(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, squared=False
        )
    else:
        return root_mean_squared_error(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


def _rmse_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        rmse = _root_mean_squared_error(
            y_true=targets, y_pred=predictions, sample_weight=sample_weight
        )
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
        if not _validate_array_like_id_data(
            predictions, "precision_at_k", predictions_col_specifier
        ) or not _validate_array_like_id_data(targets, "precision_at_k", targets_col_specifier):
            return

        scores = []
        for target, prediction in zip(targets, predictions):
            # only include the top k retrieved chunks
            ground_truth = set(target)
            retrieved = prediction[:k]
            relevant_doc_count = sum(1 for doc in retrieved if doc in ground_truth)
            if len(retrieved) > 0:
                scores.append(relevant_doc_count / len(retrieved))
            else:
                # when no documents are retrieved, precision is 0
                scores.append(0)

        return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))

    return _fn


def _expand_duplicate_retrieved_docs(predictions, targets):
    counter = {}
    expanded_predictions = []
    expanded_targets = targets
    for doc_id in predictions:
        if doc_id not in counter:
            counter[doc_id] = 1
            expanded_predictions.append(doc_id)
        else:
            counter[doc_id] += 1
            new_doc_id = (
                f"{doc_id}_bc574ae_{counter[doc_id]}"  # adding a random string to avoid collisions
            )
            expanded_predictions.append(new_doc_id)
            if doc_id in expanded_targets:
                expanded_targets.add(new_doc_id)
    return expanded_predictions, expanded_targets


def _prepare_row_for_ndcg(predictions, targets):
    """Prepare data one row from predictions and targets to y_score, y_true for ndcg calculation.

    Args:
        predictions: A list of strings of at most k doc IDs retrieved.
        targets: A list of strings of ground-truth doc IDs.

    Returns:
        y_true : ndarray of shape (1, n_docs) Representing the ground-truth relevant docs.
            n_docs is the number of unique docs in union of predictions and targets.
        y_score : ndarray of shape (1, n_docs) Representing the retrieved docs.
            n_docs is the number of unique docs in union of predictions and targets.
    """
    # sklearn does an internal sort of y_score, so to preserve the order of our retrieved
    # docs, we need to modify the relevance value slightly
    eps = 1e-6

    # support predictions containing duplicate doc ID
    targets = set(targets)
    predictions, targets = _expand_duplicate_retrieved_docs(predictions, targets)

    all_docs = targets.union(predictions)
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(all_docs)}
    n_labels = max(len(doc_id_to_index), 2)  # sklearn.metrics.ndcg_score requires at least 2 labels
    y_true = np.zeros((1, n_labels), dtype=np.float32)
    y_score = np.zeros((1, n_labels), dtype=np.float32)
    for i, doc_id in enumerate(predictions):
        # "1 - i * eps" means we assign higher score to docs that are ranked higher,
        # but all scores are still approximately 1.
        y_score[0, doc_id_to_index[doc_id]] = 1 - i * eps
    for doc_id in targets:
        y_true[0, doc_id_to_index[doc_id]] = 1
    return y_score, y_true


def _ndcg_at_k_eval_fn(k):
    if not (isinstance(k, int) and k > 0):
        _logger.warning(
            f"Cannot calculate 'ndcg_at_k' for invalid parameter 'k'. "
            f"'k' should be a positive integer; found: {k}. Skipping metric logging."
        )
        return noop

    def _fn(predictions, targets):
        from sklearn.metrics import ndcg_score

        if not _validate_array_like_id_data(
            predictions, "ndcg_at_k", predictions_col_specifier
        ) or not _validate_array_like_id_data(targets, "ndcg_at_k", targets_col_specifier):
            return

        scores = []
        for ground_truth, retrieved in zip(targets, predictions):
            # 1. If no ground truth doc IDs are provided and no documents are retrieved,
            # the score is 1.
            if len(retrieved) == 0 and len(ground_truth) == 0:
                scores.append(1)  # no error is made
                continue
            # 2. If no ground truth doc IDs are provided and documents are retrieved,
            # the score is 0.
            # 3. If ground truth doc IDs are provided and no documents are retrieved,
            # the score is 0.
            if len(retrieved) == 0 or len(ground_truth) == 0:
                scores.append(0)
                continue

            # only include the top k retrieved chunks
            y_score, y_true = _prepare_row_for_ndcg(retrieved[:k], ground_truth)
            score = ndcg_score(y_true, y_score, k=len(retrieved[:k]), ignore_ties=True)
            scores.append(score)

        return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))

    return _fn


def _recall_at_k_eval_fn(k):
    if not (isinstance(k, int) and k > 0):
        _logger.warning(
            f"Cannot calculate 'recall_at_k' for invalid parameter 'k'. "
            f"'k' should be a positive integer; found: {k}. Skipping metric logging."
        )
        return noop

    def _fn(predictions, targets):
        if not _validate_array_like_id_data(
            predictions, "recall_at_k", predictions_col_specifier
        ) or not _validate_array_like_id_data(targets, "recall_at_k", targets_col_specifier):
            return

        scores = []
        for target, prediction in zip(targets, predictions):
            # only include the top k retrieved chunks
            ground_truth = set(target)
            retrieved = set(prediction[:k])
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


def _bleu_eval_fn(predictions, targets=None, metrics=None):
    # Validate input data
    if not _validate_text_data(targets, "bleu", targets_col_specifier):
        _logger.error(
            """Target validation failed.
            Ensure targets are valid for BLEU computation."""
        )
        return
    if not _validate_text_data(predictions, "bleu", predictions_col_specifier):
        _logger.error(
            """Prediction validation failed.
            Ensure predictions are valid for BLEU computation."""
        )
        return

    # Load BLEU metric
    try:
        bleu = _cached_evaluate_load("bleu")
    except Exception as e:
        _logger.warning(f"Failed to load 'bleu' metric (error: {e!r}), skipping metric logging.")
        return

    # Calculate BLEU scores for each prediction-target pair
    result = []
    invalid_indices = []

    for i, (prediction, target) in enumerate(zip(predictions, targets)):
        if len(target) == 0 or len(prediction) == 0:
            invalid_indices.append(i)
            result.append(0)  # Append 0 as a placeholder for invalid entries
            continue

        try:
            score = bleu.compute(predictions=[prediction], references=[[target]])
            result.append(score["bleu"])
        except Exception as e:
            _logger.warning(f"Failed to calculate BLEU for row {i} (error: {e!r}). Skipping.")
            result.append(0)  # Append 0 for consistency if an unexpected error occurs

    # Log warning for any invalid indices
    if invalid_indices:
        _logger.warning(
            f"BLEU score calculation skipped for the following indices "
            f"due to empty target or prediction: {invalid_indices}. "
            f"A score of 0 was appended for these entries."
        )

    # Return results
    if not result:
        _logger.warning("No BLEU scores were calculated due to input errors.")
        return

    return MetricValue(
        scores=result,
        aggregate_results=standard_aggregations(result),
    )
