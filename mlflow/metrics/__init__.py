import os

from mlflow.metrics import genai
from mlflow.metrics.base import MetricValue
from mlflow.metrics.genai.utils import _MIGRATION_GUIDE
from mlflow.metrics.metric_definitions import (
    _accuracy_eval_fn,
    _ari_eval_fn,
    _bleu_eval_fn,
    _f1_score_eval_fn,
    _flesch_kincaid_eval_fn,
    _mae_eval_fn,
    _mape_eval_fn,
    _max_error_eval_fn,
    _mse_eval_fn,
    _ndcg_at_k_eval_fn,
    _precision_at_k_eval_fn,
    _precision_eval_fn,
    _r2_score_eval_fn,
    _recall_at_k_eval_fn,
    _recall_eval_fn,
    _rmse_eval_fn,
    _rouge1_eval_fn,
    _rouge2_eval_fn,
    _rougeL_eval_fn,
    _rougeLsum_eval_fn,
    _toxicity_eval_fn,
)
from mlflow.models import (
    EvaluationMetric,
    make_metric,
)
from mlflow.utils.annotations import deprecated


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def latency() -> EvaluationMetric:
    """
    This function will create a metric for calculating latency. Latency is determined by the time
    it takes to generate a prediction for a given input. Note that computing latency requires
    each row to be predicted sequentially, which will likely slow down the evaluation process.
    """
    return make_metric(
        eval_fn=lambda x: MetricValue(),
        greater_is_better=False,
        name="latency",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def token_count(encoding: str = "cl100k_base") -> EvaluationMetric:
    """
    This function will create a metric for calculating token_count. Token count is calculated
    using tiktoken by using the `cl100k_base` tokenizer.

    Note: For air-gapped environments, you can set the TIKTOKEN_CACHE_DIR environment variable
    to specify a local cache directory for tiktoken to avoid downloading the tokenizer files.
    """

    def _token_count_eval_fn(predictions, targets=None, metrics=None):
        import tiktoken

        # ref: https://github.com/openai/tiktoken/issues/75
        # Only set TIKTOKEN_CACHE_DIR if not already set by user
        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            os.environ["TIKTOKEN_CACHE_DIR"] = ""
        enc = tiktoken.get_encoding(encoding)

        num_tokens = []
        for prediction in predictions:
            if isinstance(prediction, str):
                num_tokens.append(len(enc.encode(prediction)))
            else:
                num_tokens.append(None)

        return MetricValue(
            scores=num_tokens,
            aggregate_results={},
        )

    return make_metric(
        eval_fn=_token_count_eval_fn,
        greater_is_better=True,
        name="token_count",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def toxicity() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `toxicity`_ using the model
    `roberta-hate-speech-dynabench-r4`_, which defines hate as "abusive speech targeting
    specific group characteristics, such as ethnic origin, religion, gender, or sexual
    orientation."

    The score ranges from 0 to 1, where scores closer to 1 are more toxic. The default threshold
    for a text to be considered "toxic" is 0.5.

    Aggregations calculated for this metric:
        - ratio (of toxic input texts)

    .. _toxicity: https://huggingface.co/spaces/evaluate-measurement/toxicity
    .. _roberta-hate-speech-dynabench-r4: https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target
    """
    return make_metric(
        eval_fn=_toxicity_eval_fn,
        greater_is_better=False,
        name="toxicity",
        long_name="toxicity/roberta-hate-speech-dynabench-r4",
        version="v1",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def flesch_kincaid_grade_level() -> EvaluationMetric:
    """
    This function will create a metric for calculating `flesch kincaid grade level`_ using
    `textstat`_.

    This metric outputs a number that approximates the grade level needed to comprehend the text,
    which will likely range from around 0 to 15 (although it is not limited to this range).

    Aggregations calculated for this metric:
        - mean

    .. _flesch kincaid grade level:
        https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level
    .. _textstat: https://pypi.org/project/textstat/
    """
    return make_metric(
        eval_fn=_flesch_kincaid_eval_fn,
        greater_is_better=False,
        name="flesch_kincaid_grade_level",
        version="v1",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def ari_grade_level() -> EvaluationMetric:
    """
    This function will create a metric for calculating `automated readability index`_ using
    `textstat`_.

    This metric outputs a number that approximates the grade level needed to comprehend the text,
    which will likely range from around 0 to 15 (although it is not limited to this range).

    Aggregations calculated for this metric:
        - mean

    .. _automated readability index: https://en.wikipedia.org/wiki/Automated_readability_index
    .. _textstat: https://pypi.org/project/textstat/
    """
    return make_metric(
        eval_fn=_ari_eval_fn,
        greater_is_better=False,
        name="ari_grade_level",
        long_name="automated_readability_index_grade_level",
        version="v1",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def exact_match() -> EvaluationMetric:
    """
    This function will create a metric for calculating `accuracy`_ using sklearn.

    This metric only computes an aggregate score which ranges from 0 to 1.

    .. _accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    """
    return make_metric(
        eval_fn=_accuracy_eval_fn, greater_is_better=True, name="exact_match", version="v1"
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def rouge1() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `rouge1`_.

    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rouge1`_ uses unigram based scoring to calculate similarity.

    Aggregations calculated for this metric:
        - mean

    .. _rouge1: https://huggingface.co/spaces/evaluate-metric/rouge
    """
    return make_metric(
        eval_fn=_rouge1_eval_fn,
        greater_is_better=True,
        name="rouge1",
        version="v1",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def rouge2() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `rouge2`_.

    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rouge2`_ uses bigram based scoring to calculate similarity.

    Aggregations calculated for this metric:
        - mean

    .. _rouge2: https://huggingface.co/spaces/evaluate-metric/rouge
    """
    return make_metric(
        eval_fn=_rouge2_eval_fn,
        greater_is_better=True,
        name="rouge2",
        version="v1",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def rougeL() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `rougeL`_.

    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rougeL`_ uses unigram based scoring to calculate similarity.

    Aggregations calculated for this metric:
        - mean

    .. _rougeL: https://huggingface.co/spaces/evaluate-metric/rouge
    """
    return make_metric(
        eval_fn=_rougeL_eval_fn,
        greater_is_better=True,
        name="rougeL",
        version="v1",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def rougeLsum() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `rougeLsum`_.

    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rougeLsum`_ uses longest common subsequence based scoring to calculate similarity.

    Aggregations calculated for this metric:
        - mean

    .. _rougeLsum: https://huggingface.co/spaces/evaluate-metric/rouge
    """
    return make_metric(
        eval_fn=_rougeLsum_eval_fn,
        greater_is_better=True,
        name="rougeLsum",
        version="v1",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def precision_at_k(k) -> EvaluationMetric:
    """
    This function will create a metric for calculating ``precision_at_k`` for retriever models.

    This metric computes a score between 0 and 1 for each row representing the precision of the
    retriever model at the given ``k`` value. If no relevant documents are retrieved, the score is
    0, indicating that no relevant docs are retrieved. Let ``x = min(k, # of retrieved doc IDs)``.
    Then, in all other cases, the precision at k is calculated as follows:

        ``precision_at_k`` = (# of relevant retrieved doc IDs in top-``x`` ranked docs) / ``x``.
    """
    return make_metric(
        eval_fn=_precision_at_k_eval_fn(k),
        greater_is_better=True,
        name=f"precision_at_{k}",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def recall_at_k(k) -> EvaluationMetric:
    """
    This function will create a metric for calculating ``recall_at_k`` for retriever models.

    This metric computes a score between 0 and 1 for each row representing the recall ability of
    the retriever model at the given ``k`` value. If no ground truth doc IDs are provided and no
    documents are retrieved, the score is 1. However, if no ground truth doc IDs are provided and
    documents are retrieved, the score is 0. In all other cases, the recall at k is calculated as
    follows:

        ``recall_at_k`` = (# of unique relevant retrieved doc IDs in top-``k`` ranked docs) / (# of
        ground truth doc IDs)
    """
    return make_metric(
        eval_fn=_recall_at_k_eval_fn(k),
        greater_is_better=True,
        name=f"recall_at_{k}",
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def ndcg_at_k(k) -> EvaluationMetric:
    """
    This function will create a metric for evaluating `NDCG@k`_ for retriever models.

    NDCG score is capable of handling non-binary notions of relevance. However, for simplicity,
    we use binary relevance here. The relevance score for documents in the ground truth is 1,
    and the relevance score for documents not in the ground truth is 0.

    The NDCG score is calculated using sklearn.metrics.ndcg_score with the following edge cases
    on top of the sklearn implementation:

    1. If no ground truth doc IDs are provided and no documents are retrieved, the score is 1.
    2. If no ground truth doc IDs are provided and documents are retrieved, the score is 0.
    3. If ground truth doc IDs are provided and no documents are retrieved, the score is 0.
    4. If duplicate doc IDs are retrieved and the duplicate doc IDs are in the ground truth,
       they will be treated as different docs. For example, if the ground truth doc IDs are
       [1, 2] and the retrieved doc IDs are [1, 1, 1, 3], the score will be equivalent to
       ground truth doc IDs [10, 11, 12, 2] and retrieved doc IDs [10, 11, 12, 3].

    .. _NDCG@k: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
    """
    return make_metric(
        eval_fn=_ndcg_at_k_eval_fn(k),
        greater_is_better=True,
        name=f"ndcg_at_{k}",
    )


# General Regression Metrics
def mae() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `mae`_.

    This metric computes an aggregate score for the mean absolute error for regression.

    .. _mae: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    """
    return make_metric(
        eval_fn=_mae_eval_fn,
        greater_is_better=False,
        name="mean_absolute_error",
    )


def mse() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `mse`_.

    This metric computes an aggregate score for the mean squared error for regression.

    .. _mse: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    """
    return make_metric(
        eval_fn=_mse_eval_fn,
        greater_is_better=False,
        name="mean_squared_error",
    )


def rmse() -> EvaluationMetric:
    """
    This function will create a metric for evaluating the square root of `mse`_.

    This metric computes an aggregate score for the root mean absolute error for regression.

    .. _mse: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    """

    return make_metric(
        eval_fn=_rmse_eval_fn,
        greater_is_better=False,
        name="root_mean_squared_error",
    )


def r2_score() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `r2_score`_.

    This metric computes an aggregate score for the coefficient of determination. R2 ranges from
    negative infinity to 1, and measures the percentage of variance explained by the predictor
    variables in a regression.

    .. _r2_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    """
    return make_metric(
        eval_fn=_r2_score_eval_fn,
        greater_is_better=True,
        name="r2_score",
    )


def max_error() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `max_error`_.

    This metric computes an aggregate score for the maximum residual error for regression.

    .. _max_error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html
    """
    return make_metric(
        eval_fn=_max_error_eval_fn,
        greater_is_better=False,
        name="max_error",
    )


def mape() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `mape`_.

    This metric computes an aggregate score for the mean absolute percentage error for regression.

    .. _mape: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
    """
    return make_metric(
        eval_fn=_mape_eval_fn,
        greater_is_better=False,
        name="mean_absolute_percentage_error",
    )


# Binary Classification Metrics


def recall_score() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `recall`_ for classification.

    This metric computes an aggregate score between 0 and 1 for the recall of a classification task.

    .. _recall: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    return make_metric(eval_fn=_recall_eval_fn, greater_is_better=True, name="recall_score")


def precision_score() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `precision`_ for classification.

    This metric computes an aggregate score between 0 and 1 for the precision of
    classification task.

    .. _precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    return make_metric(eval_fn=_precision_eval_fn, greater_is_better=True, name="precision_score")


def f1_score() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `f1_score`_ for binary classification.

    This metric computes an aggregate score between 0 and 1 for the F1 score (F-measure) of a
    classification task. F1 score is defined as 2 * (precision * recall) / (precision + recall).

    .. _f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    return make_metric(eval_fn=_f1_score_eval_fn, greater_is_better=True, name="f1_score")


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def bleu() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `bleu`_.

    The BLEU scores range from 0 to 1, with higher scores indicating greater similarity to
    reference texts. BLEU considers n-gram precision and brevity penalty. While adding more
    references can boost the score, perfect scores are rare and not essential for effective
    evaluation.

    Aggregations calculated for this metric:
        - mean
        - variance
        - p90

    .. _bleu: https://huggingface.co/spaces/evaluate-metric/bleu
    """
    return make_metric(
        eval_fn=_bleu_eval_fn,
        greater_is_better=True,
        name="bleu",
        version="v1",
    )


__all__ = [
    "EvaluationMetric",
    "MetricValue",
    "make_metric",
    "flesch_kincaid_grade_level",
    "ari_grade_level",
    "exact_match",
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "toxicity",
    "mae",
    "mse",
    "rmse",
    "r2_score",
    "max_error",
    "mape",
    "recall_score",
    "precision_score",
    "f1_score",
    "token_count",
    "latency",
    "genai",
    "bleu",
]
