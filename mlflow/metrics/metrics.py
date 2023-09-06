import logging

import pandas as pd

from mlflow.metrics.base import MetricValue
from mlflow.models import make_metric

_logger = logging.getLogger(__name__)


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
    return MetricValue(scores=scores, aggregate_results={"ratio": toxicity_ratio})


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
    results = perplexity.compute(predictions=predictions, model_id="gpt2")
    return MetricValue(
        scores=results["perplexities"], aggregate_results={"mean": results["mean_perplexity"]}
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
    return MetricValue(scores=scores, aggregate_results={"mean": sum(scores) / len(scores)})


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
    return MetricValue(scores=scores, aggregate_results={"mean": sum(scores) / len(scores)})


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
        )
        aggregates = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge1"],
            use_aggregator=True,
        )
        return MetricValue(
            scores=scores["rouge1"], aggregate_results={"mean": aggregates["rouge1"]}
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
        )
        aggregates = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge2"],
            use_aggregator=True,
        )
        return MetricValue(
            scores=scores["rouge2"], aggregate_results={"mean": aggregates["rouge2"]}
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
        )
        aggregates = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeL"],
            use_aggregator=True,
        )
        return MetricValue(
            scores=scores["rougeL"], aggregate_results={"mean": aggregates["rougeL"]}
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
        )
        aggregates = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeLsum"],
            use_aggregator=True,
        )
        return MetricValue(
            scores=scores["rougeLsum"], aggregate_results={"mean": aggregates["rougeLsum"]}
        )


# general text metrics

toxicity = make_metric(
    eval_fn=_toxicity_eval_fn,
    greater_is_better=False,
    name="toxicity",
    long_name="toxicity/roberta-hate-speech-dynabench-r4",
    version="v1",
)
"""
A metric for calculating `accuracy`_ using sklearn.
    
This metric only computes an aggregate score which ranges from 0 to 1.

.. _accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
"""

perplexity = make_metric(
    eval_fn=_perplexity_eval_fn,
    greater_is_better=False,
    name="perplexity",
    long_name="perplexity/gpt2",
    version="v1",
)
"""
A metric for evaluating `perplexity`_ using the model gpt2.

The score ranges from 0 to infinity, where a lower score means that the model is better at 
predicting the given text and a higher score means that the model is not likely to predict the text.

The following aggregations are calculated for this metric:
    - mean

.. _perplexity: https://huggingface.co/spaces/evaluate-metric/perplexity
"""

flesch_kincaid_grade_level = make_metric(
    eval_fn=_flesch_kincaid_eval_fn,
    greater_is_better=False,
    name="flesch_kincaid_grade_level",
    version="v1",
)
"""
A metric for calculating `flesch kincaid grade level`_ using `textstat`_.
    
This metric outputs a number that approximates the grade level needed to comprehend the text, which
will likely range from around 0 to 15 (although it is not limited to this range).

The following aggregations are calculated for this metric:
    - mean

.. _flesch kincaid grade level:
    https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level
.. _textstat: https://pypi.org/project/textstat/
"""

ari_grade_level = make_metric(
    eval_fn=_ari_eval_fn,
    greater_is_better=False,
    name="ari_grade_level",
    long_name="automated_readability_index_grade_level",
    version="v1",
)
"""
A metric for calculating `automated readability index`_ using `textstat`_.
    
This metric outputs a number that approximates the grade level needed to comprehend the text, which
will likely range from around 0 to 15 (although it is not limited to this range).

The following aggregations are calculated for this metric:
    - mean

.. _automated readability index: https://en.wikipedia.org/wiki/Automated_readability_index
.. _textstat: https://pypi.org/project/textstat/
"""

# question answering metrics
accuracy = make_metric(
    eval_fn=_accuracy_eval_fn, greater_is_better=True, name="exact_match", version="v1"
)
"""
A metric for evaluating toxicity.
EvaluationMetric
"""

# text summarization metrics
rouge1 = make_metric(
    eval_fn=_rouge1_eval_fn,
    greater_is_better=True,
    name="rouge1",
    version="v1",
)
"""
A metric for evaluating `rouge1`_.
    
The score ranges from 0 to 1, where a higher score indicates higher similarity.
`rouge1`_ uses unigram based scoring to calculate similarity.

The following aggregations are calculated for this metric:
    - mean

.. _rouge1: https://huggingface.co/spaces/evaluate-metric/rouge
"""

rouge2 = make_metric(
    eval_fn=_rouge2_eval_fn,
    greater_is_better=True,
    name="rouge2",
    version="v1",
)
"""
A metric for evaluating `rouge2`_.
    
The score ranges from 0 to 1, where a higher score indicates higher similarity.
`rouge2`_ uses bigram based scoring to calculate similarity.

The following aggregations are calculated for this metric:
    - mean

.. _rouge2: https://huggingface.co/spaces/evaluate-metric/rouge
"""

rougeL = make_metric(
    eval_fn=_rougeL_eval_fn,
    greater_is_better=True,
    name="rougeL",
    version="v1",
)
"""
A metric for evaluating `rougeL`_.
    
The score ranges from 0 to 1, where a higher score indicates higher similarity.
`rougeL`_ uses unigram based scoring to calculate similarity.

The following aggregations are calculated for this metric:
    - mean

.. _rougeL: https://huggingface.co/spaces/evaluate-metric/rouge
"""

rougeLsum = make_metric(
    eval_fn=_rougeLsum_eval_fn,
    greater_is_better=True,
    name="rougeLsum",
    version="v1",
)
"""
A metric for evaluating `rougeLsum`_.
    
The score ranges from 0 to 1, where a higher score indicates higher similarity.
`rougeLsum`_ uses longest common subsequence based scoring to calculate similarity.

The following aggregations are calculated for this metric:
    - mean

.. _rougeLsum: https://huggingface.co/spaces/evaluate-metric/rouge
"""
