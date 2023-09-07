from mlflow.metrics.base import (
    EvaluationExample,
    MetricValue,
)
from mlflow.metrics.utils import (
    make_genai_metric,
)
from mlflow.metrics.utils.metric_definitions import (
    _accuracy_eval_fn,
    _ari_eval_fn,
    _flesch_kincaid_eval_fn,
    _perplexity_eval_fn,
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

# general text metrics

toxicity = make_metric(
    eval_fn=_toxicity_eval_fn,
    greater_is_better=False,
    name="toxicity",
    long_name="toxicity/roberta-hate-speech-dynabench-r4",
    version="v1",
)
"""
A metric for evaluating `toxicity`_ using the model `roberta-hate-speech-dynabench-r4`_, 
which defines hate as "abusive speech targeting specific group characteristics, such as 
ethnic origin, religion, gender, or sexual orientation."

The score ranges from 0 to 1, where scores closer to 1 are more toxic. The default threshold 
for a text to be considered "toxic" is 0.5.

Aggregations calculated for this metric:
    - ratio (of toxic input texts)

.. _toxicity: https://huggingface.co/spaces/evaluate-measurement/toxicity
.. _roberta-hate-speech-dynabench-r4: https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target
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

Aggregations calculated for this metric:
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

Aggregations calculated for this metric:
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

Aggregations calculated for this metric:
    - mean

.. _automated readability index: https://en.wikipedia.org/wiki/Automated_readability_index
.. _textstat: https://pypi.org/project/textstat/
"""

# question answering metrics

accuracy = make_metric(
    eval_fn=_accuracy_eval_fn, greater_is_better=True, name="exact_match", version="v1"
)
"""
A metric for calculating `accuracy`_ using sklearn.

This metric only computes an aggregate score which ranges from 0 to 1.

.. _accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
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

Aggregations calculated for this metric:
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

Aggregations calculated for this metric:
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

Aggregations calculated for this metric:
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

Aggregations calculated for this metric:
    - mean

.. _rougeLsum: https://huggingface.co/spaces/evaluate-metric/rouge
"""

__all__ = [
    "EvaluationExample",
    "EvaluationMetric",
    "MetricValue",
    "make_metric",
    "perplexity",
    "flesch_kincaid_grade_level",
    "ari_grade_level",
    "accuracy",
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "toxicity",
    "make_genai_metric",
]
