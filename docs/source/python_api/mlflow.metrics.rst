mlflow.metrics
==============

.. automodule:: mlflow.metrics
    :members:
    :undoc-members:
    :show-inheritance:

.. py:data:: mlflow.metrics.accuracy
    :type: mlflow.metrics.EvaluationMetric

    A metric for calculating `accuracy`_ using sklearn.
    
    This metric only computes an aggregate score which ranges from 0 to 1.
    
    .. _accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

.. py:data:: mlflow.metrics.ari_grade_level
    :type: mlflow.metrics.EvaluationMetric

    A metric for calculating `automated readability index`_ using `textstat`_.
    
    This metric outputs a number that approximates the grade level needed to comprehend the text, which
    will likely range from around 0 to 15 (although it is not limited to this range).
    
    The following aggregations are calculated for this metric:
        - mean
    
    .. _automated readability index: https://en.wikipedia.org/wiki/Automated_readability_index
    .. _textstat: https://pypi.org/project/textstat/

.. py:data:: mlflow.metrics.flesch_kincaid_grade_level
    :type: mlflow.metrics.EvaluationMetric

    A metric for calculating `flesch kincaid grade level`_ using `textstat`_.
    
    This metric outputs a number that approximates the grade level needed to comprehend the text, which
    will likely range from around 0 to 15 (although it is not limited to this range).
    
    The following aggregations are calculated for this metric:
        - mean
    
    .. _flesch kincaid grade level: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level
    .. _textstat: https://pypi.org/project/textstat/

.. py:data:: mlflow.metrics.perplexity
    :type: mlflow.metrics.EvaluationMetric

    A metric for evaluating `perplexity`_ using the model gpt2.
    
    The score ranges from 0 to infinity, where a lower score means that the model is better at 
    predicting the given text and a higher score means that the model is not likely to predict the text.
    
    The following aggregations are calculated for this metric:
        - mean
    
    .. _perplexity: https://huggingface.co/spaces/evaluate-metric/perplexity

.. py:data:: mlflow.metrics.rouge1
    :type: mlflow.metrics.EvaluationMetric

    A metric for evaluating `rouge1`_.
    
    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rouge1`_ uses unigram based scoring to calculate similarity.
    
    The following aggregations are calculated for this metric:
        - mean
    
    .. _rouge1: https://huggingface.co/spaces/evaluate-metric/rouge

.. py:data:: mlflow.metrics.rouge2
    :type: mlflow.metrics.EvaluationMetric

    A metric for evaluating `rouge2`_.
    
    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rouge2`_ uses bigram based scoring to calculate similarity.
    
    The following aggregations are calculated for this metric:
        - mean
    
    .. _rouge2: https://huggingface.co/spaces/evaluate-metric/rouge

.. py:data:: mlflow.metrics.rougeL
    :type: mlflow.metrics.EvaluationMetric

    A metric for evaluating `rougeL`_.
    
    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rougeL`_ uses unigram based scoring to calculate similarity.
    
    The following aggregations are calculated for this metric:
        - mean
    
    .. _rougeL: https://huggingface.co/spaces/evaluate-metric/rouge

.. py:data:: mlflow.metrics.rougeLsum
    :type: mlflow.metrics.EvaluationMetric

    A metric for evaluating `rougeLsum`_.
    
    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rougeLsum`_ uses longest common subsequence based scoring to calculate similarity.
    
    The following aggregations are calculated for this metric:
        - mean
    
    .. _rougeLsum: https://huggingface.co/spaces/evaluate-metric/rouge

.. py:data:: mlflow.metrics.toxicity
    :type: mlflow.metrics.EvaluationMetric

    A metric for evaluating `toxicity`_ using the model `roberta-hate-speech-dynabench-r4`_, which defines
    hate as "abusive speech targeting specific group characteristics, such as ethnic origin, religion, gender, 
    or sexual orientation."
    
    The score ranges from 0 to 1, where scores closer to 1 are more toxic. The default threshold for a text
    to be considered "toxic" is 0.5.
    
    The following aggregations are calculated for this metric:
        - ratio (of toxic input texts)
    
    .. _toxicity: https://huggingface.co/spaces/evaluate-measurement/toxicity
    .. _roberta-hate-speech-dynabench-r4: https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target
