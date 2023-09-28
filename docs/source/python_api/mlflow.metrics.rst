mlflow.metrics
==============

The ``mlflow.metrics`` module helps you quantitatively and qualitatively measure your models. 

TODO fill this in

.. autoclass:: mlflow.metrics.MetricValue

An ``mlflow.metrics.EvaluationMetric`` is a ____.

.. autoclass:: mlflow.metrics.EvaluationMetric

We provide the following default EvaluationMetric:

.. autoclass:: mlflow.metrics.ari_grade_level
.. autoclass:: mlflow.metrics.flesch_kincaid_grade_level
.. autoclass:: mlflow.metrics.perplexity
.. autoclass:: mlflow.metrics.rouge1
.. autoclass:: mlflow.metrics.rouge2
.. autoclass:: mlflow.metrics.rougeL
.. autoclass:: mlflow.metrics.rougeLsum
.. autoclass:: mlflow.metrics.toxicity


The following pre-canned LLM based EvaluationMetric are available:

.. autoclass:: mlflow.metrics.correctness

Users create their own EvaluationMetric using the ``mlflow.metrics.make_metric`` factory function

.. autofunction:: mlflow.metrics.make_metric

Users can also create their own LLM based EvaluationMetric using the ``mlflow.metrics.make_genai_metric`` factory function

.. autofunction:: mlflow.metrics.make_genai_metric

When using LLM based EvaluationMetrics, it is important to pass in an ``mlflow.metrics.EvaluationExample``

.. autoclass:: mlflow.metrics.EvaluationExample

.. automodule:: mlflow.metrics
    :members:
    :undoc-members:
    :exclude-members: MetricValue, EvaluationMetric, make_metric, make_genai_metric, EvaluationExample, ari_grade_level, flesch_kincaid_grade_level, perplexity, rouge1, rouge2, rougeL, rougeLsum, toxicity, correctness