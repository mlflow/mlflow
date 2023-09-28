mlflow.metrics
==============

The ``mlflow.metrics`` module helps you quantitatively and qualitatively measure your models. 

The :py:class:`MetricValue <mlflow.metrics.MetricValue>` is a ____.

.. autoclass:: mlflow.metrics.MetricValue

The :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` is a ____.

.. autoclass:: mlflow.metrics.EvaluationMetric

We provide the following default :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`:

.. autoclass:: mlflow.metrics.ari_grade_level
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: mlflow.metrics.flesch_kincaid_grade_level
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: mlflow.metrics.perplexity
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: mlflow.metrics.rouge1
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: mlflow.metrics.rouge2
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: mlflow.metrics.rougeL
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: mlflow.metrics.rougeLsum
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: mlflow.metrics.toxicity
    :members:
    :undoc-members:
    :show-inheritance:

The following pre-canned LLM based :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` are available:

.. autoclass:: mlflow.metrics.correctness

Users create their own EvaluationMetric using the :py:func:`make_metric <mlflow.metrics.make_metric>` factory function

.. autofunction:: mlflow.metrics.make_metric

Users can also create their own LLM based EvaluationMetric using the :py:func:`make_genai_metric <mlflow.metrics.make_genai_metric>` factory function

.. autofunction:: mlflow.metrics.make_genai_metric

When using LLM based EvaluationMetrics, it is important to pass in an :py:class:`EvaluationExample <mlflow.metrics.EvaluationExample>`

.. autoclass:: mlflow.metrics.EvaluationExample

.. automodule:: mlflow.metrics
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: MetricValue, EvaluationMetric, make_metric, make_genai_metric, EvaluationExample, ari_grade_level, flesch_kincaid_grade_level, perplexity, rouge1, rouge2, rougeL, rougeLsum, toxicity, correctness