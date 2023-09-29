mlflow.metrics
==============

The ``mlflow.metrics`` module helps you quantitatively and qualitatively measure your models. These :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`s are used by the :py:func:`mlflow.evaluate()` API, either computed automatically depending on the ``model_type`` or specified via the ``custom_metrics`` parameter. Evaluation results are stored as :py:class:`MetricValue <mlflow.metrics.MetricValue>` and are logged to the MLflow run.

.. autoclass:: mlflow.metrics.MetricValue

.. autoclass:: mlflow.metrics.EvaluationMetric

We provide the following default :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` for evaluating text models. 

.. autodata:: mlflow.metrics.ari_grade_level
   :annotation:

.. autodata:: mlflow.metrics.flesch_kincaid_grade_level
   :annotation:

.. autodata:: mlflow.metrics.perplexity
   :annotation:

.. autodata:: mlflow.metrics.rouge1
   :annotation:

.. autodata:: mlflow.metrics.rouge2
   :annotation:

.. autodata:: mlflow.metrics.rougeL
   :annotation:

.. autodata:: mlflow.metrics.rougeLsum
   :annotation:

.. autodata:: mlflow.metrics.toxicity
   :annotation:

Users create their own :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` using the :py:func:`make_metric <mlflow.metrics.make_metric>` factory function

.. autofunction:: mlflow.metrics.make_metric

We provide the following pre-canned "intelligent" :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`s for evaluating text models. These metrics use an LLM to evaluate the quality of a model's output text. These factory functions help you customize the intelligent metric to your use case.

.. autofunction:: mlflow.metrics.correctness

.. autofunction:: mlflow.metrics.relevance

Users can also create their own LLM based :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` using the :py:func:`make_genai_metric <mlflow.metrics.make_genai_metric>` factory function

.. autofunction:: mlflow.metrics.make_genai_metric

When using LLM based :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`s, it is important to pass in an :py:class:`EvaluationExample <mlflow.metrics.EvaluationExample>`

.. autoclass:: mlflow.metrics.EvaluationExample

.. automodule:: mlflow.metrics
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: MetricValue, EvaluationMetric, make_metric, make_genai_metric, EvaluationExample, ari_grade_level, flesch_kincaid_grade_level, perplexity, rouge1, rouge2, rougeL, rougeLsum, toxicity, correctness, relevance