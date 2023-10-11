mlflow.metrics
==============

The ``mlflow.metrics`` module helps you quantitatively and qualitatively measure your models. 

.. autoclass:: mlflow.metrics.EvaluationMetric

These :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` are used by the :py:func:`mlflow.evaluate()` API, either computed automatically depending on the ``model_type`` or specified via the ``extra_metrics`` parameter.

The following code demonstrates how to use :py:func:`mlflow.evaluate()` with an  :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`.

.. code-block:: python

    import mlflow
    from mlflow.metrics import EvaluationExample, correctness

    eval_df = pd.DataFrame(
        {
            "inputs": [
                "What is MLflow?",
            ],
            "ground_truth": [
                "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.",
            ],
        }
    )

    example = EvaluationExample(
        input="What is MLflow?",
        output="MLflow is an open-source platform for managing machine "
        "learning workflows, including experiment tracking, model packaging, "
        "versioning, and deployment, simplifying the ML lifecycle.",
        score=4,
        justification="The definition effectively explains what MLflow is "
        "its purpose, and its developer. It could be more concise for a 5-score.",
        grading_context={
            "ground_truth": "MLflow is an open-source platform for managing "
            "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
            "a company that specializes in big data and machine learning solutions. MLflow is "
            "designed to address the challenges that data scientists and machine learning "
            "engineers face when developing, training, and deploying machine learning models."
        },
    )
    correctness_metric = correctness(examples=[example])
    results = mlflow.evaluate(
        logged_model.model_uri,
        eval_df,
        targets="ground_truth",
        model_type="question-answering",
        extra_metrics=[correctness_metric],
    )

Evaluation results are stored as :py:class:`MetricValue <mlflow.metrics.MetricValue>`. Aggregate results are logged to the MLflow run as metrics, while per-example results are logged to the MLflow run as artifacts in the form of an evaluation table.

.. autoclass:: mlflow.metrics.MetricValue

We provide the following builtin :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` for evaluating models. These metrics are computed automatically depending on the ``model_type``. For more information on the ``model_type`` parameter, see :py:func:`mlflow.evaluate()` API.

.. autodata:: mlflow.metrics.mae
   :annotation:

.. autodata:: mlflow.metrics.mape
   :annotation:

.. autodata:: mlflow.metrics.max_error
   :annotation:

.. autodata:: mlflow.metrics.mse
   :annotation:

.. autodata:: mlflow.metrics.rmse
   :annotation:

.. autodata:: mlflow.metrics.r2_score
   :annotation:

.. autodata:: mlflow.metrics.precision_score
   :annotation:

.. autodata:: mlflow.metrics.recall_score
   :annotation:

.. autodata:: mlflow.metrics.f1_score
   :annotation:

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

We provide the following pre-canned "intelligent" :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` for evaluating text models. These metrics use an LLM to evaluate the quality of a model's output text. The following factory functions help you customize the intelligent metric to your use case.

.. autofunction:: mlflow.metrics.correctness

.. autofunction:: mlflow.metrics.strict_correctness

.. autofunction:: mlflow.metrics.relevance

Users can also create their own LLM based :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` using the :py:func:`make_genai_metric <mlflow.metrics.make_genai_metric>` factory function.

.. autofunction:: mlflow.metrics.make_genai_metric

When using LLM based :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`, it is important to pass in an :py:class:`EvaluationExample <mlflow.metrics.EvaluationExample>`

.. autoclass:: mlflow.metrics.EvaluationExample

.. automodule:: mlflow.metrics
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: MetricValue, EvaluationMetric, make_metric, make_genai_metric, EvaluationExample, ari_grade_level, flesch_kincaid_grade_level, perplexity, rouge1, rouge2, rougeL, rougeLsum, toxicity, correctness, strict_correctness, relevance, mae, mape, max_error, mse, rmse, r2_score, precision_score, recall_score, f1_score
