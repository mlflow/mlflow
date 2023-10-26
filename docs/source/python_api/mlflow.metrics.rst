mlflow.metrics
==============

The ``mlflow.metrics`` module helps you quantitatively and qualitatively measure your models. 

.. autoclass:: mlflow.metrics.EvaluationMetric

These :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` are used by the :py:func:`mlflow.evaluate()` API, either computed automatically depending on the ``model_type`` or specified via the ``extra_metrics`` parameter.

The following code demonstrates how to use :py:func:`mlflow.evaluate()` with an  :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`.

.. code-block:: python

    import mlflow
    from mlflow.metrics import EvaluationExample, answer_similarity

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
    answer_similarity_metric = answer_similarity(examples=[example])
    results = mlflow.evaluate(
        logged_model.model_uri,
        eval_df,
        targets="ground_truth",
        model_type="question-answering",
        extra_metrics=[answer_similarity_metric],
    )

Evaluation results are stored as :py:class:`MetricValue <mlflow.metrics.MetricValue>`. Aggregate results are logged to the MLflow run as metrics, while per-example results are logged to the MLflow run as artifacts in the form of an evaluation table.

.. autoclass:: mlflow.metrics.MetricValue

We provide the following builtin factory functions to create :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` for evaluating models. These metrics are computed automatically depending on the ``model_type``. For more information on the ``model_type`` parameter, see :py:func:`mlflow.evaluate()` API.

.. autofunction:: mlflow.metrics.mae

.. autofunction:: mlflow.metrics.mape

.. autofunction:: mlflow.metrics.max_error

.. autofunction:: mlflow.metrics.mse

.. autofunction:: mlflow.metrics.rmse

.. autofunction:: mlflow.metrics.r2_score

.. autofunction:: mlflow.metrics.precision_score

.. autofunction:: mlflow.metrics.recall_score

.. autofunction:: mlflow.metrics.f1_score

.. autofunction:: mlflow.metrics.ari_grade_level

.. autofunction:: mlflow.metrics.flesch_kincaid_grade_level

.. autofunction:: mlflow.metrics.rouge1

.. autofunction:: mlflow.metrics.rouge2

.. autofunction:: mlflow.metrics.rougeL

.. autofunction:: mlflow.metrics.rougeLsum

Retriever Specific Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following metrics are builtin metrics for the ``'retriever'`` model type, meaning it will be automatically calculated with a default ``k`` value of 3. 

It is recommended to use a static dataset (Pandas Dataframe or MLflow Pandas Dataset) containing 
columns for: input queries, retrieved relevant doc IDs, and ground-truth doc IDs. A "doc ID" is a 
string that uniquely identifies a document. All doc IDs should be entered as a tuple of doc ID 
strings.

The ``targets`` parameter should specify the column name of the ground-truth relevant doc IDs.

If you choose to use a static dataset, the ``predictions`` parameter should specify the column name 
of the retrieved relevant doc IDs. Alternatively, if you choose to specify a function for the 
``model`` parameter, the function should take a Pandas DataFrame as input and return a Pandas 
DataFrame with a column of retrieved relevant doc IDs, specified by the ``predictions`` parameter.

``k`` should be a positive integer specifying the number of retrieved doc IDs to consider for each 
input query. ``k`` defaults to 3. To use another ``k`` value, you have two options with the :py:func:`mlflow.evaluate` API:

1. ``evaluator_config={"k": 5}``
2. ``extra_metrics = [mlflow.metrics.precision_at_k(k=5), mlflow.metrics.precision_at_k(k=6), 
   mlflow.metrics.recall_at_k(4), mlflow.metrics.recall_at_k(5)]``

    Note that the ``k`` value in the ``evaluator_config`` will be ignored in this case. It is 
    recommended to remove the ``model_type`` as well, or else precision@3 will be calculated along 
    with precision@5, precision@6, recall@4, and recall@5.

.. autofunction:: mlflow.metrics.precision_at_k

.. autofunction:: mlflow.metrics.recall_at_k

.. autofunction:: mlflow.metrics.toxicity

.. autofunction:: mlflow.metrics.token_count

.. autofunction:: mlflow.metrics.latency

Users create their own :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` using the :py:func:`make_metric <mlflow.metrics.make_metric>` factory function

.. autofunction:: mlflow.metrics.make_metric

We provide the following pre-canned "intelligent" :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` for evaluating text models. These metrics use an LLM to evaluate the quality of a model's output text. Note that your use of a third party LLM service (e.g., OpenAI) for evaluation may be subject to and governed by the LLM service's terms of use. The following factory functions help you customize the intelligent metric to your use case.

.. autofunction:: mlflow.metrics.answer_similarity

.. autofunction:: mlflow.metrics.answer_correctness

.. autofunction:: mlflow.metrics.faithfulness

.. autofunction:: mlflow.metrics.answer_relevance

Users can also create their own LLM based :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` using the :py:func:`make_genai_metric <mlflow.metrics.make_genai_metric>` factory function.

.. autofunction:: mlflow.metrics.make_genai_metric

When using LLM based :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`, it is important to pass in an :py:class:`EvaluationExample <mlflow.metrics.EvaluationExample>`

.. autoclass:: mlflow.metrics.EvaluationExample

.. automodule:: mlflow.metrics
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: MetricValue, EvaluationMetric, make_metric, make_genai_metric, EvaluationExample, ari_grade_level, flesch_kincaid_grade_level, rouge1, rouge2, rougeL, rougeLsum, toxicity, answer_similarity, answer_correctness, faithfulness, answer_relevance, mae, mape, max_error, mse, rmse, r2_score, precision_score, recall_score, f1_score, token_count, latency
