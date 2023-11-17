mlflow.metrics
==============

The ``mlflow.metrics`` module helps you quantitatively and qualitatively measure your models. 

.. autoclass:: mlflow.metrics.EvaluationMetric

These :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` are used by the :py:func:`mlflow.evaluate()` API, either computed automatically depending on the ``model_type`` or specified via the ``extra_metrics`` parameter.

The following code demonstrates how to use :py:func:`mlflow.evaluate()` with an  :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`.

.. code-block:: python

    import mlflow
    from mlflow.metrics.genai import EvaluationExample, answer_similarity

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

Information about how an :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` is calculated, such as the grading prompt used is available via the ``metric_details`` property.

.. code-block:: python

    import mlflow
    from mlflow.metrics.genai import relevance

    my_relevance_metric = relevance()
    print(my_relevance_metric.metric_details)

Evaluation results are stored as :py:class:`MetricValue <mlflow.metrics.MetricValue>`. Aggregate results are logged to the MLflow run as metrics, while per-example results are logged to the MLflow run as artifacts in the form of an evaluation table.

.. autoclass:: mlflow.metrics.MetricValue

We provide the following builtin factory functions to create :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` for evaluating models. These metrics are computed automatically depending on the ``model_type``. For more information on the ``model_type`` parameter, see :py:func:`mlflow.evaluate()` API.

Regressor Metrics
-----------------

.. autofunction:: mlflow.metrics.mae

.. autofunction:: mlflow.metrics.mape

.. autofunction:: mlflow.metrics.max_error

.. autofunction:: mlflow.metrics.mse

.. autofunction:: mlflow.metrics.rmse

.. autofunction:: mlflow.metrics.r2_score

Classifier Metrics
------------------

.. autofunction:: mlflow.metrics.precision_score

.. autofunction:: mlflow.metrics.recall_score

.. autofunction:: mlflow.metrics.f1_score

Text Metrics
------------

.. autofunction:: mlflow.metrics.ari_grade_level

.. autofunction:: mlflow.metrics.flesch_kincaid_grade_level

Question Answering Metrics
---------------------------

Includes all of the above **Text Metrics** as well as the following:

.. autofunction:: mlflow.metrics.exact_match

.. autofunction:: mlflow.metrics.rouge1

.. autofunction:: mlflow.metrics.rouge2

.. autofunction:: mlflow.metrics.rougeL

.. autofunction:: mlflow.metrics.rougeLsum

.. autofunction:: mlflow.metrics.toxicity

.. autofunction:: mlflow.metrics.token_count

.. autofunction:: mlflow.metrics.latency

Retriever Metrics
-----------------

The following metrics are built-in metrics for the ``'retriever'`` model type, meaning they will be 
automatically calculated with a default ``retriever_k`` value of 3. 

To evaluate document retrieval models, it is recommended to use a dataset with the following 
columns:

- Input queries
- Retrieved relevant doc IDs
- Ground-truth doc IDs

Alternatively, you can also provide a function through the ``model`` parameter to represent 
your retrieval model. The function should take a Pandas DataFrame containing input queries and 
ground-truth relevant doc IDs, and return a DataFrame with a column of retrieved relevant doc IDs.

A "doc ID" is a string that uniquely identifies a document. Each row of the retrieved and 
ground-truth doc ID columns should consist of a list of doc IDs.

Parameters:

- ``targets``: A string specifying the column name of the ground-truth relevant doc IDs
- ``predictions``: A string specifying the column name of the retrieved relevant doc IDs in either 
  the static dataset or the Dataframe returned by the ``model`` function
- ``retriever_k``: A positive integer specifying the number of retrieved docs IDs to consider for 
  each input query. ``retriever_k`` defaults to 3. You can change ``retriever_k`` by using the 
  :py:func:`mlflow.evaluate` API:

    1. .. code-block:: python

        # with a model and using `evaluator_config`
        mlflow.evaluate(
            model=retriever_function,
            data=data,
            targets="ground_truth",
            model_type="retriever",
            evaluators="default",
            evaluator_config={"retriever_k": 5}
        )
    2. .. code-block:: python

        # with a static dataset and using `extra_metrics`
        mlflow.evaluate(
            data=data,
            predictions="retrieved_docs",
            targets="ground_truth_docs",
            predictions="predictions_param",
            targets="targets_param",
            model_type="retriever",
            extra_metrics = [
                mlflow.metrics.precision_at_k(5),
                mlflow.metrics.precision_at_k(6),
                mlflow.metrics.recall_at_k(5),
                mlflow.metrics.ndcg_at_k(5)
            ]   
        )
    
    NOTE: In the 2nd method, it is recommended to omit the ``model_type`` as well, or else 
    ``precision@3`` and ``recall@3`` will be  calculated in  addition to ``precision@5``, 
    ``precision@6``, ``recall@5``, and ``ndcg_at_k@5``.

.. autofunction:: mlflow.metrics.precision_at_k

.. autofunction:: mlflow.metrics.recall_at_k

.. autofunction:: mlflow.metrics.ndcg_at_k

Users create their own :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>` using the :py:func:`make_metric <mlflow.metrics.make_metric>` factory function

.. autofunction:: mlflow.metrics.make_metric

.. automodule:: mlflow.metrics
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: MetricValue, EvaluationMetric, make_metric, EvaluationExample, ari_grade_level, flesch_kincaid_grade_level, exact_match, rouge1, rouge2, rougeL, rougeLsum, toxicity, answer_similarity, answer_correctness, faithfulness, answer_relevance, mae, mape, max_error, mse, rmse, r2_score, precision_score, recall_score, f1_score, token_count, latency, precision_at_k, recall_at_k, ndcg_at_k

Generative AI Metrics
---------------------

We also provide generative AI ("genai") :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`\s for evaluating text models. These metrics use an LLM to evaluate the quality of a model's output text. Note that your use of a third party LLM service (e.g., OpenAI) for evaluation may be subject to and governed by the LLM service's terms of use. The following factory functions help you customize the intelligent metric to your use case.

.. automodule:: mlflow.metrics.genai
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: EvaluationExample, make_genai_metric

You can also create your own generative AI :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`\s using the :py:func:`make_genai_metric <mlflow.metrics.genai.make_genai_metric>` factory function.

.. autofunction:: mlflow.metrics.genai.make_genai_metric

When using generative AI :py:class:`EvaluationMetric <mlflow.metrics.EvaluationMetric>`\s, it is important to pass in an :py:class:`EvaluationExample <mlflow.metrics.genai.EvaluationExample>`

.. autoclass:: mlflow.metrics.genai.EvaluationExample

Users must set the appropriate environment variables for the LLM service they are using for 
evaluation. For example, if you are using OpenAI's API, you must set the ``OPENAI_API_KEY`` 
environment variable. If using Azure OpenAI, you must also set the ``OPENAI_API_TYPE``, 
``OPENAI_API_VERSION``, ``OPENAI_API_BASE``, and ``OPENAI_DEPLOYMENT_NAME`` environment variables. 
See `Azure OpenAI documentation <https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints>`_
Users do not need to set these environment variables if they are using a gateway route.