.. _llm-tracking:

=====================
MLflow LLM Tracking
=====================

The MLflow LLM Tracking component is an API and UI for logging LLM inputs, outputs and prompts
when running your machine learning code and for later visualizing the results.

.. contents:: Table of Contents
  :local:
  :depth: 2

.. _llm-tracking-concepts:

Concepts
==========

MLflow LLM Tracking is organized around the concept of *runs*, which are executions of some piece of
data science code. Each run records the following information:

Parameters
    Key-value input parameters of your choice. Both keys and values are strings. These could be LLM
    parameters like top_k, temperature, etc.

Metrics
    Key-value metrics, where the value is numeric. Each metric can be updated throughout the
    course of the run (for example, to track how your model's loss function is converging), and
    MLflow records and lets you visualize the metric's full history.

Predictions
    For offline evaluation, you can log predictions for your model by passing in inputs, outputs
    and prompts. These predictions are logged as csv as part of MLflow artifacts.

Artifacts
    Output files in any format. For example, you can record images (for example, PNGs), models
    (for example, a pickled openai model), and data files (for example, a
    `Parquet <https://parquet.apache.org/>`_ file) as artifacts.

You can optionally organize runs into *experiments*, which group and compare together runs for a
specific task. You can create an experiment using the ``mlflow experiments`` CLI, with
:py:func:`mlflow.create_experiment`, or using the corresponding REST parameters. The MLflow API and
UI let you create and search for experiments.

Once your runs have been recorded, you can query them and compare predictions using the :ref:`tracking_ui`.

.. _how_llm_predictions_recorded:

How LLM Tracking Information Recorded
=======================================
Parameters: :py:func:`mlflow.log_param` logs a single key-value param in the currently active run. The key and
value are both strings. Use :py:func:`mlflow.log_params` to log multiple params at once.

Metrics: :py:func:`mlflow.log_metric` logs a single key-value metric. The value must always be a number.
MLflow remembers the history of values for each metric. Use :py:func:`mlflow.log_metrics` to log
multiple metrics at once.

Predictions: :py:func:`mlflow.llm.log_predictions` logs inputs, outputs and prompts. Inputs and prompts could either
be a list of strings or list of dict where as output would be a list of strings.

Artifacts: :py:func:`mlflow.log_artifact` logs a local file or directory as an artifact, optionally taking an
``artifact_path`` to place it in within the run's artifact URI. Run artifacts can be organized into
directories, so you can place the artifact in a directory this way.

.. _where_llm_tracking_information_are_recorded:

Where LLM Tracking Information Are Recorded
=============================================
All the tracking information is recorded as part of MLflow Experiment run.


