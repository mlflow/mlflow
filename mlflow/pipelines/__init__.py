# pylint: disable=line-too-long

"""
MLflow Pipelines is an opinionated framework for structuring MLOps workflows that simplifies and
standardizes machine learning application development and productionization. MLflow Pipelines
makes it easy for data scientists to follow best practices for creating production-ready ML
deliverables, allowing them to focus on utilizing their expert knowledge to develop excellent
models. MLflow Pipelines also enables ML engineers and DevOps teams to seamlessly deploy these
models to production and incorporate them into applications.

MLflow Pipelines provides predefined production-quality pipeline templates for common ML problem
types, such as regression & classification, and MLOps tasks, such as batch scoring. Pipelines are
structured as git repositories with YAML-based configuration files and Python code, offering
developers a declarative approach to ML application development that reduces boilerplate.

MLflow Pipelines also implements a cache-aware executor for pipeline steps, ensuring that steps
are only executed when associated :py:ref:`code or configurations <mlflow-pipeline-layout-example>`
have changed. This enables data scientists, ML engineers, and DevOps teams to iterate very quickly
within their domains of expertise. MLflow offers |run() APIs| for executing pipelines, as well as
an |mlflow pipelines run CLI|.

.. _mlflow-pipeline-layout-example:

.. rubric:: Pipeline layout

The following example pipeline repository layout includes the main components of the
|MLflow Regression Pipeline repository|, corresponding to the
:py:ref:`MLflow Regression Pipeline <mlflow-regression-pipeline>`.

::

  ├── pipeline.yaml
  ├── steps
  │   ├── ingest.py
  │   ├── split.py
  │   ├── transform.py
  │   ├── train.py
  │   ├── custom_metrics.py
  ├── profiles
  │   ├── local.yaml
  │   ├── databricks.yaml
  ├── tests
  │   ├── ingest_test.py
  │   ├── ...
  │   ├── train_test.py
  │   ├── ...

The main components of the pipeline layout, which are common across all pipelines, are:

    - ``pipeline.yaml``: The main pipeline configuration file that declaratively defines the
      attributes and behavior of each pipeline step, such as the input dataset to use for training
      a model or the performance criteria for promoting a model to production. For example,
      see the |pipeline.yaml| configuration file from the |MLflow Regression Pipeline repository|.

    - ``steps``: A directory containing Python code modules used by the pipeline steps. For example,
      the |MLflow Regression Pipeline repository| defines the estimator type and parameters to use
      when training a model in |steps/train.py| and defines custom metric computations in
      |steps/custom_metrics.py|.

    - ``profiles``: A directory containing customizations for the configurations defined in
      ``pipeline.yaml``. For example, the |MLflow Regression Pipeline repository|
      defines a |local profile| that |customizes the dataset used for local model development|
      and |specifies a local MLflow Tracking store for logging model content|
      The |MLflow Regression Pipeline repository| also defines a |databricks profile| for
      development on Databricks.

      .. code-block:: python
        :caption: Profile usage example

        import os
        from mlflow.pipelines import Pipeline

        os.chdir("~/mlp-regression-template")
        pipeline_with_local_profile = Pipeline(profile="local")
        pipeline_with_databricks_profile = Pipeline(profile="databricks")

    - ``tests``: A directory containing Python test code for pipeline steps. For example, the
      |MLflow Regression Pipeline repository| implements tests for the transformer and the estimator
      defined in the respective ``steps/transform.py`` and ``steps/train.py`` modules.

.. |mlflow pipelines run CLI| replace:: :ref:`mlflow pipelines run <cli>` CLI
.. |run() APIs| replace:: :py:func:`run() <mlflow.pipelines.regression.v1.pipeline.RegressionPipeline.run>` APIs
.. |pipeline.yaml| replace:: `pipeline.yaml <https://github.com/mlflow/mlp-regression-template/blob/main/pipeline.yaml>`__
.. |steps/train.py| replace:: `steps/train.py <https://github.com/mlflow/mlp-regression-template/blob/main/steps/train.py>`__
.. |steps/custom_metrics.py| replace:: `steps/custom_metrics.py <https://github.com/mlflow/mlp-regression-template/blob/main/steps/custom_metrics.py>`__
.. |MLflow Regression Pipeline repository| replace:: `MLflow Regression Pipeline repository <https://github.com/mlflow/mlp-regression-template>`__
.. |local profile| replace:: `profiles/local.yaml profile <https://github.com/mlflow/mlp-regression-template/blob/main/profiles/local.yaml>`__
.. |databricks profile| replace:: `profiles/databricks.yaml profile <https://github.com/mlflow/mlp-regression-template/blob/main/profiles/databricks.yaml>`__
.. |customizes the dataset used for local model development| replace:: `customizes the dataset used for local model development <https://github.com/mlflow/mlp-regression-template/blob/1f6e1b28acac23cc47621138ab2b1e4aed1654a1/profiles/local.yaml#L7>`__
.. |specifies a local MLflow Tracking store for logging model content| replace:: `specifies a local MLflow Tracking store for logging model content <https://github.com/mlflow/mlp-regression-template/blob/1f6e1b28acac23cc47621138ab2b1e4aed1654a1/profiles/local.yaml#L1-L4>`__
"""

# pylint: enable=line-too-long

from mlflow.pipelines.pipeline import Pipeline

__all__ = ["Pipeline"]
