# pylint: disable=line-too-long

"""
MLflow Pipelines is an opinionated framework for structuring MLOps workflows that simplifies and
standardizes machine learning application development and productionization. MLflow Pipelines
makes it easy for data scientists to follow best practices for creating production-ready ML
deliverables, allowing them to focus on developing excellent models. MLflow Pipelines also enables
ML engineers and DevOps teams to seamlessly deploy these models to production and incorporate them
into applications.

MLflow Pipelines provides production-quality :ref:`Pipeline Templates <pipeline-templates>` for
common ML problem types, such as regression & classification, and MLOps tasks, such as batch
scoring. Pipelines are structured as git repositories with YAML-based configuration files and
Python code, offering developers a declarative approach to ML application development that reduces
boilerplate.

MLflow Pipelines also implements a cache-aware executor for pipeline steps, ensuring that steps
are only executed when associated
:py:ref:`code or configurations <pipeline-repositories-key-concept>` have changed. This enables
data scientists, ML engineers, and DevOps teams to iterate very quickly within their domains of
expertise. MLflow offers |run() APIs| for executing pipelines, as well as an
|mlflow pipelines run CLI|.

For more information, see the :ref:`MLflow Pipelines Overview <pipelines>`.

.. |mlflow pipelines run CLI| replace:: :ref:`mlflow pipelines run <cli>` CLI
.. |run() APIs| replace:: :py:func:`run() <mlflow.pipelines.regression.v1.pipeline.RegressionPipeline.run>` APIs
"""

# pylint: enable=line-too-long

from mlflow.pipelines.pipeline import Pipeline

__all__ = ["Pipeline"]
