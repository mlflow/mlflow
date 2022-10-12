# pylint: disable=line-too-long

"""
MLflow Pipelines is a framework that enables you to quickly develop high-quality models and deploy
them to production. Compared to ad-hoc ML workflows, MLflow Pipelines offers several major benefits:

- **Pipeline templates**: :ref:`Predefined templates <pipeline-templates>` for common ML tasks,
  such as :ref:`regression modeling <regression-template>`, enable you to get started quickly and
  focus on building great models, eliminating the large amount of boilerplate code that is
  traditionally required to curate datasets, engineer features, train & tune models, and package
  models for production deployment.

- **Pipeline engine**: The intelligent pipeline execution engine accelerates model development by
  caching results from each step of the process and re-running the minimal set of steps as changes
  are made.

- **Production-ready structure**: The modular, git-integrated :ref:`pipeline structure
  <pipeline-templates-key-concept>` dramatically simplifies the handoff from development to
  production by ensuring that all model code, data, and configurations are easily reviewable and
  deployable by ML engineers.

For more information, see the :ref:`MLflow Pipelines overview <pipelines>`.
"""

# pylint: enable=line-too-long

from mlflow.pipelines.pipeline import Pipeline

__all__ = ["Pipeline"]
