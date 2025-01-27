"""
MLflow Recipes is a framework that enables you to quickly develop high-quality models and deploy
them to production. Compared to ad-hoc ML workflows, MLflow Recipes offers several major benefits:

- **Recipe templates**: `Predefined templates <../../recipes/index.html#recipe-templates>`_ for
  common ML tasks, such as `regression modeling <../../recipes/index.html#regression-template>`_,
  enable you to get started quickly and focus on building great models, eliminating the large amount
  of boilerplate code that is traditionally required to curate datasets, engineer features, train &
  tune models, and package models for production deployment.

- **Recipe engine**: The intelligent recipe execution engine accelerates model development by
  caching results from each step of the process and re-running the minimal set of steps as changes
  are made.

- **Production-ready structure**: The modular, git-integrated `recipe structure
  <../../recipes/index.html#recipe-templates-key-concept>`_ dramatically simplifies the handoff from
  development to production by ensuring that all model code, data, and configurations are easily
  reviewable and deployable by ML engineers.

For more information, see the `MLflow Recipes overview <../../recipes/index.html>`_.
"""

from mlflow.recipes.recipe import Recipe

__all__ = ["Recipe"]
