=========================
Extra MLflow Dependencies
=========================

When you `install the MLflow Python package <https://mlflow.org/docs/latest/quickstart.html#installing-mlflow>`_,
a set of core dependencies needed to use most MLflow functionality (tracking, projects, models APIs)
is also installed.

However, in order to use certain framework-specific MLflow APIs or configuration options,
you need to install additional, "extra" dependencies. For example, the model persistence APIs under
the ``mlflow.sklearn`` module require scikit-learn to be installed. Some of the most common MLflow
extra dependencies can be installed via ``pip install mlflow[extras]``.

The full set of extra dependencies are documented, along with the modules that depend on them,
in the following files:

* extra-ml-requirements.txt: ML libraries needed to use model persistence and inference APIs
* test-requirements.txt: Libraries required to use non-default artifact-logging and tracking server configurations
