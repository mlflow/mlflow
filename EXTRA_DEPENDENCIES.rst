=========================
Extra MLflow Dependencies
=========================

After `installing MLflow <https://mlflow.org/docs/latest/quickstart.html#installing-mlflow>`_,
you may need to install additional packages in order to use specific MLflow modules or functionality.
For example, the model persistence APIs under the ``mlflow.tensorflow`` module require
TensorFlow to be installed.

Extra dependencies are documented, along with the modules that depend on them, in the following
files:

* extra-ml-requirements.txt: ML libraries needed to use model persistence and inference APIs
* small-requirements.txt, large-requirements.txt: Libraries required to use non-default
  artifact-logging and tracking server configurations
