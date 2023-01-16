=======================================================================
MLflow Skinny: A Lightweight Machine Learning Lifecycle Platform Client
=======================================================================

MLflow Skinny is a lightweight MLflow package without SQL storage, server, UI, or data science dependencies.
MLflow Skinny supports:

* Tracking operations (logging / loading / searching params, metrics, tags + logging / loading artifacts)
* Model registration, search, artifact loading, and transitions
* Execution of GitHub projects within notebook & against a remote target.

Additional dependencies can be installed to leverage the full feature set of MLflow. For example:

* To use the `mlflow.sklearn` component of MLflow Models, install `scikit-learn`, `numpy` and `pandas`.
* To use SQL-based metadata storage, install `sqlalchemy`, `alembic`, and `sqlparse`.
* To use serving-based features, install `flask` and `pandas`.

