=======================================================================
MLflow Skinny: A Lightweight Machine Learning Lifecycle Platform Client
=======================================================================

MLflow Skinny is a lightweight MLflow package without SQL storage, server, UI, or data science dependencies. This is recommended for users who primarily need the tracking and logging capabilities without importing the full suite of MLflow features. 

MLflow Skinny supports:

* Tracking operations (logging / loading / searching params, metrics, tags + logging / loading artifacts)
* Model registration, search, artifact loading, and transitions
* Execution of GitHub projects within notebook & against a remote target.

.. list-table:: MLflow Package vs MLflow Skinny 
   :widths: 25 25 50
   :header-rows: 1

   * - Feature 
     - MLflow
     - MLflow Skinny
   * - Tracking operations
     - x
     - x
   * - Projects
     - x
     - x
   * - Model Registry 
     - x
     - x
   * - Model Deployment 
     - x
     - 
   * - Experimentation UI
     - x
     - 
   * - Models UI
     - x
     - 
     
     
Additional dependencies can be installed to leverage the full feature set of MLflow. For example:

* To use the ``mlflow.sklearn`` component of MLflow Models, install ``scikit-learn``, ``numpy``, and ``pandas``.
* To use SQL-based metadata storage, install ``sqlalchemy``, ``alembic``, and ``sqlparse``.
* To deploy models ``pip install mlflow``

Installing
----------
Install skinny client install from PyPi via ``pip install -i https://test.pypi.org/simple/ mlflow-skinny==0.1.0``

Documentation
-------------
Official documentation for MLflow can be found at https://mlflow.org/docs/latest/index.html.

Community
---------
For help or questions about MLflow usage (e.g. "how do I do X?") see the `docs <https://mlflow.org/docs/latest/index.html>`_
or `Stack Overflow <https://stackoverflow.com/questions/tagged/mlflow>`_.

To report a bug, file a documentation issue, or submit a feature request, please open a GitHub issue.

For release announcements and other discussions, please subscribe to our mailing list (mlflow-users@googlegroups.com)
or join us on `Slack`_.
