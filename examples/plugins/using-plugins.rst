Using MLflow Plugins
--------------------

MLflow plugins are simply Python packages that can be installed via PyPI or conda. In this example,
we'll install a tracking plugin from source and use it within an example script.

Install the Plugin
~~~~~~~~~~~~~~~~~~

To get started, clone MLflow and install `this example plugin <https://github.com/mlflow/mlflow/tree/master/tests/resources/mlflow-test-plugin>`_:

.. code-block:: bash

  git clone https://github.com/mlflow/mlflow
  cd mlflow
  pip install -e tests/resources/mlflow-test-plugin


Our plugin defines a custom tracking store for tracking URIs with the ``file-plugin`` scheme that
simply delegates to MLflow's built-in file-based run storage. To use
our plugin, we can run any code that uses MLflow, setting the tracking URI to one with a
``file-plugin://`` scheme:

.. code-block:: bash

  MLFLOW_TRACKING_URI=file-plugin:$(PWD)/mlruns python examples/plugins/train.py

Launch the MLflow UI:

.. code-block:: bash

  cd ..
  mlflow server --backend-store-uri ./mlflow/mlruns


And view results at http://localhost:5000/

