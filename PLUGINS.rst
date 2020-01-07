Writing Your Own MLflow Plugin(s)
---------------------------------

Introduction
~~~~~~~~~~~~

MLflow Python API plugins provide a powerful mechanism for customizing the behavior of the MLflow
Python client, allowing you to define custom behaviors for logging metrics, params, and artifacts,
set special context tags at run creation, and override model registry methods for registering
models etc.

Example Plugin
~~~~~~~~~~~~~~
MLflow plugins are defined as standalone Python packages which can then be distributed for
installation via PyPI or conda. See https://github.com/mlflow/mlflow/tree/branch-1.5/tests/resources/mlflow-test-plugin for an
example package that implements all currently-supported plugin types.

In particular, note that the example package contains a ``setup.py`` that declares a number of
`entry points <https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins>`_:

.. code-block:: python

    setup(
        name="mflow-test-plugin",
        # Require MLflow as a dependency of the plugin, so that plugin users can simply install
        # the plugin & then immediately use it with MLflow
        install_requires=["mlflow"],
        ...
        entry_points={
            # Define a tracking AbstractStore plugin for tracking URIs with scheme 'file-plugin'
            "mlflow.tracking_store": "file-plugin=mlflow_test_plugin:PluginFileStore",
            # Define a ArtifactRepository plugin for artifact URIs with scheme 'file-plugin'
            "mlflow.artifact_repository":
                "file-plugin=mlflow_test_plugin:PluginLocalArtifactRepository",
            # Define a RunContextProvider plugin. The entry point name for run context providers
            # is not used, and so is set to the string "unused" here
            "mlflow.run_context_provider": "unused=mlflow_test_plugin:PluginRunContextProvider",
            # Define a model-registry AbstractStore plugin for tracking URIs with scheme 'file-plugin'
            "mlflow.model_registry_store":
                "file-plugin=mlflow_test_plugin:PluginRegistrySqlAlchemyStore",
        },
    )

The elements of this ``entry_points`` dictionary specify our various plugins:

Table with columns:

Entry-point group  Entry-point name Entry-point value

mlflow.tracking_store  Tracking URI scheme associated with the plugin. For example e.g. "mlflow_test_plugin:PluginFileStore". Custom subclass

* ``"mlflow.tracking_store": "file-plugin=mlflow_test_plugin:PluginFileStore"`` - this line
  specifies a custom subclass of `mlflow.tracking.store.AbstractStore <https://github.com/mlflow/mlflow/blob/branch-1.5/mlflow/store/tracking/abstract_store.py#L8>`_
  (specifically, the `PluginFileStore class <https://github.com/mlflow/mlflow/blob/branch-1.5/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L9>`_
  within the ``mlflow_test_plugin`` module) and associates it with tracking URIs with the scheme
  ``file-plugin``. Users who install the plugin & set a tracking URI of the form "file-plugin://<path>" will
  use the custom AbstractStore implementation defined in ``PluginFileStore``. The full tracking URI
  is passed to the ``PluginFileStore`` constructor.

* ``"mlflow.artifact_repository": "mlflow_test_plugin:PluginLocalArtifactRepository"`` - this line
  specifies a custom subclass of `mlflow.tracking.store.AbstractStore <https://github.com/mlflow/mlflow/blob/branch-1.5/mlflow/store/tracking/abstract_store.py#L8>`_
  (specifically, the `PluginFileStore class <https://github.com/mlflow/mlflow/blob/branch-1.5/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L9>`_
  within the ``mlflow_test_plugin`` module) and associates it with tracking URIs with the scheme
  ``file-plugin``. Users who install the plugin & set a tracking URI of the form "file-plugin://<path>" will
  use the custom AbstractStore implementation defined in ``PluginFileStore``. The full tracking URI
  is passed to the ``PluginFileStore`` constructor.


* ``"mlflow.tracking_store": "file-plugin=mlflow_test_plugin:PluginFileStore"`` - this line
  specifies a custom subclass of `mlflow.tracking.store.AbstractStore <https://github.com/mlflow/mlflow/blob/branch-1.5/mlflow/store/tracking/abstract_store.py#L8>`_
  (specifically, the `PluginFileStore class <https://github.com/mlflow/mlflow/blob/branch-1.5/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L9>`_
  within the ``mlflow_test_plugin`` module) and associates it with tracking URIs with the scheme
  ``file-plugin``

.. csv-table:: Table Title
   :file: plugins.csv
   :widths: 30, 70
   :header-rows: 1


.. list-table:: Frozen Delights!
   :widths: 15 10 30
   :header-rows: 1

   * - Treat
     - Quantity
     - Description
   * - Albatross
     - 2.99
     - On a stick!
   * - Crunchy Frog
     - 1.49
     - If we took the bones out, it wouldn't be
       crunchy, now would it?
   * - Gannet Ripple
     - 1.99
     - On a stick!
