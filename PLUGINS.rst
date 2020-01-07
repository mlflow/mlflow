Writing Your Own MLflow Plugin(s)
---------------------------------

Introduction
~~~~~~~~~~~~

MLflow Python API plugins provide a powerful mechanism for customizing the behavior of the MLflow
Python client, allowing you to define custom behaviors for logging metrics, params, and artifacts,
set special context tags at run creation, and override model registry methods for registering
models etc.

Defining a Plugin
~~~~~~~~~~~~~~~~~
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

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Entry-point group
     - Entry-point name and value
   * - mlflow.tracking_store
     - The entry point value (e.g. ``mlflow_test_plugin:PluginFileStore``) specifies a custom subclass of
       `mlflow.tracking.store.AbstractStore <https://github.com/mlflow/mlflow/blob/branch-1.5/mlflow/store/tracking/abstract_store.py#L8>`_
       (e.g., the `PluginFileStore class <https://github.com/mlflow/mlflow/blob/branch-1.5/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L9>`_
       within the ``mlflow_test_plugin`` module)

       The entry point name (e.g. ``file-plugin``) is the tracking URI scheme with which to associate the custom AbstractStore implementation.
       In the example above, who install the plugin & set a tracking URI of the form ``file-plugin://<path>`` will use the custom AbstractStore
       implementation defined in ``PluginFileStore``. The full tracking URI is passed to the ``PluginFileStore`` constructor.
   * - mlflow.artifact_repository
     - The entry point value (e.g. ``mlflow_test_plugin:PluginLocalArtifactRepository``) specifies a custom subclass of
       `mlflow.store.artifact.artifact_repo.ArtifactRepository <https://github.com/mlflow/mlflow/blob/master/mlflow/store/artifact/artifact_repo.py#L12>`_
       (e.g., the `PluginLocalArtifactRepository class <https://github.com/mlflow/mlflow/blob/branch-1.5/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L18>`_
       within the ``mlflow_test_plugin`` module)

       The entry point name (e.g. ``file-plugin``) is the artifact URI scheme with which to associate the custom ArtifactRepository implementation.
       In the example above, who install the plugin & log to a run whose artifact URI is of the form "file-plugin://<path>" will use the
       custom ArtifactRepository implementation defined in ``PluginLocalArtifactRepository``.
       The full artifact URI is passed to the ``PluginLocalArtifactRepository`` constructor.
   * - mlflow.run_context_provider
     - The entry point name is unused. The entry point value (e.g. ``mlflow_test_plugin:PluginRunContextProvider``) specifies a custom subclass of
       `mlflow.tracking.context.abstract_context.RunContextProvider <https://github.com/mlflow/mlflow/blob/branch-1.5/mlflow/tracking/context/abstract_context.py#L4>`_
       (e.g., the `PluginRunContextProvider class <https://github.com/mlflow/mlflow/blob/branch-1.5/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L23>`_
       within the ``mlflow_test_plugin`` module) to register.

       When a run is created via the fluent ``mlflow.start_run`` method, MLflow
       iterates through all registered RunContextProviders. For each context provider where ``in_context`` returns True, MLflow calls
       the ``tags`` method on the context provider to compute context tags for the run. All the context tags are then merged together
       and set on the newly-created run.
   * - mlflow.model_registry_store
     - **Note**: The model registry is in beta (as of MLflow 1.5), so APIs are not guaranteed to be stable & model-registry plugins may break in the
       future.

       The entry point value (e.g. ``mlflow_test_plugin:PluginRegistrySqlAlchemyStore``) specifies a custom subclass of
       `mlflow.tracking.model_registry.AbstractStore <https://github.com/mlflow/mlflow/blob/branch-1.5/mlflow/store/model_registry/abstract_store.py#L6>`_
       (e.g., the `PluginRegistrySqlAlchemyStore class <https://github.com/mlflow/mlflow/blob/branch-1.5/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L33>`_
       within the ``mlflow_test_plugin`` module)

       The entry point name (e.g. ``file-plugin``) is the tracking URI scheme with which to associate the custom AbstractStore implementation.
       In the example above, who install the plugin & set a tracking URI of the form "file-plugin://<path>" will use the custom AbstractStore
       implementation defined in ``PluginFileStore``. The full tracking URI is passed to the ``PluginFileStore`` constructor.


Testing Your Plugin
~~~~~~~~~~~~~~~~~~~

We recommend testing your plugin to ensure that it follows the contract expected by MLflow. For
example, a tracking AbstractStore plugin should contain tests verifying correctness of its
``log_metric``, ``log_param``, ... etc implementations.


Distributing Your Plugin
~~~~~~~~~~~~~~~~~~~~~~~~

Assuming you've structured your plugin similarly to the example plugin, you can `distribute it
via PyPI <https://packaging.python.org/guides/distributing-packages-using-setuptools/>`_. We
recommend against including your plugin in MLflow proper to keep the package size small, but
please feel free to reach out via GitHub issues if you feel your plugin addresses a
sufficiently-common use case to warrant inclusion.

Congrats, you've now written & distributed your own MLflow plugin!
