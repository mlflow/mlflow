.. _plugins:

==============
MLflow Plugins
==============

As a framework-agnostic tool for machine learning, the MLflow Python API provides developer APIs for
writing plugins that integrate with different ML frameworks and backends.

Plugins provide a powerful mechanism for customizing the behavior of the MLflow
Python client and integrating third-party tools.

The MLflow Python API currently supports several types of plugins:

* **Tracking Store plugins**: specify custom client behavior when users call
  tracking API methods like :py:func:`mlflow.start_run`, :py:func:`mlflow.log_metric`, :py:func:`mlflow.log_param`,
  allowing you to integrate MLflow with third-party storage solutions
* **ArtifactRepository plugins**: specify custom client behavior when users call
  :py:func:`mlflow.log_artifact`, :py:func:`mlflow.log_artifacts`,
  allowing you to integrate MLflow with third-party storage solutions
* **Run context providers**: specify context tags to be set on runs created via the
  :py:func:`mlflow.start_run` fluent API.
* **Model Registry Store plugins**: specify custom client behavior when users call
  Model Registry APIs like :py:func:`mlflow.register_model`, ,
  allowing you to integrate MLflow with third-party storage solutions

.. contents:: Table of Contents
  :local:
  :depth: 2


Using an MLflow Plugin
----------------------

MLflow plugins are Python packages that can be installed via PyPI or conda.
This example installs a tracking plugin from source and uses it within an example script.

Install the plugin
~~~~~~~~~~~~~~~~~~

To get started, clone MLflow and install `this example plugin <https://github.com/mlflow/mlflow/tree/master/tests/resources/mlflow-test-plugin>`_:

.. code-block:: bash

  git clone https://github.com/mlflow/mlflow
  cd mlflow
  pip install -e tests/resources/mlflow-test-plugin


Run code using the plugin
~~~~~~~~~~~~~~~~~~~~~~~~~
This plugin defines a custom tracking store for tracking URIs with the ``file-plugin`` scheme that
delegates to MLflow's built-in file-based run storage. To use
the plugin, you can run any code that uses MLflow, setting the tracking URI to one with a
``file-plugin://`` scheme:

.. code-block:: bash

  MLFLOW_TRACKING_URI=file-plugin:$(PWD)/mlruns python examples/quickstart/mlflow_tracking.py

Launch the MLflow UI:

.. code-block:: bash

  cd ..
  mlflow server --backend-store-uri ./mlflow/mlruns


View results at http://localhost:5000


Writing Your Own MLflow Plugin(s)
---------------------------------

Defining a Plugin
~~~~~~~~~~~~~~~~~
You define an MLflow plugin as a standalone Python package which can then be distributed for
installation via PyPI or conda. See https://github.com/mlflow/mlflow/tree/master/tests/resources/mlflow-test-plugin for an
example package that implements all currently-supported plugin types.

In particular, the example package contains a ``setup.py`` that declares a number of
`entry points <https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins>`_:

.. code-block:: python

    setup(
        name="mflow-test-plugin",
        # Require MLflow as a dependency of the plugin, so that plugin users can simply install
        # the plugin and then immediately use it with MLflow
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

Each element of this ``entry_points`` dictionary specifies a single plugin. You
can choose to implement one or more plugin types in your package, and need not implement them all.
The type of plugin defined by each entry point and its corresponding reference implementation in
MLflow are described below. You can work from the reference implementations when writing your own
plugin:

.. list-table::
   :widths: 10 10 80 10
   :header-rows: 1

   * - Description
     - Entry-point group
     - Entry-point name and value
     - Reference Implementation
   * - Plugins for overriding definitions of tracking APIs like ``mlflow.log_metric``, ``mlflow.start_run`` for a specific
       tracking URI scheme.
     - mlflow.tracking_store
     - The entry point value (e.g. ``mlflow_test_plugin:PluginFileStore``) specifies a custom subclass of
       `mlflow.tracking.store.AbstractStore <https://github.com/mlflow/mlflow/blob/master/mlflow/store/tracking/abstract_store.py#L8>`_
       (e.g., the `PluginFileStore class <https://github.com/mlflow/mlflow/blob/master/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L9>`_
       within the ``mlflow_test_plugin`` module).

       The entry point name (e.g. ``file-plugin``) is the tracking URI scheme with which to associate the custom AbstractStore implementation.

       Users who install the example plugin and set a tracking URI of the form ``file-plugin://<path>`` will use the custom AbstractStore
       implementation defined in ``PluginFileStore``. The full tracking URI is passed to the ``PluginFileStore`` constructor.
     - `FileStore <https://github.com/mlflow/mlflow/blob/master/mlflow/store/tracking/file_store.py>`_

   * - Plugins for defining artifact read/write APIs like ``mlflow.log_artifact``, ``MlflowClient.download_artifacts`` for a specified
       artifact URI scheme (e.g. the scheme used by your in-house blob storage system).
     - mlflow.artifact_repository
     - The entry point value (e.g. ``mlflow_test_plugin:PluginLocalArtifactRepository``) specifies a custom subclass of
       `mlflow.store.artifact.artifact_repo.ArtifactRepository <https://github.com/mlflow/mlflow/blob/master/mlflow/store/artifact/artifact_repo.py#L12>`_
       (e.g., the `PluginLocalArtifactRepository class <https://github.com/mlflow/mlflow/blob/master/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L18>`_
       within the ``mlflow_test_plugin`` module).

       The entry point name (e.g. ``file-plugin``) is the artifact URI scheme with which to associate the custom ArtifactRepository implementation.

       Users who install the example plugin and log to a run whose artifact URI is of the form ``file-plugin://<path>`` will use the
       custom ArtifactRepository implementation defined in ``PluginLocalArtifactRepository``.
       The full artifact URI is passed to the ``PluginLocalArtifactRepository`` constructor.
     - `LocalArtifactRepository <https://github.com/mlflow/mlflow/blob/master/mlflow/store/artifact/local_artifact_repo.py>`_


   * - Plugins for specifying custom context tags at run creation time, e.g. tags indicating the git repo
       the run is associated with.
     - mlflow.run_context_provider
     - The entry point name is unused. The entry point value (e.g. ``mlflow_test_plugin:PluginRunContextProvider``) specifies a custom subclass of
       `mlflow.tracking.context.abstract_context.RunContextProvider <https://github.com/mlflow/mlflow/blob/master/mlflow/tracking/context/abstract_context.py#L4>`_
       (e.g., the `PluginRunContextProvider class <https://github.com/mlflow/mlflow/blob/master/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L23>`_
       within the ``mlflow_test_plugin`` module) to register.
     - `GitRunContext <https://github.com/mlflow/mlflow/blob/master/mlflow/tracking/context/git_context.py#L36>`_,
       `DefaultRunContext <https://github.com/mlflow/mlflow/blob/master/mlflow/tracking/context/default_context.py#L41>`_

   * - Plugins for overriding definitions of Model Registry APIs like ``mlflow.register_model``.
     - mlflow.model_registry_store
     - .. note:: The Model Registry is in beta (as of MLflow 1.5), so APIs are not guaranteed to be stable and model-registry plugins may break in the future.

       The entry point value (e.g. ``mlflow_test_plugin:PluginRegistrySqlAlchemyStore``) specifies a custom subclass of
       `mlflow.tracking.model_registry.AbstractStore <https://github.com/mlflow/mlflow/blob/master/mlflow/store/model_registry/abstract_store.py#L6>`_
       (e.g., the `PluginRegistrySqlAlchemyStore class <https://github.com/mlflow/mlflow/blob/master/tests/resources/mlflow-test-plugin/mlflow_test_plugin/__init__.py#L33>`_
       within the ``mlflow_test_plugin`` module)

       The entry point name (e.g. ``file-plugin``) is the tracking URI scheme with which to associate the custom AbstractStore implementation.

       Users who install the example plugin and set a tracking URI of the form ``file-plugin://<path>`` will use the custom AbstractStore
       implementation defined in ``PluginFileStore``. The full tracking URI is passed to the ``PluginFileStore`` constructor.
     - `SqlAlchemyStore <https://github.com/mlflow/mlflow/blob/master/mlflow/store/model_registry/sqlalchemy_store.py#L34>`_

Testing Your Plugin
~~~~~~~~~~~~~~~~~~~

We recommend testing your plugin to ensure that it follows the contract expected by MLflow. For
example, a tracking AbstractStore plugin should contain tests verifying correctness of its
``log_metric``, ``log_param``, ... etc implementations. See also the tests for MLflow's
reference implementations as an example:

* `Example tracking AbstractStore tests <https://github.com/mlflow/mlflow/blob/master/tests/store/tracking/test_file_store.py>`_
* `Example ArtifactRepository tests <https://github.com/mlflow/mlflow/blob/master/tests/store/artifact/test_local_artifact_repo.py>`_
* `Example RunContextProvider tests <https://github.com/mlflow/mlflow/blob/master/tests/tracking/context/test_git_context.py>`_
* `Example model-registry AbstractStore tests <https://github.com/mlflow/mlflow/blob/master/tests/store/model_registry/test_sqlalchemy_store.py>`_


Distributing Your Plugin
~~~~~~~~~~~~~~~~~~~~~~~~

Assuming you've structured your plugin similarly to the example plugin, you can `distribute it
via PyPI <https://packaging.python.org/guides/distributing-packages-using-setuptools/>`_.

Congrats, you've now written and distributed your own MLflow plugin!
