======================================================
Remote Experiment Tracking with MLflow Tracking Server
======================================================

MLflow's Tracking Server supports utilizing the host as a proxy server for operations involving artifacts.
Once configured with the appropriate access requirements, an administrator can start the tracking server to enable
assumed-role operations involving the saving, loading, or listing of model artifacts, images, documents, and files.
This eliminates the need to allow end users to have direct path access to a remote object store (e.g., s3, adls, gcs, hdfs) for artifact handling and eliminates the
need for an end-user to provide access credentials to interact with an underlying object store.

.. figure:: ../../_static/images/scenario_5.png
    :align: center
    :figwidth: 800

.. code-block:: bash
    :caption: Command to run the tracking server in this configuration

    mlflow server \
      --backend-store-uri postgresql://user:password@postgres:5432/mlflowdb \
      # Artifact access is enabled through the proxy URI 'mlflow-artifacts:/',
      # giving users access to this location without having to manage credentials
      # or permissions.
      --artifacts-destination s3://bucket_name \
      --host remote_host

Enabling the Tracking Server to perform proxied artifact access in order to route client artifact requests to an object store location:

 * **Part 1a and b**:

  * The MLflow client creates an instance of a `RestStore` and sends REST API requests to log MLflow entities
  * The Tracking Server creates an instance of an `SQLAlchemyStore` and connects to the remote host for inserting
    tracking information in the database (i.e., metrics, parameters, tags, etc.)

 * **Part 1c and d**:

  * Retrieval requests by the client return information from the configured `SQLAlchemyStore` table

 * **Part 2a and b**:

  * Logging events for artifacts are made by the client using the ``HttpArtifactRepository`` to write files to MLflow Tracking Server
  * The Tracking Server then writes these files to the configured object store location with assumed role authentication

 * **Part 2c and d**:

  * Retrieving artifacts from the configured backend store for a user request is done with the same authorized authentication that was configured at server start
  * Artifacts are passed to the end user through the Tracking Server through the interface of the ``HttpArtifactRepository``

.. note::
    When an experiment is created, the artifact storage location from the configuration of the tracking server is logged in the experiment's metadata.
    When enabling proxied artifact storage, any existing experiments that were created while operating a tracking server in
    non-proxied mode will continue to use a non-proxied artifact location. In order to use proxied artifact logging, a new experiment must be created.
    If the intention of enabling a tracking server in ``-serve-artifacts`` mode is to eliminate the need for a client to have authentication to
    the underlying storage, new experiments should be created for use by clients so that the tracking server can handle authentication after this migration.

.. warning::
    The MLflow artifact proxied access service enables users to have an *assumed role of access to all artifacts* that are accessible to the Tracking Server.
    Administrators who are enabling this feature should ensure that the access level granted to the Tracking Server for artifact
    operations meets all security requirements prior to enabling the Tracking Server to operate in a proxied file handling role.

Example: xxxxxxxxx
==================

Logging to a tracking server
----------------------------

To log to a tracking server, set the ``MLFLOW_TRACKING_URI`` environment variable to the server's URI,
along with its scheme and port (for example, ``http://10.0.0.1:5000``) or call :py:func:`mlflow.set_tracking_uri`.

The :py:func:`mlflow.start_run`, :py:func:`mlflow.log_param`, and :py:func:`mlflow.log_metric` calls
then make API requests to your remote tracking server.

  .. code-section::

    .. code-block:: python

        import mlflow

        remote_server_uri = "..."  # set to your server URI
        mlflow.set_tracking_uri(remote_server_uri)
        # Note: on Databricks, the experiment name passed to mlflow_set_experiment must be a
        # valid path in the workspace
        mlflow.set_experiment("/my-experiment")
        with mlflow.start_run():
            mlflow.log_param("a", 1)
            mlflow.log_metric("b", 2)

    .. code-block:: R

        library(mlflow)
        install_mlflow()
        remote_server_uri = "..." # set to your server URI
        mlflow_set_tracking_uri(remote_server_uri)
        # Note: on Databricks, the experiment name passed to mlflow_set_experiment must be a
        # valid path in the workspace
        mlflow_set_experiment("/my-experiment")
        mlflow_log_param("a", "1")


.. _tracking_auth:

In addition to the ``MLFLOW_TRACKING_URI`` environment variable, the following environment variables
allow passing HTTP authentication to the tracking server:

- ``MLFLOW_TRACKING_USERNAME`` and ``MLFLOW_TRACKING_PASSWORD`` - username and password to use with HTTP
  Basic authentication. To use Basic authentication, you must set `both` environment variables .
- ``MLFLOW_TRACKING_TOKEN`` - token to use with HTTP Bearer authentication. Basic authentication takes precedence if set.
- ``MLFLOW_TRACKING_INSECURE_TLS`` - If set to the literal ``true``, MLflow does not verify the TLS connection,
  meaning it does not validate certificates or hostnames for ``https://`` tracking URIs. This flag is not recommended for
  production environments. If this is set to ``true`` then ``MLFLOW_TRACKING_SERVER_CERT_PATH`` must not be set.
- ``MLFLOW_TRACKING_SERVER_CERT_PATH`` - Path to a CA bundle to use. Sets the ``verify`` param of the
  ``requests.request`` function
  (see `requests main interface <https://requests.readthedocs.io/en/master/api/>`_).
  When you use a self-signed server certificate you can use this to verify it on client side.
  If this is set ``MLFLOW_TRACKING_INSECURE_TLS`` must not be set (false).
- ``MLFLOW_TRACKING_CLIENT_CERT_PATH`` - Path to ssl client cert file (.pem). Sets the ``cert`` param
  of the ``requests.request`` function
  (see `requests main interface <https://requests.readthedocs.io/en/master/api/>`_).
  This can be used to use a (self-signed) client certificate.


.. note::
    If the MLflow server is *not configured* with the ``--serve-artifacts`` option, the client directly pushes artifacts
    to the artifact store. It does not proxy these through the tracking server by default.

    For this reason, the client needs direct access to the artifact store. For instructions on setting up these credentials,
    see :ref:`Artifact Stores <artifact-stores>`.

Tracking Server versioning
~~~~~~~~~~~~~~~~~~~~~~~~~~

The version of MLflow running on the server can be found by querying the ``/version`` endpoint.
This can be used to check that the client-side version of MLflow is up-to-date with a remote tracking server prior to running experiments.
For example:

.. code-block:: python

    import requests
    import mlflow

    response = requests.get("http://<mlflow-host>:<mlflow-port>/version")
    assert response.text == mlflow.__version__  # Checking for a strict version match

TOOD:add some description and links about other modes like artifact-only