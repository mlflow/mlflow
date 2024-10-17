======================
MLflow Tracking Server
======================

MLflow tracking server is a stand-alone HTTP server that serves multiple REST API endpoints for tracking runs/experiments.
While MLflow Tracking can be used in local environment, hosting a tracking server is powerful in the team development
workflow:

* **Collaboration**: Multiple users can log runs to the same endpoint, and query runs and models logged by other users.
* **Sharing Results**: The tracking server also serves :ref:`tracking_ui` endpoint, where team members can easily explore each other's results.
* **Centralized Access**: The tracking server can be run as a proxy for the remote access for metadata and artifacts, making it easier to secure and audit access to data.

Start the Tracking Server
=========================

Starting the tracking server is as simple as running the following command:

.. code-block:: bash

    mlflow server --host 127.0.0.1 --port 8080

Once the server starts runing, you should see the following output:

.. code-block::

  [2023-11-01 10:28:12 +0900] [28550] [INFO] Starting gunicorn 20.1.0
  [2023-11-01 10:28:12 +0900] [28550] [INFO] Listening at: http://127.0.0.1:8080 (28550)
  [2023-11-01 10:28:12 +0900] [28550] [INFO] Using worker: sync
  [2023-11-01 10:28:12 +0900] [28552] [INFO] Booting worker with pid: 28552
  [2023-11-01 10:28:12 +0900] [28553] [INFO] Booting worker with pid: 28553
  [2023-11-01 10:28:12 +0900] [28555] [INFO] Booting worker with pid: 28555
  [2023-11-01 10:28:12 +0900] [28558] [INFO] Booting worker with pid: 28558
  ...

There are many options to configure the server, refer to :ref:`Configure Server <configure-server>` for more details.

.. important:: 
  The server listens on http://localhost:5000 by default and only accepts
  connections from the local machine. To let the server accept connections
  from other machines, you will need to pass ``--host 0.0.0.0`` to listen on
  all network interfaces (or a specific interface address). This is typically
  required configuration when running the server **in a Kubernetes pod or a
  Docker container**.

  Note that doing this for a server running on a public network is not recommended
  for security reasons. You should consider using  a reverse proxy like NGINX or Apache
  httpd, or connecting over VPN (See :ref:`Secure Tracking Server <tracking-auth>` for more details).

.. _logging_to_a_tracking_server:

Logging to a Tracking Server
============================

Once you started the tracking server, you can connect your local clients by set the ``MLFLOW_TRACKING_URI`` environment variable to the 
server's URI, along with its scheme and port (for example, ``http://10.0.0.1:5000``) or call :py:func:`mlflow.set_tracking_uri`.

The :py:func:`mlflow.start_run`, :py:func:`mlflow.log_param`, and :py:func:`mlflow.log_metric` calls
then make API requests to your remote tracking server.

.. code-section::

  .. code-block:: python

      import mlflow

      remote_server_uri = "..."  # set to your server URI
      mlflow.set_tracking_uri(remote_server_uri)
      mlflow.set_experiment("/my-experiment")
      with mlflow.start_run():
          mlflow.log_param("a", 1)
          mlflow.log_metric("b", 2)

  .. code-block:: R

      library(mlflow)
      install_mlflow()
      remote_server_uri = "..." # set to your server URI
      mlflow_set_tracking_uri(remote_server_uri)
      mlflow_set_experiment("/my-experiment")
      mlflow_log_param("a", "1")

  .. code-block:: scala

      import org.mlflow.tracking.MlflowClient

      val remoteServerUri = "..." // set to your server URI
      val client = new MlflowClient(remoteServerUri)

      val experimentId = client.createExperiment("my-experiment")
      client.setExperiment(experimentId)

      val run = client.createRun(experimentId)
      client.logParam(run.getRunId(), "a", "1")

.. note::
    On Databricks, the experiment name passed to mlflow_set_experiment must be a valid path in the workspace e.g. ``/Workspace/Users/mlflow-experiments/my-experiment``

.. _configure-server:

Configure Server
================
This section describes how to configure the tracking server for some common use cases. Please run ``mlflow server --help`` for the full list of command line options.

Backend Store
-------------
By default, the tracking server logs runs metadata to the local filesystem under ``./mlruns`` directory.
You can configure the different backend store by adding ``--backend-store-uri`` option:

Example

.. code-block:: bash

    mlflow server --backend-store-uri sqlite:///my.db

This will create a SQLite database ``my.db`` in the current directory, and logging requests from clients will be pointed to this database.

.. _tracking-server-artifact-store:

Remote artifacts store
----------------------

Using the Tracking Server for proxied artifact access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, the tracking server stores artifacts in its local filesystem under ``./mlartifacts`` directory. To configure 
the tracking server to connect to remote storage and serve artifacts, start the server with ``--artifacts-destination`` flag.

.. code-block:: bash

    mlflow server \
        --host 0.0.0.0 \
        --port 8885 \
        --artifacts-destination s3://my-bucket

With this setting, MLflow server works as a proxy for accessing remote artifacts. The MLflow clients make HTTP request to the server for fetching artifacts.

.. important::
  If you are using remote storage, you have to configure the credentials for the server to access the artifacts. Be aware of that The MLflow artifact proxied 
  access service enables users to have an *assumed role of access to all artifacts* that are accessible to the Tracking Server. Refer :ref:`Manage Access <artifacts-stores-manage-access>` for further details.

The tracking server resolves the uri ``mlflow-artifacts:/`` in tracking request from the client to an otherwise 
explicit object store destination (e.g., "s3:/my_bucket/mlartifacts") for interfacing with artifacts. The following patterns will all resolve to the configured proxied object store location (in above example, ``s3://my-root-bucket/mlartifacts``):

 * ``https://<host>:<port>/mlartifacts``
 * ``http://<host>/mlartifacts``
 * ``mlflow-artifacts://<host>/mlartifacts``
 * ``mlflow-artifacts://<host>:<port>/mlartifacts``
 * ``mlflow-artifacts:/mlartifacts``


.. important:: 
  The MLflow client caches artifact location information on a per-run basis.
  It is therefore not recommended to alter a run's artifact location before it has terminated.

.. _tracking-server-no-proxy:

Use tracking server w/o proxying artifacts access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In some cases, you may want to directly access remote storage without proxying through the tracking server.
In this case, you can start the server with ``--no-serve-artifacts`` flag, and setting ``--default-artifact-root`` to the remote storage URI
you want to redirect the request to.

.. code-block:: bash

    mlflow server --no-serve-artifacts --default-artifact-root s3://my-bucket

With this setting, the MLflow client still makes minimum HTTP requests to the tracking server for fetching proper remote storage URI,
but can directly upload artifacts to / download artifacts from the remote storage. While this might not be a good practice for access and 
secury governance, it could be useful when you want to avoid the overhead of proxying artifacts through the tracking server.

.. note::
    If the MLflow server is *not configured* with the ``--serve-artifacts`` option, the client directly pushes artifacts
    to the artifact store. It does not proxy these through the tracking server by default.

    For this reason, the client needs direct access to the artifact store. For instructions on setting up these credentials,
    see :ref:`Artifact Stores documentation <artifacts-stores-manage-access>`.

.. note::
    When an experiment is created, the artifact storage location from the configuration of the tracking server is logged in the experiment's metadata.
    When enabling proxied artifact storage, any existing experiments that were created while operating a tracking server in
    non-proxied mode will continue to use a non-proxied artifact location. In order to use proxied artifact logging, a new experiment must be created.
    If the intention of enabling a tracking server in ``-serve-artifacts`` mode is to eliminate the need for a client to have authentication to
    the underlying storage, new experiments should be created for use by clients so that the tracking server can handle authentication after this migration.

.. _tracking-server-artifacts-only:

Optionally using a Tracking Server instance exclusively for artifact handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MLflow Tracking Server can be configured to use different backend store and artifact store, and provides a single endpoint for the clients.

However, if the volume of tracking server requests is sufficiently large and performance issues are noticed, a tracking server
can be configured to serve in ``--artifacts-only`` mode, operating in tandem with an instance that
operates with ``--no-serve-artifacts`` specified. This configuration ensures that the processing of artifacts is isolated
from all other tracking server event handling.

When a tracking server is configured in ``--artifacts-only`` mode, any tasks apart from those concerned with artifact
handling (i.e., model logging, loading models, logging artifacts, listing artifacts, etc.) will return an HTTPError.
See the following example of a client REST call in Python attempting to list experiments from a server that is configured in
``--artifacts-only`` mode:

.. code-block:: bash

    # Lauch the artifact-only server
    mlflow server --artifacts-only ...

.. code-block:: python

    import requests

    # Attempt to list experiments from the server
    response = requests.get("http://0.0.0.0:8885/api/2.0/mlflow/experiments/list")

Output

.. code-block:: text

    >> HTTPError: Endpoint: /api/2.0/mlflow/experiments/list disabled due to the mlflow server running in `--artifacts-only` mode.

Using an additional MLflow server to handle artifacts exclusively can be useful for large-scale MLOps infrastructure.
Decoupling the longer running and more compute-intensive tasks of artifact handling from the faster and higher-volume
metadata functionality of the other Tracking API requests can help minimize the burden of an otherwise single MLflow
server handling both types of payloads.

.. note::
    If an MLflow server is running with the ``--artifacts-only`` flag, the client should interact with this server explicitly by
    including either a ``host`` or ``host:port`` definition for uri location references for artifacts.
    Otherwise, all artifact requests will route to the MLflow Tracking server, defeating the purpose of running a distinct artifact server.

.. _tracking-auth:

Secure Tracking Server
======================

The ``--host`` option exposes the service on all interfaces. If running a server in production, we
would recommend not exposing the built-in server broadly (as it is unauthenticated and unencrypted),
and instead putting it behind a reverse proxy like NGINX or Apache httpd, or connecting over VPN.

You can then pass authentication headers to MLflow using these environment variables .

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

Tracking Server versioning
==========================

The version of MLflow running on the server can be found by querying the ``/version`` endpoint.
This can be used to check that the client-side version of MLflow is up-to-date with a remote tracking server prior to running experiments.
For example:

.. code-block:: python

    import requests
    import mlflow

    response = requests.get("http://<mlflow-host>:<mlflow-port>/version")
    assert response.text == mlflow.__version__  # Checking for a strict version match
