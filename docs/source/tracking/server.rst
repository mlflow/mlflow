======================
MLflow Tracking Server
======================

.. _tracking-server-local:

Run MLflow Tracking Server Locally
----------------------------------

Networking
----------

The ``--host`` option exposes the service on all interfaces. If running a server in production, we
would recommend not exposing the built-in server broadly (as it is unauthenticated and unencrypted),
and instead putting it behind a reverse proxy like NGINX or Apache httpd, or connecting over VPN.
You can then pass authentication headers to MLflow using these :ref:`environment variables <tracking_auth>`.

Additionally, you should ensure that the ``--backend-store-uri`` (which defaults to the
``./mlruns`` directory) points to a persistent (non-ephemeral) disk or database connection.

.. _artifact_only_mode:

Using the Tracking Server for proxied artifact access
-----------------------------------------------------

To use an instance of the MLflow Tracking server for artifact operations,
start a server with the optional parameters ``--serve-artifacts`` to enable proxied artifact access and set a
path to record artifacts to by providing a value for the argument ``--artifacts-destination``. The tracking server will,
in this mode, stream any artifacts that a client is logging directly through an assumed (server-side) identity,
eliminating the need for access credentials to be handled by end-users.

.. note::
    Authentication access to the value set by ``--artifacts-destination`` must be configured when starting the tracking
    server, if required.

To start the MLflow server with proxy artifact access enabled to an HDFS location (as an example):

.. code-block:: bash

    export HADOOP_USER_NAME=mlflowserverauth

    mlflow server \
        --host 0.0.0.0 \
        --port 8885 \
        --artifacts-destination hdfs://myhost:8887/mlprojects/models \

Optionally using a Tracking Server instance exclusively for artifact handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the volume of tracking server requests is sufficiently large and performance issues are noticed, a tracking server
can be configured to serve in ``--artifacts-only`` mode, operating in tandem with an instance that
operates with ``--no-serve-artifacts`` specified. This configuration ensures that the processing of artifacts is isolated
from all other tracking server event handling.

When a tracking server is configured in ``--artifacts-only`` mode, any tasks apart from those concerned with artifact
handling (i.e., model logging, loading models, logging artifacts, listing artifacts, etc.) will return an HTTPError.
See the following example of a client REST call in Python attempting to list experiments from a server that is configured in
``--artifacts-only`` mode:

.. code-block:: python

    import requests

    response = requests.get("http://0.0.0.0:8885/api/2.0/mlflow/experiments/list")

Output

.. code-block:: text

    >> HTTPError: Endpoint: /api/2.0/mlflow/experiments/list disabled due to the mlflow server running in `--artifacts-only` mode.

Using an additional MLflow server to handle artifacts exclusively can be useful for large-scale MLOps infrastructure.
Decoupling the longer running and more compute-intensive tasks of artifact handling from the faster and higher-volume
metadata functionality of the other Tracking API requests can help minimize the burden of an otherwise single MLflow
server handling both types of payloads.


.. _logging_to_a_tracking_server:

Logging to a Tracking Server
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


Run MLflow Tracking Server on localhost
---------------------------------------
