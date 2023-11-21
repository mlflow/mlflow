======================================================
Remote Experiment Tracking with MLflow Tracking Server
======================================================

In this tutorial, you will learn how to set up MLflow Tracking environment for team development using :ref:`MLflow Tracking Server <tracking_server>`.

There are a few benefits to utilize MLflow Tracking Server:

* **Collaboration**: Multiple users can log runs to the same endpoint, and query runs and models logged by other users.
* **Sharing Results**: The tracking server also serves :ref:`tracking_ui` endpoint, where team members can easily explore each other's results.
* **Centralized Access**: The tracking server can be run as a proxy for the remote access for metadata and artifacts, making it easier to secure and audit access to data.

How it works?
=============

The following picture depicts the end state once you have completed this tutorial.

.. note::
  In this tutorial, we will use a PostgreSQL database to store experiment and run data and an S3 bucket to store artifacts, but 
  you can modify the steps below to use other data stores. Find the list of supported data stores in `artifact stores <artifacts-stores.html>`_ and `backend stores <backend-stores.html>` documentations.


When you start logging runs to the MLflow Tracking Server, the following happens:

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


Get Started
===========

Step 1 - Get MLflow
-------------------
MLflow is available on PyPI. If you don't already have it installed on your system (both local and remote), you can install it with:

.. code-section::

    .. code-block:: bash
        :name: download-mlflow

        pip install mlflow

Step 2 - Set up remote data stores
----------------------------------
MLflow Tracking Server can interact with a variety of data stores to store experiment and run data as well as artifacts.
In this tutorial, we will use a PostgreSQL database to store experiment and run data and an S3 bucket to store artifacts, but 
you can modify the steps below to use other data stores supported by MLflow.

Create a PostgreSQL database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TBA

Create an S3 bucket
~~~~~~~~~~~~~~~~~~~

TBA

Set up access credentials for the S3 bucket
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TBA


Step 3 - Start the Tracking Server
----------------------------------

Now login to your remote machine and start the MLflow Tracking Server with the following command:

.. code-block:: bash
    :caption: Command to run the tracking server in this configuration

    mlflow server \
      --backend-store-uri postgresql://user:password@postgres:5432/mlflowdb \
      --artifacts-destination s3://bucket_name \
      --host remote_host

Step 4: Logging to the Tracking Server
--------------------------------------

Set environment variable
~~~~~~~~~~~~~~~~~~~~~~~~
TBA

Alternatively, you can use the :py:func:`mlflow.set_tracking_uri` API to set the tracking URI.


Start a run
~~~~~~~~~~~

TBA

Artifact access is enabled through the proxy URI 'mlflow-artifacts:/',
giving users access to this location without having to manage credentials
or permissions.

Step 5: View logged Run in Tracking UI
--------------------------------------

MLflow Tracking Server also hosts Tracking UI on the same endpoint. 
You can access the UI by navigating to ``http://remote_host:5000`` and find the logger run.


What's Next?
============

Now you have learned how to set up MLflow Tracking Server for remote experiment tracking!

There are a couple of more advanced topics you can explore:
* **Other configurations for the Tracking Server**: By default, MLflow Tracking Server serves both backend store and artifact store. 
  You can also configure the Tracking Server to serve only backend store or artifact store, to handle different use cases such as large 
  traffic or security concerns. See :ref:`other use cases <other-tracking-setup>` for how to customize the Tracking Server for these use cases.

* **Secure the Tracking Server**: The ``--host`` option exposes the service on all interfaces. If running a server in production, we
  would recommend not exposing the built-in server broadly (as it is unauthenticated and unencrypted. Read :ref:`Secure Tracking Server <tracking-auth>`
  for the best practices to secure the Tracking Server in production.

* **New Features**: MLflow team constantly develops new features to support broader use cases. See `New Features <../new-features/index.html>`_ to catch up with the latest features.