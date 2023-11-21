======================================================
Connect MLflow directly to remote storage and database
======================================================

In this tutorial, you will learn how to use external storage and database with MLflow Tracking **without** using :ref:`MLflow Tracking Server <tracking_server>`.

.. warning::
    While it might be ok to directly access remote storages and databases if you just want to try out MLflow Tracking for solo development purpose or for a small team,
    **we strongly recommended to use MLflow Tracking Server when you set up team collaboration environment with MLflow Tracking**, because of multiple benefits of having
    a centralized endpoint for accessing assets, for example, unified access control, easy result sharing, and more. Please follow 
    `Remote Experiment Tracking with MLflow Tracking Server <remote-server.html>`
    tutorial to learn how to set up team environment.

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
MLflow is available on PyPI. If you don't already have it installed on your local machine, you can install it with:

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TBA

Step 3 - Configure MLflow environment varialbles
------------------------------------------------

TBA

Step 4: Start logging
---------------------

TBA

Step 5: View logged Run in Tracking UI
--------------------------------------

Once your training job finishes, you can run following command to launch the MLflow UI:

.. code-section::

    .. code-block:: bash
        :name: view-results

        mlflow ui --port 8080

Then, navigate to `http://localhost:8080 <http://localhost:8080>`_ in your browser to view the results.


What's Next?
============

Now you have learned how to connect MLflow Tracking with remote storage and database.

There are a couple of more advanced topics you can explore:
* **Remote environment setup for team development**: While accessing remote data directly might be ok for solo development, you should 
  consider using :ref:`MLflow Tracking Server <tracking-server>` when you set up team collaboration environment with MLflow Tracking. Read 
  `Remote Experiment Tracking with MLflow Tracking Server <remote-server.html>`_ tutorial to learn more.

* **New Features**: MLflow team constantly develops new features to support broader use cases. See `New Features <../new-features/index.html>`_ to catch up with the latest features.