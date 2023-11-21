=======================================
Experiment Tracking with Local Database
=======================================

In this tutorial, you will learn how to use local database to track your experiment metadata with MLflow. By default, MLflow Tracking logs run to local files,
which may cause some frustration due to fractured small files and lack of simple access interface. Also, if you are using Python, you can use SQLite that runs 
upon local file (e.g. ``mlruns.db``) and has a built-in client ``sqlite3``, eliminating the effort to install any additional dependencies and setting up database server.

How it works?
=============

The following picture depicts the end state once you have completed this tutorial.

.. note::
  In this tutorial, we will use a SQLite database for the sake of easy setup, but any SQLAlchemy-compatible databases should work as well.

When you start logging runs to MLflow, the following happens:

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

Step 2 - Create SQLite Database
-------------------------------

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

* **Remote environment setup for team development**: While storing runs and experiments data in local machine is perfectly fine for solo development, you should 
  consider using :ref:`MLflow Tracking Server <tracking-server>` when you set up team collaboration environment with MLflow Tracking. Read 
  `Remote Experiment Tracking with MLflow Tracking Server <remote-server.html>`_ tutorial to learn more.
* **New Features**: MLflow team constantly develops new features to support broader use cases. See `New Features <../new-features/index.html>`_ to catch up with the latest features.