.. _tracking:

===============
MLflow Tracking
===============

The MLflow Tracking component is an API and UI for logging parameters, code versions, metrics, and output files
when running your machine learning code and for later visualizing the results.
MLflow Tracking lets you log and query experiments using :ref:`Python <python-api>`, :ref:`REST <rest-api>`, :ref:`R-api`, and :ref:`java_api` APIs.

.. figure:: ../_static/images/tracking/tracking-metrics-ui-temp.png
    :align: center
    :figwidth: 900

    A screenshot of the MLflow Tracking UI, showing a plot for validation loss metrics during model training.

Quickstart
==========
If you haven't use MLflow Tracking at all, we strongly recommend you to go through following tutorial(s) to get familar with the basic experiment tracking workflow.

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="getting-started/intro-quickstart/index.html" >
                    <div class="header">
                        MLflow Tracking Quickstart
                    </div>
                    <p>
                    A great place to start to learn the fundamentals of MLflow Tracking! Learn in 5 minutes how to log, register, and load a model for inference. 
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="getting-started/logging-first-model/index.html" >
                    <div class="header">
                        Intro to MLflow Tutorial
                    </div>
                    <span>
                        Learn how to get started with the basics of MLflow in a step-by-step instructional tutorial that shows the critical 
                        path to logging your first model
                    </span>
                </a>
            </div>
        </article>
    </section>

Concepts
========

.. _runs:

Runs
----
MLflow Tracking is organized around the concept of *runs*, which are executions of some piece of
data science code, for example, single ``python train.py`` execution. Each run records metadata
(various information about your run such as metrics, parameters, start and end time) and artifacts
(output files from the run such as model weights, images, etc).

Experiments
-----------
Experiment groups together runs for a specific task. You can create an experiment using CLI, API, or UI.
The MLflow API and UI also let you create and search for experiments. See :ref:`Organize runs into experiments <organizing-runs-in-experiments>` 
for more details on how to organize your runs into experiments.


.. _start-logging:

Tracking Runs
=============

.. toctree::
    :maxdepth: 1
    :hidden:

    tracking-api

.. note::
    By default without no particular server / database configuration, MLflow Tracking logs data to local `mlruns` directory. If you want to log your Runs to different location,
    for example remote database and cloud storage to share your results with your team, follow instructions in :ref:`Set up MLflow Tracking environment <tracking-setup>` section.

There are two ways to start tracking your runs with MLflow Tracking - auto logging and manual logging.

`Auto Logging <autolog.html>`_
------------------------------

Auto logging is a powerful feature that allows you to log metrics, parameters, and models without the need for explicit log statements. All you need to do is to call
:py:func:`mlflow.autolog` before your training code.

    .. code-block:: python

        import mlflow

        mlflow.autolog()

        # Your training code...


Autolog supports popular libraries such as :ref:`Scikit-learn <autolog-sklearn>`, :ref:`XGBoost <autolog-xgboost>`, :ref:`Pytorch <autolog-pytorch>`, :ref:`Keras <autolog-keras>`, :ref:`Spark <autolog-spark>`, and more.
See :ref:`auto logging documentation <automatic-logging>` for supported libraries and how to use autolog APIs with each of them.


Manual Logging
--------------
When you want to customize your logging logic or handle models not supported by auto logginc, you can manually insert :ref:`Logging functions <tracking_logging_functions>` to your ML code.
Following example shows simple use case to log your experiment data and models to local files with MLflow Tracking.

.. code-block:: python

    import mlflow

    with mlflow.start_run():
        mlflow.log_params({"learning_rate": 0.01, "epochs": 50})

        for epoch in range(0, 3):
            # Your model training code here...

            mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
            mlflow.log_metric(key="val_loss", value=2 * epoch, step=epoch)

        mlflow.log_artifact(local_path="model/model.pkl")

Please visit `tracking API documentation <tracking-api.html>`_ for more details about using MLflow Tracking APIs for manually logging your experiment data and models.

Explore Runs and Results
========================

.. _tracking_ui:

Tracking UI
-----------
The Tracking UI lets you visually explore your experiments and runs, with features including

* Experiment-based run listing and comparison (including run comparison across multiple experiments)
* Searching for runs by parameter or metric value
* Visualizing run metrics
* Downloading run results (artifacts and metadata)

If you log runs to a local ``mlruns`` directory,
run the following command in the directory above it, then access to ``localhost:5000`` in your browser.

.. code-block:: bash

    mlflow ui --port 5000

Alternatively, the :ref:`MLflow tracking server <tracking_server>` serves the same UI and enables remote storage of run artifacts.
In that case, you can view the UI at ``http://<ip address of your MLflow tracking server>:5000`` from any machine that can connect to your tracking server.


.. _tracking_query_api:

Querying Runs Programmatically
------------------------------

You can also access all of the functions in the Tracking UI programmatically with :py:class:`MlflowClient <mlflow.client.MlflowClient>`.


.. _tracking-setup:

Set up MLflow Tracking environment
==================================

.. note::
    If you just want to log your experiment data and models to local files, you can skip this section.

MLflow Tracking supports many different scenario for your development workflow. This section will guide you through how to set up MLflow Tracking environment for your particular use case.
From bird's-eye view, MLflow Tracking environment consists of the following components.

Components
----------

`MLflow Tracking APIs <tracking-api.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1
    :hidden:

    tracking-api

Java, R , and REST APIs for logging runs and artifacts, and communicating with MLflow Tracking Server if necessary.

`Backend store <backend-stores.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1
    :hidden:

    backend-stores

Backend store persists various metadata for each :ref:`Run <runs>`, such as run id, start & end time, parameters, metrics, etc.
MLflow supports two types storage for backend, either **file-system based** like local files and **database-based** like PostgresQL.

`Artifact store <artifacts-stores.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1
    :hidden:

    artifacts-stores

Artifact store persists (typicaly large) arifacts for each run to such as model weights (e.g. a pickled scikit-learn model),
images (e.g. PNGs), model and data files (e.g. `Parquet <https://parquet.apache.org/>`_ file). MLflow stores artifacts ina a
local file (`mlruns`) by default, but also supports different storage options such as Amazon S3 and Azure Blob Storage.

.. _tracking_server:

`MLflow Tracking Server <server.html>`_ (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1
    :hidden:

    server

MLflow Tracking Server is a stand-alone HTTP server that provides REST APIs for accessing backend and/or artifact store.
Tracking server also offers flexibility to configure what data to server, govern access control, versioning, and etc.

.. _tracking_setup:

Common Setups
-------------

.. toctree::
    :maxdepth: 1
    :hidden:

    ../getting-started/quickstart-1/index
    tutorials/local-database
    tutorials/remote-server

By locating these components properly, you can create MLflow Tracking environment suitable for your team development workflow.
The following diagram and table show a few common set ups for MLflow Tracking environment.

.. figure:: ../_static/images/tracking/tracking-setup-overview.png
    :align: center
    :figwidth: 900

    Different ways to set up MLflow enviromnent for tracking your experiments.

.. list-table::
    :widths: 45 15 20 20
    :header-rows: 1

    * - Scenario
      - Use Case
      - Description
      - Tutorial

    * - **1. Localhost (default)**
      - Solo development
      - By default, MLflow records metadata and artifacts for each run to a local directory ``mlruns``. This is the simplest way to get started with MLflow Tracking, without setting up any external server, database, and storage.
      - `QuickStart <getting-started/quickstart-1/index.html>`_

    * - **2. Local Tracking with Local Database**
      - Solo development
      - The MLflow client can interface with SQLAlchemy-compatible database (e.g. SQLite, PostgresQL, MySQl) for `backend <backend-stores.html>`_. Saving metadata to a database allows you cleaner management of your experiment data, while skipping the effort of setting up a server.
      - `Tracking Experiments with Local Database <tutorials/local-database.html>`_

    * - **3. Remote Tracking with** `MLflow Tracking Server <server.html>`_
      - Team development
      - MLflow Tracking Server can be configured with an artifacts HTTP proxy, passing artifact requests through the tracking server to store and retrieve artifacts without having to interact with underlying object store services. This is particularly useful for team development scenarios where you want to store artifacts and experiment metadata in a shared location with proper access control.
      - `Remote Experiment Tracking with MLflow Tracking Server <tutorials/remote-server.html>`_

.. _other-tracking-setup:

Other Configuration with `MLflow Tracking Server <server.html>`_
----------------------------------------------------------------
MLflow Tracking Server provides customizability for other special use cases. Please follow `Remote Experiment Tracking with MLflow Tracking Server <tutorials/remote-server.html>`_ for 
learning the basic setup, and continue to the following materials for the advanced configurations to meet your needs.

.. |image-local-server| image:: ../_static/images/tracking/tracking-setup-local-server.png
    :align: middle
    :width: 50px
    :alt: Run MLflow Tracking Server on localhost

.. |image-no-proxy| image:: ../_static/images/tracking/tracking-setup-no-serve-artifacts.png
    :align: middle
    :width: 50px
    :alt: Bypass Tracking Server proxy for artifacts access

.. |image-artifact-only| image:: ../_static/images/tracking/tracking-setup-artifacts-only.png
    :align: middle
    :width: 50px
    :alt: Use MLflow Tracking Server exclusively as a proxy for artifacts

.. list-table::
    :widths: 30 50 20
    :header-rows: 1

    * - Scenario
      - Description
      - Diagram

    * - Run MLflow Tracking Server on localhost
      - This is mainly useful for testing your team development workflow locally, or running your machine learning code on container environment.
      - |image-local-server| 

    * - Bypass Tracking Server proxy for artifacts access
      - MLflow Tracking Server by default serves both artifacts and only metadata. However, in some case you want to allow direct access to the remote artifacts storage to avoid overhead of proxy, while preserving the functionality of metadata tracking.
        Refer :ref:`Use tracking server w/o proxying artifacts access <stracking-server-no-proxy>` for how to set this up.
      - |image-no-proxy|  

    * - Use MLflow Tracking Server exclusively as a proxy for artifacts
      - If you are in large organization or training huge models, you might have high artifact transfer volumes. In such case, you may want to split out the traffic for serving artifacts, so as not to impact tracking functionality.
        You can achieve this by configure the server with ``--artifact-only`` mode, as described in :ref:`Use tracking server w/o proxying artifacts access <stracking-server-no-proxy>`.
      - |image-artifact-only| 

FAQ
===

Can I launch multiple runs in parallel?
---------------------------------------
Yes, MLflow supports launching multiple runs in parallel. See :ref:`Launching Multiple Runs in One Program <launching-multiple-runs>` for more details.

How to organize many MLflow Runs nicely?
----------------------------------------
MLflow provides a few ways to organize your runs,

* :ref:`Organize runs into experiments <organizing-runs-in-experiments>` - Experiments is a logical container for your runs. You can create an experiment using CLI, API, or UI.
* :ref:`Add tags to runs <add-tags-to-runs>` - You can associate arbitrary tags with each run, which allows you to filter and search runs based on tags.

Can I directly access remote storage, without running Tracking Server?
----------------------------------------------------------------------

Yes, while it is the best practice to have MLflow Tracking Server as a proxy for artifacts access for team development workflow, you may not need that 
if you are using it for personal project or testing. You can achieve this by the following workaround:

1. Set up artifacts configuration such as credentials, endpoints, just like you would do for MLflow Tracking Server. See :ref:`configure artifact storage <artifacts-store-supported-storages>` for more details.
2. Create an experiment with explicit artifact location,

    .. code-block:: python

        experiment_name = "your_experiment_name"
        mlflow.create_experiment(experiment_name, artifact_location="s3://your-bucket")
        mlflow.set_experiment(experiment_name)

Your runs under this experiment will log artifacts to the remote storage directly.
