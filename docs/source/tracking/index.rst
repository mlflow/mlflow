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
data science code, for example, single `python train.py` execution. Each run records metadata
(various information about your run such as metrics, parameters, start and end time) and artifacts
(output files from the run such as model weights, images, etc).

Experiments
-----------
Experiment groups together runs for a specific task. You can create an experiment using CLI, API, or UI.
The MLflow API and UI also let you create and search for experiments. See :ref:`Organize runs into experiments <organizing-runs-in-experiments>` 
for more details on how to organize your runs into experiments.


.. _start-logging:

Start Tracking Runs
===================

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

Please visit `client documentation <client.html>`_ for more details about using MLflow Tracking APIs for manually logging your experiment data and models.

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
run the following command in the directory above it, then access to `localhost:5000` in your browser.

.. code-block:: bash

    mlflow ui --port 5000

Alternatively, the :ref:`MLflow tracking server <tracking_server>` serves the same UI and enables remote storage of run artifacts.
In that case, you can view the UI at ``http://<ip address of your MLflow tracking server>:5000`` from any machine that can connect to your tracking server.


.. _tracking_query_api:

Querying Runs Programmatically
------------------------------

You can access all of the functions in the Tracking UI programmatically through client SDK in the :py:mod:`mlflow.client` module. 


.. _tracking-setup:

Set up MLflow Tracking environment
==================================

.. note::
    If you just want to log your experiment data and models to local files, you can skip this section.

MLflow Tracking supports many different scenario for your development workflow. This section will guide you through how to set up MLflow Tracking environment for your particular use case.
From bird's-eye view, MLflow Tracking environment consists of the following components.

Components
----------

.. toctree::
    :maxdepth: 1
    :hidden:

    client
    artifacts-stores
    backend-stores
    server

`MLflow Client APIs <client.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MLflow provides APIs for logging runs and artifacts, and communicating with MLflow Tracking Server if necessary.

`Backend store <backend-stores.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Backend store persists various metadata for each :ref:`Run <runs>`, such as

* Run ID
* Start & end time
* Parameters
* Metrics
* Code version (only if you launch runs from an :ref:`MLflow Project <projects>`).
* Source file name (only if you launch runs from an :ref:`MLflow Project <projects>`).


MLflow supports two types storage for backend, either **file-system based** like local files and **database-based** like PostgresQL.
Visit `backend store documentation <backend-stores.html>`_ for how to set up backend store with each type.

.. _artifact-stores:

`Artifact store <artifacts-stores.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Artifact store persists (typicaly large) arifacts for each run to such as model weights (e.g. a pickled scikit-learn model),
images (e.g. PNGs), model and data files (e.g. `Parquet <https://parquet.apache.org/>`_ file). MLflow stores artifacts ina a
local file (`mlruns`) by default, but also supports different storage options such as Amazon S3 and Azure Blob Storage. Visit
`artifact store documentation <artifacts-stores.html>`_ for how to set up artifact store for your use case.

.. _tracking_server:

`MLflow Tracking Server <server.html>`_ (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MLflow Tracking Server is a stand-alone HTTP server that provides REST APIs for accessing backend and/or artifact store.
Tracking server also offers flexibility to configure what data to server, govern access control, versioning, and etc.
Please visit `tracking server documentation <server.html>`_ for more details.

Common Setups
-------------

.. toctree::
    :maxdepth: 1
    :hidden:

    tutorials/local-with-various-store
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

    * - **Localhost (default)**
      - Solo development
      - * By default, MLflow records metadata and artifacts for each run to a local directory ``mlruns``. This is the simplest way to get started with MLflow Tracking, without setting up any external server, database, and storage.
      - `QuickStart <getting-started/quickstart-1/index.html>`_

    * - **Connect MLflow with database and cloud storage**
      - Solo / Team development
      - The MLflow client can interface with a variety of `backend <backend-stores.html>`_ and `artifact <artifact-stores.html>`_ storage configurations, such as 
        * SQLAlchemy-compatible databases
        * Amazon S3
        * Azure Blob Storage
        * and more
      - `Connect MLflow Tracking to Data Stores <tutorials/local-with-various-store.html>`_

    * - **Remote Tracking with** `MLflow Tracking Server <server.html>`
      - Team development
      - MLflow Tracking Server can be configured with an artifacts HTTP proxy, passing artifact requests through the tracking server to store and retrieve artifacts without having to interact with underlying object store services. This is particularly useful for team development scenarios where you want to store artifacts and experiment metadata in a shared location with proper access control.
      - `Remote Experiment Tracking with MLflow Tracking Server <tutorials/remote-server.html>`_


Other Configuration with `MLflow Tracking Server <server.html>`_
----------------------------------------------------------------
Here are some additional guidance on how to set up MLflow Tracking environment for other special use cases.

- `Run MLflow Tracking Server on localhost <server.html>`_: This is mainly useful for testing your team development workflow locally, or running your machine learning code on container environment.
- `Use MLflow Tracking Server for remote metadata store <server.html>`_: MLflow Tracking Server by default doesn't serve artifacts but only metadata. This is useful when manage your artifacts differently like cloud storage (while you can access cloud storage via MLflow server as well).
- `Use MLflow Tracking Server for serving artifacts only <server.html>`_: In contrary to above, you can also configure MLflow Tracking Server to serve artifacts only. This mode is useful when you have high artifact transfer volumes (e.g. LLM model weights) hence split the traffic so as not to impact tracking functionality.


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
