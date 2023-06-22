.. _tutorial-tracking:

MLflow Tracking Tutorial
========================

As a Data Scientist, developing a Machine Learning solution requires iterating an experiment many times. The model architecture, data preparation, and hyperparameters all change as you discover more about the problem. The Tracking component of MLflow is designed to record and explore these variations and their results.

There are 4 subcomponents within MLflow Tracking:

- A logging API to specify the parameters, metrics, and artifacts (files) of your runs
- A Tracking Server that stores this data
- A Tracking UI that visualizes and filters your experiments
- An Artifact Store that stores models, metadata, and other artifacts produced by your runs

MLflow has Python, R, and Java/Scala/JVM APIs, but this tutorial will use Python. There are a number of backend stores for the Tracking Server and Artifact Stores, but this tutorial will primarily show the default file-store-based and SQLAlchemy-based RDBMS-based backends. For other APIs and backends, see the :ref:tracking reference documentation.

.. note::
    You may choose to get an overview of MLflow by running one or both of the :ref:`Data Scientist Oriented Quickstart<quickstart>` or the :ref:`MLOps Professional Oriented Quickstart<mlops-quickstart000>`. The Quickstarts don't cover all of the features this tutorial will, but they will orient you to the main MLflow features and components.

Setup
------

If you have not already installed MLflow, install it using pip:

.. code-block:: bash

  pip install mlflow

You may instead choose to install `mlflow[extras]`, but this is not required for this tutorial. If you have already installed `mlflow-skinny`, you will need to install the full `mlflow` module to run this tutorial (in particular, to run the Tracking UI).

Experiments and Runs
-------------------------------

A *Run* is a single execution of your training workflow. An *Experiment* is a collection of related runs. Each run in an experiment has a unique ID, friendly name, and basic metadata such as creation date, duration, and the git commit of the code.

In addition, you should use MLflow to log:

- **Parameters**: Key-value pairs of input parameters
- **Metrics**: Key-value pairs of metrics, showing performance changes during training
- **Artifacts**: Output data files in any format. In particular, the model file produced by your training job.

Artifacts may be stored in a different backend than parameters and metrics. Commonly, parameters and metrics are stored in a relational database, while artifacts are stored on a shared filesystem or object store. This tutorial will show how to use the default file-based artifact store, but you can also use Amazon S3, Azure Blob Storage, or FTP servers. See tk

Run the Tracking Server
------------------------

The Tracking Server has a variety of backend configurations. By default, the tracking server will use the local filesystem , creating an `mlruns` subdirectory of the directory from which you run it. 

.. code-block:: bash

  mlflow server # Creates a subdirectory ./mlruns

Commonly, a relational database is used for the backend. To use a database, you must specify a SQLAlchemy database URI. For example, to use a SQLite database, you can run:

.. code-block:: bash

  mlflow server --backend-store-uri sqlite:///mlruns.db 

This will use (and create, if necessary) a SQlite database in the current directory for storing parameters and metrics. Artifacts will still be stored in the `mlruns` subdirectory.

It is common to use cloud storage to hold artifacts. The details will vary somewhat from cloud-to-cloud due to authentication and access control differences, but the general idea is the same. To use a cloud storage backend, you specify a URI for the artifact store with the ``artifacts-destination`` argument for the Tracking Server.

For instance, to combine a SQLite store for parameters and metrics with an Azure blob-storage artifact store:

- Create an Azure blob storage account (called, for instance, ``my-account``) and a blob container (called, for instance, ``artifact-store``).
- Find the connection string for the storage account. In the Azure Portal, you can do this from the storage account's **Access Keys** blade. Set the environment variable ``AZURE_STORAGE_CONNECTION_STRING`` to this connection string.
- Construct the ``wasbs:``-prefixed URL for the path to your blob container. The form for this is ``f"wasbs://{container}@{account}.blob.core.windows.net/"``.
- Run the Tracking Server with the ``--artifacts-destination`` argument set to this URL.

.. code-block:: bash

  export AZURE_STORAGE_CONNECTION_KEY=DefaultEndpointsProtocol=https;AccountName=etc...
  mlflow server --backend-store-uri sqlite:///mlruns.db --artifacts-destination wasbs://artifact-store@my-account.blob.core.windows.net

  {>> Needs a discussion of MLFLOW_TRACKING_URI <<}

Logging API
----------------

Once you have a Tracking Server running, you can use the MLflow Tracking API to log parameters, metrics, and artifacts from your runs. The Tracking API is organized in terms of **runs** and **experiments**. An experiment is a set of runs which have the same name. 

If you do not set an experiment name, the Tracking Server will associate your runs with the ``Default`` experiment. You can also set the run name, or the Tracking Server will generate a random one for you. The run name is not required to be unique. The run ID is a UUID generated by the Tracking Server and is the primary key for the run.

CLI vs. API Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~

You may configure MLflow in two ways:

- Using command-line facilities such as environment variables and command-line switches. This is often quicker and easier, but less reproducible.
- Using code. This "infrastructure as code" approach is reproducible and easier to share, but requires more organization and initial investment.

Hereon out, this tutorial is going to use the code-based approach. For brevity, we will not separate infrastructure functions from the rest of the code and will not show, for instance, loading and using keys and values from a separate JSON or YAML file.

~~For instance, to reproduce the above Tracking Server configuration using the API, you might use code similar to:~~ This is for the client

.. code-block:: python

  import mlflow
  # Never put credentials in code. Use environment variables if not a secret manager.
  assert(os.environ.get("AZURE_STORAGE_CONNECTION_STRING") is not None)
  mlflow.set_tracking_uri("sqlite:///mlruns.db")
  mlflow.set_artifact_uri("wasbs://artifact-store@my-account...")


Parameters
~~~~~~~~~~

**Parameters** are write-once values that do not change throughout a single run. For instance, learning rate, embedding size, and other hyperparameters are usually logged as parameters. You may additionally log other values that do not change during a run, such as the dataset source, its size, etc.

You can store a single key-value pair with the ``mlflow.log_param`` API. For instance:

.. code-block:: python

  mlflow.log_param("learning_rate", 1E-3)

As your code evolves, you may end up storing parameters in one or two ``Dictionary`` objects. You can quickly log all these values with the ``mlflow.log_params`` API. For instance:

.. code-block:: python

  params = {"learning_rate": 1E-3, "batch_size": 32, "epochs": 30, "dataset": "CIFAR10"}
  mlflow.log_params(params)  

Once you have logged a parameter in a given run, you may not overwrite the value. Doing so will raise an exception of type `MLflowException`.

Metrics
~~~~~~~~

**Metrics** are values that change during training. For instance, loss and accuracy are common metrics. You can log a single metric with the ``mlflow.log_metric`` API. For instance:

.. code-block:: python

  for loss in range(2,10):
    mlflow.log_metric("loss", 1.0 / loss)

As with parameters, you can log multiple metrics at once with the ``mlflow.log_metrics`` API. For instance:

.. code-block:: python

  metrics = {"loss": 0.5, "accuracy": 0.75}
  mlflow.log_metrics(metrics)

Artifacts
~~~~~~~~~~

**Artifacts** are files produced by your training run. Typically these will be results, summaries, visualizations, and so forth. You may log a single artifact with ``mlflow.log_artifact`` or a directory of artifacts with ``mlflow.log_artifacts``. For instance:

.. code-block:: python
  path_to_summary = "summary.txt"
  path_to_visualizations = "visualizations/"

  mlflow.log_artifact(path_to_summary)
  mlflow.log_artifacts(visualizations)


Separating backend and artifact stores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

{>> This may all be wrong. I need to go through Scenario 6 and grok it. S6 is the right one for huge companies, while S4 is the more common one for smaller. So if I get S6, S4 becomes the step towards that and I can just point / in-passing changes for S6 <<}
MLflow allows you to have a different **artifact store** than the **backend store** used for logging parameters and metrics. Commonly, you may want to store parameters and metrics in a relational database, but store artifacts in a shared filesystem or cloud-based object store. 

================  ================                       ================
  Store type        Specified with                          Typical items
================  ================                       ================
  Backend store    `--backend-store-uri`                  Runs, parameters, metrics, tags, notes, metadata
  Artifact Store    `--artifacts-destination`             Models, files, images, in-memory objects, model summary
================  ================                        ================

Authentication and security varies between stores. You may need to set environment variables or configure other credentials, depending on the store type. See the documentation for the store type you are using for details. This example shows a sample configuration where the backend store is a SQLite database and the artifact store is on a netware share:

.. code-block:: bash

    mlflow server --backend-store-uri sqlite:///mlruns.db --artifacts-destination file:///mnt/nas/mlflow-artifacts

The Tracking UI
----------------

The Tracking UI is a web application that visualizes the results of experiments and runs for which you used the MLflow Tracking API. You can run the Tracking UI with the ``mlflow ui`` command and it accepts many of the same arguments as the Tracking Server (``--port``, ``--host``, ``--backend-store-uri``, etc.). By default, ``mlflow ui`` will look for the ``MLFLOW_TRACKING_URI`` environment variable and use that as the backend store URI. If you do not set this environment variable and do not use the ``--backend-store-uri`` parameter, the Tracking UI will visualize the **mlruns/** subdirectory of the current working directory.

{>> Can you also just hit the server URI with a browser? Or vv, can you just run the UI and get the server? <<}

When you navigate to the Tracking UI, you will see a page similar to this:

.. image:: ../../_static/tracking-ui.png
   :width: 100%

Down the left-hand side of the browser, the UI lists the **Experiments** that are being tracked (1). Individual **Runs** are shown in the main body of the page (2). The search box allows you to rapidly filter the displayed runs (3) (search capabilities are discussed later). You can switch between a **Table view** and a **Chart view** summary of runs (4). The **Models** tab displays the registered models that are tracked (5).

The **Chart view** allows you to compare runs with visualizations of parameters used and metrics generated. The **Parallel Coordinates** chart is particularly useful for comparing runs with many parameters and metrics. In the followiung image, the final column shows the error in the validation set, while the left-hand columns show the learning rate and momentum used in the 14 runs. As you can see from the redder lines in the graph, when the learning rate is 0, the error is almost 0.9, and high ``momentum`` arguments lead to similar poor results. When the ``momentum`` is set to lower values, the model does a better job. 

.. image:: ../../_static/ui-tutorial/parallel-coordinates.png
   :width: 100%

