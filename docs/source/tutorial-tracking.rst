.. _tutorial-tracking:

Train methodically with MLflow Tracking
=======================================

As a Data Scientist, developing a Machine Learning solution requires iterating an experiment many times. The model architecture, data preparation, and hyperparameters all change as you discover more about the problem. The Tracking component of MLflow is designed to record and explore these variations and their results.

There are 4 subcomponents within MLflow Tracking:

- A logging API to specify the parameters, metrics, and artifacts (files written) of your runs
- A Tracking Server that stores this data
- A Tracking UI that visualizes and filters your experiments
- An Artifact Store that stores models, metadata, and other artifacts produced by your runs

MLflow has Python, R, and Java/Scala/JVM APIs, but this tutorial will use Python. There are a number of backend stores for the Tracking Server and Artifact Stores, but this tutorial will primarily show the default file-store-based and SQLAlchemy-based RDBMS-based backends. For other APIs and backends, see the :ref:`tracking` reference documentation.

.. note::
    You may choose to get an overview of MLflow by running one or both of the :ref:`Data Scientist Oriented Quickstart<quickstart>` or the :ref:`MLOps Professional Oriented Quickstart<quickstart-mlops>`. The Quickstarts don't cover all of the features this tutorial will, but they will orient you to MLflow's major features and components.

Setup
------

If you have not already installed MLflow, install it using pip:

.. code-block:: bash

  pip install mlflow

You may instead choose to install ``mlflow[extras]``, but this is not required for this tutorial. If you have already installed ``mlflow-skinny``, you will need to install the full ``mlflow`` package to run this tutorial (in particular, to run the Tracking UI).

Experiments and Runs
-------------------------------

A *Run* is a single execution of your training workflow. An *Experiment* is a collection of related runs. Each run in an experiment has a unique ID, friendly name, and basic metadata such as creation date, duration, and the git commit of the code.

In addition, you should use MLflow to log:

- **Parameters**: Key-value pairs of input parameters or other values that do not change during a single run
- **Metrics**: Key-value pairs of metrics, showing performance changes during training
- **Artifacts**: Output data files in any format. In particular, the model file produced by your training job

Artifacts may be stored in a different backend than parameters and metrics. Commonly, parameters and metrics are stored in a relational database, while artifacts are stored on a shared filesystem or object store. This tutorial will show how to use the default file-based artifact store, but you can also use Amazon S3, Azure Blob Storage, or FTP servers. For more, see :ref:`artifact-stores`.

Run the Tracking Server
------------------------

The Tracking Server has a variety of backend configurations. By default, the tracking server will use the local filesystem, creating an **mlruns/** subdirectory of the directory from which you run it. 

.. code-block:: bash

  mlflow server # Creates a subdirectory ./mlruns

Commonly, a relational database is used for the backend. To use a database, you must specify a SQLAlchemy-compatible database URI. For example, to use a SQLite database, you can run:

.. code-block:: bash

  mlflow server --backend-store-uri sqlite:///mlruns.db 

This will use (and create, if necessary) a SQlite database in the current directory for storing parameters and metrics. Artifacts will still be stored in the **mlruns/** subdirectory.

It is common to use cloud storage to hold artifacts. The details will vary somewhat from cloud-to-cloud due to authentication and access control differences, but the general idea is the same. To use a cloud storage backend, you specify the URI of the artifact store as the argument to the `artifacts-destination`` parameter of the Tracking Server. 

For instance, to combine a SQLite store for parameters and metrics with an Azure blob-storage artifact store:

- Create an Azure blob storage account (called, for instance, ``my-account``) and a blob container (called, for instance, ``artifact-store``).
- Find the connection string for the storage account. In the Azure Portal, you can do this from the storage account's **Access Keys** blade. Set the environment variable ``AZURE_STORAGE_CONNECTION_STRING`` to this connection string.
- Construct the ``wasbs:``-prefixed URL for the path to your blob container. The form for this is ``f"wasbs://{container}@{account}.blob.core.windows.net/"``.
- Run the Tracking Server with the ``--artifacts-destination`` argument set to this URL.

.. code-block:: bash

  export AZURE_STORAGE_CONNECTION_KEY=DefaultEndpointsProtocol=https;AccountName=etc...
  mlflow server --backend-store-uri sqlite:///mlruns.db --artifacts-destination wasbs://artifact-store@my-account.blob.core.windows.net

  {>> Is this correct? There's also ``default_artifact_root`` ... Nope, I just don't follow the difference between `d_a_r` and `a-d` <<}

Logging API
----------------

Once you have a Tracking Server running, you can use the MLflow Tracking API to log parameters, metrics, and artifacts from your runs. The Tracking API is organized in terms of **experiments** and **runs**. An experiment is a collection of runs addressing the same use-case. 

If you do not set an experiment name, the Tracking Server will associate your runs with the ``Default`` experiment. You can also set the run name, or the Tracking Server will generate a random one for you. The run name is not required to be unique. The run ID is a UUID generated by the Tracking Server and is the primary key for the run.

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

Your model is also an artifact. You should log you should log your model with the ``mlflow.log_model`` API.


Separating backend and artifact stores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

{>> This may all be wrong. I need to go through Scenario 6 and grok it. S6 is the right one for huge companies, while S4 is the more common one for smaller. So if I get S6, S4 becomes the step towards that and I can just point / in-passing changes for S6 <<}
MLflow allows you to have a different **artifact store** than the **backend store** used for logging parameters and metrics. Commonly, you may want to store parameters and metrics in a relational database, but store artifacts in a shared filesystem or cloud-based object store. 

================  =======================  =======================
Store type        Specified with            Typical items
================  =======================  =======================
Backend store     --backend-store-uri      Runs, parameters, metrics, tags, notes, metadata
Artifact Store    --artifacts-destination  Models, files, images, in-memory objects, model summary
================  =======================  =======================

This example shows a sample configuration where the backend store is a SQLite database and the artifact store is on a netware share:

.. code-block:: bash

    mlflow server --backend-store-uri sqlite:///mlruns.db --artifacts-destination file:///mnt/nas/mlflow-artifacts

