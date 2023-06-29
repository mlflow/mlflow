.. _tutorial-tracking:

Train methodically with MLflow Tracking
=======================================

As a Data Scientist, developing a Machine Learning solution requires iterating an experiment many times. The model architecture, data preparation, and hyperparameters all change as you discover more about the problem. The Tracking component of MLflow is designed to record and explore these variations and their results.

.. note::
    You may choose to get an overview of MLflow by running one or both of the :ref:`Data Scientist Oriented Quickstart<quickstart>` or the :ref:`MLOps Professional Oriented Quickstart<quickstart-mlops>`. The Quickstarts don't cover all of the features this tutorial will, but they will orient you to MLflow's major features and components.

MLflow Tracking has four subcomponents:

- A **Tracking Server** that stores this data. The Tracking Server has:

  - A **Backend store** for parameters and metrics, and
  - An **Artifact store** for artifacts, such as models, visualizations, and generated data files

- A **logging API** to specify the parameters, metrics, and artifacts (files written) of your runs
- A **Tracking UI** that visualizes and filters your experiments
- A **Model Registry** that stores and manages models you register from runs

{>> Is the Registry part of Tracking? It's in the UI, but it's kind of like the hand-off from data science tracking to mlops deployment <<}

For training your models, the most-relevant components are the Tracking Server and the Tracking API.

MLflow has Python, R, and Java/Scala/JVM APIs, but this tutorial will use Python. 

Setup
------

1. Create or activate a Python virtual environment for your work; while it's not strictly necessary, we highly recommend using virtual environments. 

2. If you have not already installed MLflow, install it using pip:

.. code-block:: bash

  pip install mlflow

You may instead choose to install ``mlflow[extras]``, but this is not required for this tutorial. If you have already installed ``mlflow-skinny``, you will need to install the full ``mlflow`` package to run this tutorial (in particular, to run the Tracking UI).

3. Clone the MLflow GitHub repository which includes example code for this tutorial: 

.. code-block:: bash

  git clone https://github.com/mlflow/mlflow

4. Change your current directory to the **examples/hyperparam** directory of the **mlflow** directory you just cloned.

Run the Tracking Server
------------------------

The Tracking Server has two subcomponents: a backend store and an artifact store. You can configure the Tracking Server to use different backends for each of these as discussed below. 

By default, the tracking server will use the local filesystem for both the backend and artifact stores, creating an **mlruns/** subdirectory of the directory from which you run it. In the directory containing your training code (for instance, the **examples/hyperparam** directory), you *may* run the Tracking Server without explicitly setting either the backend or artifact store URIs:

.. code-block:: bash

  # Not preferred
  mlflow server

You *should* use a SQLAlchemy-compatible database for the backend store. To use a database, you set the ``--backend-store-uri`` argument to a database URI. For example, to use a SQLite database, you can run:

.. code-block:: bash

  # Preferred
  mlflow server --backend-store-uri sqlite:///mlruns.db

Assuming that you've installed `SQLite <https://www.sqlite.org/index.html>`, this will use (and create, if necessary) a SQLite database in the current directory for storing parameters and metrics. Artifacts will still be stored in the **mlruns/** subdirectory.

{>> Does `mlflow` package install SQLAlchemy package or might they have to install that to? <<}

Naturally, the details of installing the database server will vary by vendor.

.. important::
  You must run the tracking server using a database-backed backend store to use the Model Registry.

You can also use the ``--backend-store-uri`` to specify a network-accessible file-system or database server.

By default, the tracking server will listen on port 5000. You can change this with the ``--port`` argument. 

Using a difference artifact store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this tutorial, we will assume the use of the local file system to store artifacts. However, it is common to use cloud storage to hold artifacts. The details will vary somewhat from cloud-to-cloud due to authentication and access control differences, but the general idea is the same. To use a cloud storage backend, you specify the URI of the artifact store as the argument to the `artifacts-destination`` parameter of the Tracking Server. 

For instance, to combine a SQLite store for parameters and metrics with an Azure blob-storage artifact store:

- Create an Azure blob storage account (called, for instance, ``my-account``) and a blob container (called, for instance, ``artifact-store``).
- Find the connection string for the storage account. In the Azure Portal, you can do this from the storage account's **Access Keys** blade. Set the environment variable ``AZURE_STORAGE_CONNECTION_STRING`` to this connection string.
- Construct the ``wasbs:``-prefixed URL for the path to your blob container. The form for this is ``f"wasbs://{container}@{account}.blob.core.windows.net/"``.

{>> Which is a little behind-the-times because WASB is headed for deprecation. But I couldn't get ABFS to work <<} 
- Run the Tracking Server with the ``--artifacts-destination`` argument set to this URL.

.. code-block:: bash

  export AZURE_STORAGE_CONNECTION_KEY=DefaultEndpointsProtocol=https;AccountName=etc...
  mlflow server --backend-store-uri sqlite:///mlruns.db --artifacts-destination wasbs://artifact-store@my-account.blob.core.windows.net

  {>> Is this correct? There's also ``default_artifact_root`` ... Nope, I just don't follow the difference between `d_a_r` and `a-d` <<}

For other APIs and backends, see the :ref:`tracking` reference documentation.
{>> :ref:`artifact-stores` ? <<}

Logging API Example
-------------------------------

You should now have an instance of the Tracking Server running. Now it is time to begin tracking your experiment. 

A *Run* is a single execution of your training workflow. An *Experiment* is a collection of related runs. Each run in an experiment has a unique ID, friendly name, and basic metadata such as creation date, duration, and the git commit of the code.

In addition, you should use MLflow to log:

- **Parameters**: Key-value pairs of input parameters or other values that do not change during a single run
- **Metrics**: Key-value pairs of metrics, showing performance changes during training
- **Artifacts**: Output data files in any format. In particular, the model file produced by your training job

If you do not set an experiment name, the Tracking Server will associate your runs with the ``Default`` experiment. You can also set the run name, or the Tracking Server will generate a random one for you. The run name is not required to be unique. The run ID is a UUID generated by the Tracking Server and is the primary key for the run.

1. Set the ``MLFLOW_TRACKING_URI`` environment variable to the URI of your Tracking Server:

.. code-block:: bash

  export MLFLOW_TRACKING_URI=http://localhost:5000

(Note that this is ``http`` and not ``https``.)

2. Set your working directory to the **examples/hyperparam** subdirectory

3. Begin the hyperparameter sweep with:

.. code-block:: bash

  mlflow run -e hyperopt .

This command will take several minutes to execute. Because this project is defined using :ref:`projects`, the runtime environment (including `Tensorflow <https://www.tensorflow.org/>` and `hyperopt<https://github.com/hyperopt/hyperopt>`) will be created automatically and then the ``hyperopt`` entry point defined in the **MLproject** file is run. The ``hyperopt`` entry point calls the **search_hyperopt.py** program, which repeatedly calls the ``train`` entry point in the same file, which in turn executes **train.py**. By default, 12 runs of 32 epochs are run.

Both **search_hyperopt.py** and **train.py** (and, for that matter, **search_random.py**) contain MLFlow logging calls, as discussed below. The example also contains several more advanced techniques, such as using child runs, automatically setting experimental tags, tracking best-to-date metrics, and so forth. These are not discussed in this tutorial, but the example is well worth reading.

The ``ActiveRun`` object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examine the ``run()`` function of the ``train.py`` file in the **examples/hyperparam** directory. You'll see the following structure:

.. code:: python

  def run(training_data, epochs, batch_size, learning_rate, momentum, seed):
    # ... data and model preparation ...
    with mlflow.start_run():
      # ... training code ... 

The call to :py:func:`mlflow.start_run` returns an object of type :py:class:`mlflow.ActiveRun`. The ``ActiveRun`` object contains metadata about the run that you may find useful. If you want to keep a reference to that object, you can use:

.. code:: python

  with mlflow.start_run() as run:
    # ... training code ... 
    run_id = run.info.run_id # For instance

Example metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the same **train.py** file, examine the ``eval_and_log_metrics()`` function, reproduced here:

.. code:: python

  def eval_and_log_metrics(prefix, actual, pred, epoch):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mlflow.log_metric("{}_rmse".format(prefix), rmse, step=epoch)
      return rmse

The ``prefix`` argument is one of ``"train"``, ``"val"``, or ``"test"`` and the call to :py:func:`mlflow.log_metric` records the current error in a metric named ``f"{prefix}_rmse``. 

Example parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **hyperparam** example does a hyperparameter sweep, calling the ``train`` entry point of the **MLProject** file repeatedly, using MLflow child runs to gather related runs under the sweep. Open the **search_hyperopt.py** file and examine the ``eval(params)`` function. This nested function is called repeatedly through the hyperparameter sweep. It logs the hyperparameters as parameters using the :py:func:`mlflow.log_params` API:

.. code:: python

  def eval(params):
    # ... other code ...
    with mlflow.start_run(nested=True) as child_run:
      # ... other code ...
      mlflow.log_params({"lr": lr, "momentum": momentum})

Here, the learning rate and momentum of each child run is logged as a parameter. The ``nested=True`` argument to :py:func:`mlflow.start_run` tells MLflow to associate the child run with the parent run, so that you can see the parent-child relationship in the UI. 

Example artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **hyperparam** example saves the models, but no other artifacts. Open the **train.py** file and examine the ``MLflowCheckpoint.__exit__()`` function, reproduced in part below:

.. code:: python

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ... other code ...
        predictions = self._best_model.predict(self._test_x)
        signature = infer_signature(self._test_x, predictions)
        mlflow.tensorflow.log_model(self._best_model, "model", signature=signature)

This snippet demonstrates a recommended pattern for logging models. First, the model predicts results for the test data. The test data (``self._test_x``) and ``predictions`` are passed to the :py:func:`mlflow.models.infer_signature` function, which returns a :py:class:`mlflow.models.ModelSignature` object. This object describes the model's inputs and outputs and provides better runtime diagnostics of incorrect inputs when the model is deployed.

The call to :py:func:`mlflow.tensorflow.log_model` saves the model in the Tensorflow "flavor" of MLflow. Each ML library that supports MLflow will implement ``log_model`` (and it's complement ``load_model``) differently. 

The most general form of the function is :py:func:`mlflow.pyfunc.log_model`, which makes no assumptions about the model beyond it being callable from a Python function. {>> tk: I need the best pyfunc ref we got here, and then I need to switch it to the dedicated doc <<}

Parameters, Metrics, and Artifacts
----------------------------------

Parameters
~~~~~~~~~~~~~~

**Parameters** are write-once values that do not change throughout a single run. You may additionally log other values that do not change during a run, such as the dataset source, its size, etc.

You can store a single key-value pair with the ``mlflow.log_param`` API. For instance:

.. code-block:: python

  mlflow.log_param("learning_rate", 1E-3)

As your code evolves, you may end up storing parameters in one or two ``Dictionary`` objects. You can quickly log all these values with the ``mlflow.log_params`` API, as was done in the **hyperparam** sample. For instance:

.. code-block:: python

  params = {"learning_rate": 1E-3, "batch_size": 32, "epochs": 30, "dataset": "CIFAR10"}
  mlflow.log_params(params)  

Once you have logged a parameter in a given run, you may not overwrite the value. Doing so will raise an exception of type `MLflowException`.

Metrics
~~~~~~~~

**Metrics** are values that change during training. For instance, loss and accuracy are common metrics. You can log a single metric with the ``mlflow.log_metric`` API, as was done in the **hyperparam** sample. For instance:

.. code-block:: python

  for loss in range(2,10):
    mlflow.log_metric("loss", 1.0 / loss)

As with parameters, you can log a dictionary of values all at once with the ``mlflow.log_metrics`` API. For instance:

.. code-block:: python

  metrics = {"loss": 0.5, "accuracy": 0.75}
  mlflow.log_metrics(metrics)

Artifacts
~~~~~~~~~~

**Artifacts** are files produced by your training run. Typically these will be models, result summaries, visualizations, and so forth. You may log a single artifact with ``mlflow.log_artifact`` or a directory of artifacts with ``mlflow.log_artifacts``. For instance:

.. code-block:: python

  path_to_summary = "summary.txt"
  path_to_visualizations = "visualizations/"

  mlflow.log_artifact(path_to_summary)
  mlflow.log_artifacts(visualizations)

Your model is also an artifact. You should log you should log your model with the ``mlflow.log_model`` API.

Conclusion
----------

This how-to showed you how to use two of MLflow Tracking's components: the Tracking Server and the Tracking API. You learned that the Tracker Server has two stores: a backing store that contains metrics and parameters and an artifact store that contains artifacts. You learned that you must use a SQLAlchemy-compatible database as the backing store if you wish to use MLflow's Model Registry. You learned how to use the Tracking API to log parameters, metrics, and artifacts. You also learned how to infer the signature of your model, and to pass that signature to the ``mlflow.log_model`` API.
