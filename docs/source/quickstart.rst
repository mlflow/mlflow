.. _quickstart:

Quickstart
==========

Install MLflow, instrument your code, and view results in a few minutes.
-------------------------------------------------------------------------------

In 15-20 minutes, you will:

- Install MLflow
- Add MLflow tracking to your code
- View run metrics in the MLflow tracking UI
- (Optional) Run a tracking server to share results with others
- (Optional) Use Databricks to store your results
- Store the models produced by your runs
- Run a model from a previous run

As a data scientist, your explorations involve running your evolving training code many times. MLflow Tracking allows you to record important information your run, review and compare it with other runs, and share results with others. As an ML Engineer or MLOps professional, it allows you to compare, share, and deploy the best models produced by the team.

.. image:: _static/images/quickstart/quickstart_tracking_overview.png
    :width: 800px
    :align: center
    :alt: Diagram showing Data Science and MLOps workflow with MLflow

MLflow is available for Python, R, and Java, but this quickstart shows Python only. For Java, see :ref:`java_api`. For R, see :ref:`R-api`.

..
    Task: https://databricks.atlassian.net/browse/DOC-8674?atlOrigin=eyJpIjoiNjg5ZmYxODUzNjYxNDQzY2FjYTUxMTYyMDE4ZWFjNGQiLCJwIjoiaiJ9

Installing MLflow
-----------------

Install MLflow from PyPi using pip:

.. code-section::

    .. code-block:: shell

        pip install mlflow

For more options, see :ref:`quickstart_drilldown_install`.

Adding MLflow tracking to your code
-----------------------------------

For many popular ML libraries, you make a single function call: ``mlflow.autolog()``. If you are using one of the supported libraries, this will automatically log the parameters, metrics, and artifacts of your run (see list at :ref:`automatic-logging`). For instance, the following autologs a **scikit-learn** run:

.. code-section::

    .. code-block:: python

        import mlflow

        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_diabetes
        from sklearn.ensemble import RandomForestRegressor

        mlflow.autolog()

        db = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

        # Create and train models.
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
        rf.fit(X_train, y_train)

        # Use the model to make predictions on the test dataset.
        predictions = rf.predict(X_test)

In addition, or if you are using a library for which ``autolog`` is not yet supported, you may use key-value pairs to track:

.. list-table::
   :widths: 10 10 80
   :header-rows: 1

   * - Name
     - Used for
     - Function call
   * - Parameters
     - Constant values (for instance, configuration parameters)
     - ``mlflow.log_param``, ``mlflow.log_params``
   * - Metrics
     - Values updated during the run (for instance, accuracy)
     - ``mlflow.log_metric``
   * - Artifacts
     - Files produced by the run (for instance, model weights)
     - ``mlflow.log_artifacts``, ``mlflow.log_image``, ``mlflow.log_text``

This example demonstrates the use of these functions:

.. code-section::
    .. code-block:: python

        import os
        from random import random, randint
        from mlflow import log_metric, log_param, log_params, log_artifacts

        if __name__ == "__main__":
            # Log a parameter (key-value pair)
            log_param("config_value", randint(0, 100))

            # Log a dictionary of parameters
            log_params({"param1": randint(0, 100), "param2": randint(0, 100)}

            # Log a metric; metrics can be updated throughout the run
            log_metric("accuracy", random() / 2.0)
            log_metric("accuracy", random() + 0.1)
            log_metric("accuracy", random() + 0.2)

            # Log an artifact (output file)
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            with open("outputs/test.txt", "w") as f:
                f.write("hello world!")
            log_artifacts("outputs")

If you are using a library that supports autologging, but wish to disable it, you may do so by calling ``mlflow.autolog(disable=True)``.

- For more details on automatic logging, see :ref:`automatic-logging`.
- For more details on the explicit logging API, see :ref:`tracking_logging_functions`.

Viewing MLflow runs and experiments
-----------------------------------

Once you've run your code, you may view the results with MLflow's tracking UI. To start the UI, run:

.. code-section::

    .. code-block:: shell

        mlflow ui

And then navigate to http://localhost:5000 in your browser. You will see a page similar to:

.. image:: _static/images/quickstart/quickstart_ui_screenshot.png
    :width: 800px
    :align: center
    :alt: Screenshot of MLflow tracking UI
..

You are in the **Default** experiment, which now contains the tracking data for your run. An experiment is a collection of related runs. You can view the parameters, metrics, and artifacts of your run by clicking on the run name.

For more details on the tracking UI, see :ref:`tracking`.

Sharing MLflow runs and experiments
-----------------------------------

For getting started, the last example stored the tracking data locally. Generally, you will want to use shared storage. Locally, MLflow stores tracking data and artifacts in an **mlruns/** subdirectory of where you ran the code. The tracking UI, when run locally, visualizes this. You can store your data remotely by:

- Calling ``mlflow.set_tracking_uri`` in your code; or
- Setting the ``MLFLOW_TRACKING_URI`` environment variable

With either option, you can track your runs in a shared filesystem, a SQLAlchemy-compatible database, a tracking server, or a Databricks workspace.

A tracking server is a lightweight HTTP server. You can run a tracking server on a network-accessible server by running:

.. code-section::

    .. code-block:: shell

        mlflow server

For instance, if you've run the above command on a machine with IP address **192.168.0.1** and port **5000**, you can add tracking data to it either by:

.. code-section::

    .. code-block:: python

        mlflow.set_tracking_uri("http://192.168.0.1:5000")
        mlflow.autolog() # Or other tracking functions

Or, on your development machine, you can set the ``MLFLOW_TRACKING_URI`` environment variable to the URL of that server:

.. code-section::

    .. code-block:: shell

        export MLFLOW_TRACKING_URI=http://192.168.0.1:5000

Now, when you run your code, it will send tracking data to the tracking server. You can view the tracking data by navigating to the URI with a browser.

There are many options available for the tracking backend. For more details, see :ref:`tracking_server`.

Using MLflow with a Databricks workspace
----------------------------------------

You need to configure Mlflow to use your Databricks workspace (To get started with Databricks, see: `Get started: Account and Workspace setup <https://docs.databricks.com/getting-started/index.html>`_). You will need to know the URL of your Databricks workspace. You can find the URL in the Configuration page of the workspace:

.. image:: _static/images/quickstart/quickstart_databricks_workspace_url.png
    :width: 800px
    :align: center
    :alt: Screenshot showing location of URL in Databricks Workspace Configuration page

At the command-line, run the following command to configure your experiment:

.. code-section::

    .. code-block:: shell

        databricks configure

Set the ``Databricks Host`` to the URL of your Databricks workspace, and set the ``Username`` and ``Password`` to the credentials you use to access the workspace. If you've created an authentication token for your Databricks workspace (``databricks tokens create``), you can use it instead of your password. Call ``databricks configure`` with the ``-t, \--token`` option.

..
    Consider adding a link to the Databricks CLI docs for tokens create

In your training code, modify the call to ``mlflow.set_tracking_uri`` to use Databricks and set the experiment to the path of your experiment in Databricks, replacing ``user_name`` and ``experiment_name`` with the appropriate values:

.. code-section::

    .. code-block:: python

        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(f"/Users/{user_name}/{experiment_name}")

If the specified experiment does not exist, it will be created.

Storing a model in MLflow
-------------------------

An MLflow Model is a directory that packages machine learning models and support files in a standard format. The directory contains:

- An **MLModel** file in YAML format specifying the model's **flavor** (or **flavors**), dependencies, signature (if supplied), and important metadata;
- The various files required by the model's flavor(s) to instantiate the model. This will often be a serialized Python object;
- Files necessary for recreating the model's runtime environment (for instance, a **conda.yaml** file); and
- Optionally, an input sample

When using autologging, MLflow will automatically log the run's model. You can also log a model manually by calling ``mlflow.{library_module_name}.log_model``. In addition, if you wish to load the model soon, it may be convenient to output the run's ID directly to the console. For that, you'll need the object of type ``mlflow.ActiveRun`` for the current run. You get that object by wrapping all of your logging code in a ``with mlflow.start_run() as run:`` block. For example:

For example:

.. code-section::

    .. code-block:: python

        import mlflow

        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_diabetes
        from sklearn.ensemble import RandomForestRegressor

        with mlflow.start_run() as run:
            # Load the diabetes dataset.
            db = load_diabetes()
            X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

            # Create and train models.
            rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = rf.predict(X_test)
            print(predictions)

            mlflow.sklearn.log_model(rf, "model")

            print("Run ID: {}".format(run.info.run_id))

In the case of the ``sklearn`` flavor, ``log_model`` stores the following files in the **artifacts** directory of the run's directory on the tracking server:

.. code-section::

    .. code-block:: shell

        model/
        |-- MLmodel
        |-- conda.yaml
        |-- model.pkl
        |-- python_env.yaml
        |-- requirements.txt

If you've not called ``set_tracking_uri`` or set the ``MLFLOW_TRACKING_URI`` environment variable to point to a remote tracking server, this **model** directory will be under the **mlruns** directory.

For more information, including a list of supported model flavors and storing your own flavor, see :ref:`models_built-in-model-flavors`.

Running a model from a specific training run
--------------------------------------------

To load and run a model stored in a previous run, you can use the ``mlflow.{library_module_name}.load_model`` function. You'll need the run ID of the run that logged the model. You can find the run ID in the tracking UI:

.. image:: _static/images/quickstart/quickstart_run_id.png
    :width: 400px
    :align: center
    :alt: Screenshot showing location of run ID in tracking UI

.. code-section::

    .. code-block:: python

        import mlflow

        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_diabetes

        db = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

        model = mlflow.sklearn.load_model("runs:/97fd7ade5106ee341e0b4c63a53a9776231")
        predictions = model.predict(X_test)
        print(predictions)

Note that while ``log_model`` saves environment-specifying files such as **conda.yaml** and **requirements.txt**, ``load_model`` does not automatically recreate that environment. To do so, you need to use your preferred method (**conda**, **virtualenv**, **pip**, etc.), using the artifacts saved by ``log_model``.

To learn more about loading runs, see :ref:`quickstart_drilldown_log_and_load_model`.

Next Steps
----------
..
    First, code:

- :ref:`MLflow tutorials and examples <tutorials-and-examples>`
- Use the MLflow Registry to store and share versioned models, see :ref:`registry`
- Use MLflow Projects for packaging your code in a reproducible and reusable way, see :ref:`projects`
- Use MLflow Recipes (experimental) to create workflows for faster iterations and easier deployment, see :ref:`recipes`
- :ref:`MLflow concepts <concepts>`
- :ref:`java_api`
- :ref:`R-api`
