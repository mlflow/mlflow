====================
MLflow Tracking APIs
====================

`MLflow Tracking <../tracking.html>`_ provides Python, R, Java, or REST API to log your experiment data and models.

`Auto Logging <autolog.html>`_
==============================

.. toctree::
    :maxdepth: 1
    :hidden:

    autolog

`Auto logging <autolog.html>`_ is a powerful feature that allows you to log metrics, parameters, and models without the need for explicit log statements
but just a single ``mlflow.autolog()`` call at the top of your ML code. See :ref:`auto logging documentation <automatic-logging>` for supported libraries and how to use autolog APIs with each of them.

    .. code-block:: python

        import mlflow

        mlflow.autolog()

        # Your training code...

Manual Logging
==============

Alternatively, you log MLflow metrics by adding ``log`` methods in your ML code. For example:

Python
  .. code-block:: python

    with mlflow.start_run():
        for epoch in range(0, 3):
            mlflow.log_metric(key="quality", value=2 * epoch, step=epoch)

Java and Scala
  .. code-block:: java

    MlflowClient client = new MlflowClient();
    RunInfo run = client.createRun();
    for (int epoch = 0; epoch < 3; epoch ++) {
        client.logMetric(run.getRunId(), "quality", 2 * epoch, System.currentTimeMillis(), epoch);
    }

.. _tracking_logging_functions:

Logging functions
-----------------

Here are the full list of logging functions provided by the Tracking API (Python). Note that Java and R APIs provide similar but
limited set of logging functions.

:py:func:`mlflow.set_tracking_uri` connects to a tracking URI. You can also set the
``MLFLOW_TRACKING_URI`` environment variable to have MLflow find a URI from there. In both cases,
the URI can either be a HTTP/HTTPS URI for a remote server, a database connection string, or a
local path to log data to a directory. The URI defaults to ``mlruns``.

:py:func:`mlflow.get_tracking_uri` returns the current tracking URI.

:py:func:`mlflow.create_experiment` creates a new experiment and returns the experiment ID. Runs can be
launched under the experiment by passing the experiment ID to ``mlflow.start_run`` or 
by setting the active experiment with :py:func:`mlflow.set_experiment`, passing in the experiment ID of the created 
experiment.

:py:func:`mlflow.set_experiment` sets an experiment as active and returns the active experiment instance. If you do not 
specify an experiment in :py:func:`mlflow.start_run`, new runs are launched under this experiment.

.. note:: 
    If the experiment being set by name does not exist, a new experiment will be 
    created with the given name. After the experiment has been created, it will be set 
    as the active experiment. On certain platforms, such as Databricks, the experiment name 
    must be an absolute path, e.g. ``"/Users/<username>/my-experiment"``.


:py:func:`mlflow.start_run` starts a new run and returns a :py:class:`mlflow.ActiveRun` object 
usable as a context manager for the current run. If an active run is already in progress, you 
should either end the current run before starting the new run or nest the new run within the current run using ``nested=True``.
You do not need to call ``start_run`` explicitly: calling one of the logging functions
with no active run automatically starts a new one.

.. note::
  - If the argument ``run_name`` is not set within :py:func:`mlflow.start_run`, a unique run name will be generated for each run.

:py:func:`mlflow.end_run` ends the currently active run, if any, taking an optional run status.

:py:func:`mlflow.active_run` returns a :py:class:`mlflow.entities.Run` object corresponding to the
currently active run, if any.
**Note**: You cannot access currently-active run attributes
(parameters, metrics, etc.) through the run returned by ``mlflow.active_run``. In order to access
such attributes, use the :py:class:`MlflowClient <mlflow.client.MlflowClient>` as follows:

.. code-block:: python

    client = mlflow.MlflowClient()
    data = client.get_run(mlflow.active_run().info.run_id).data

:py:func:`mlflow.last_active_run` returns a :py:class:`mlflow.entities.Run` object corresponding to the
currently active run, if any. Otherwise, it returns a :py:class:`mlflow.entities.Run` object corresponding
the last run started from the current Python process that reached a terminal status (i.e. FINISHED, FAILED, or KILLED).

:py:func:`mlflow.get_parent_run` returns a :py:class:`mlflow.entities.Run` object corresponding to the
parent run for the given run id, if one exists. Otherwise, it returns None.

:py:func:`mlflow.log_param` logs a single key-value param in the currently active run. The key and
value are both strings. Use :py:func:`mlflow.log_params` to log multiple params at once.

:py:func:`mlflow.log_metric` logs a single key-value metric. The value must always be a number.
MLflow remembers the history of values for each metric. Use :py:func:`mlflow.log_metrics` to log
multiple metrics at once.

:py:func:`mlflow.log_input` logs a single :py:class:`mlflow.data.dataset.Dataset` object corresponding to the currently 
active run. You may also log a dataset context string and a dict of key-value tags.

:py:func:`mlflow.set_tag` sets a single key-value tag in the currently active run. The key and
value are both strings. Use :py:func:`mlflow.set_tags` to set multiple tags at once.

:py:func:`mlflow.log_artifact` logs a local file or directory as an artifact, optionally taking an
``artifact_path`` to place it in within the run's artifact URI. Run artifacts can be organized into
directories, so you can place the artifact in a directory this way.

:py:func:`mlflow.log_artifacts` logs all the files in a given directory as artifacts, again taking
an optional ``artifact_path``.

:py:func:`mlflow.get_artifact_uri` returns the URI that artifacts from the current run should be
logged to.


Tracking Tips
=============

Logging with explicit step and timestamp
----------------------------------------

The ``log`` methods support two alternative methods for distinguishing metric values on the x-axis: ``timestamp`` and ``step`` (number of training iterations, number of epochs, and so on).
By default, ``timestamp`` is set to the current time and ``step`` is set to 0. You can override these values by passing
``timestamp`` and ``step`` arguments to the ``log`` methods. For example:

.. code-block:: python

    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch, timestamp=now)

``step``  has the following requirements and properties:

- Must be a valid 64-bit integer value.
- Can be negative.
- Can be out of order in successive write calls. For example, (1, 3, 2) is a valid sequence.
- Can have "gaps" in the sequence of values specified in successive write calls. For example, (1, 5, 75, -20) is a valid sequence.

If you specify both a timestamp and a step, metrics are recorded against both axes independently.

.. _organizing_runs_in_experiments:

Organizing Runs in Experiments
------------------------------
MLflow allows you to group runs under experiments, which can be useful for comparing runs intended
to tackle a particular task. You can create experiments via multiple way - MLflow UI, the :ref:`cli` (``mlflow experiments``),
or the :py:func:`mlflow.create_experiment` Python API. You can pass the experiment name for an individual run
using the CLI (for example, ``mlflow run ... --experiment-name [name]``) or the ``MLFLOW_EXPERIMENT_NAME``
environment variable. Alternatively, you can use the experiment ID instead, via the
``--experiment-id`` CLI flag or the ``MLFLOW_EXPERIMENT_ID`` environment variable.

.. code-block:: bash

    # Set the experiment via environment variables
    export MLFLOW_EXPERIMENT_NAME=fraud-detection

    mlflow experiments create --experiment-name fraud-detection

.. code-block:: python

    # Launch a run. The experiment is inferred from the MLFLOW_EXPERIMENT_NAME environment
    # variable, or from the --experiment-name parameter passed to the MLflow CLI (the latter
    # taking precedence)
    with mlflow.start_run():
        mlflow.log_param("a", 1)
        mlflow.log_metric("b", 2)

.. _child_runs:

Creating Child Runs
-------------------
You can also create multiple runs inside a single run. This is useful for scenario like hyperparameter tuning,
cross-validation folds, where you need another level of organization within an experiment. You can create child runs
by passing ``parent_run_id`` to :py:func:`mlflow.start_run` function. For example:

.. code-block:: python

        # Start parent run
        with mlflow.start_run() as parent_run:
            param = [0.01, 0.02, 0.03]

            # Create a child run for each parameter setting
            for p in param:
                with mlflow.start_run(nested=True) as child_run:
                    mlflow.log_param("p", p)
                    ...
                    mlflow.log_metric("val_loss", val_loss)

You can fetch all child runs under a parent run using tags.

.. code-block::python

    child_runs = mlflow.search_runs(experiment_ids=[YOUR_EXPERIMENT_ID], filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'")
    print("child runs:")
    print(results[['run_id', 'params.p', 'metrics.val_loss']])
    # child runs:
    #             run_id params.p  metrics.val_loss
    #   0        0c0b...     0.01            0.1234
    #   0        c2a0...     0.02            0.0890
    #   0        g2j1...     0.03            0.1567


.. _launching-multiple-runs:

Launching Multiple Runs in One Program
--------------------------------------
Sometimes you want to launch multiple MLflow runs in the same program: for example, maybe you are
performing a hyperparameter search locally or your experiments are just very fast to run. The way to
do this depends on whether you want to run them :ref:`sequentially <sequential-runs>` or in :ref:`parallel <parallel-runs>`.

.. _sequential-runs:

Sequential Runs
~~~~~~~~~~~~~~~
Running multiple runs one-by-one is easy to do because the ``ActiveRun`` object returned by :py:func:`mlflow.start_run`
is a Python `context manager <https://docs.python.org/2.5/whatsnew/pep-343.html>`_. You can "scope" each run to
just one block of code as follows:

.. code-block:: python

    # First run
    with mlflow.start_run():
        mlflow.log_param("x", 1)
        mlflow.log_metric("y", 2)
        ...

    # Another run
    with mlflow.start_run():
        ...

The run remains open throughout the ``with`` statement, and is automatically closed when the
statement exits, even if it exits due to an exception.

.. _parallel-runs:

Parallel Runs
~~~~~~~~~~~~~
MLflow also supports running multiple runs in parallel using `multiprocessing <https://docs.python.org/2/library/multiprocessing.html>`_ or multi threading.

**Multi-processing** is straightforward: just call :py:func:`mlflow.start_run` in a separate process and it creates a new run for each. For example:

.. code-block:: python

    import mlflow
    import multiprocessing as mp


    def train_model(param):
        with mlflow.start_run():
            mlflow.log_param("p", param)
            ...


    if __name__ == "__main__":
        mlflow.set_experiment("multi-process")
        params = [0.01, 0.02, ...]
        with mp.Pool(processes=4) as pool:
            pool.map(train_model, params)

.. attention::

  The above code will only work if the ``fork`` method is used for creating child processes. If you are using
  the ``spawn`` method, the child processes will not automatically inherit the global state of the parent process, which includes
  the active experiment and tracking URI, resulting in the child processes not being able to log to the same experiment. To
  overcome this limitation, you can set the active experiment and tracking URI explicitly in each child process as shown below.
  The ``fork`` is the default method on POSIX systems except MacOS, but otherwise ``spawn`` method will be used by default.

  .. code-block:: python

    import mlflow
    import multiprocessing as mp


    def train_model(params):
        # Set the experiment and tracking URI in each child process
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("multi-process")
        with mlflow.start_run():
            ...


    if __name__ == "__main__":
        params = [0.01, 0.02, ...]
        pool = mp.get_context("spawn").Pool(processes=4)
        pool.map(train_model, params)


**Multi-threading** is a bit more complicated because MLflow uses a global state to keep track of the currently active run i.e. having multiple active
runs in the same process may cause data corruption. However, you can avoid this issue and use multi threading by using :ref:`Child Runs <child_runs>`.
You can start child runs in each thread by passing ``nested=True`` to :py:func:`mlflow.start_run` as described in the previous section. For example:

.. code-block:: python

        import mlflow
        import threading


        def train_model(param):
            # Create a child run by passing nested=True
            with mlflow.start_run(nested=True):
                mlflow.log_param("p", param)
                ...


        if __name__ == "__main__":
            params = [0.01, 0.02, ...]
            threads = []
            for p in params:
                t = threading.Thread(target=train_model, args=(p,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

Also `here <https://github.com/mlflow/mlflow/blob/master/examples/hyperparam/search_random.py>`_ is the full example of hyperparameter tuning using child runs with multi threading.

.. _add-tags-to-runs:

Adding Tags to Runs
-------------------

You can annotate runs with arbitrary tags to organize them into groups. This allows you to easily filter and search
Runs in Tracking UI by using :ref:`filter expression <mlflow_tags>`.

.. _system_tags:

System Tags
~~~~~~~~~~~

Tag keys that start with ``mlflow.`` are reserved for internal use.
The following tags are set automatically by MLflow, when appropriate:

.. note:: 
    ``mlflow.note.content`` is an exceptional case where the tag is **not set automatically** and can be overridden by the user to include additional information about the run.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Key
     - Description
   * - ``mlflow.note.content``
     - A descriptive note about this run. This reserved tag is **not set automatically** and can be overridden by the user to include additional information about the run. The content is displayed on the run's page under the Notes section.
   * - ``mlflow.parentRunId``
     - The ID of the parent run, if this is a nested run.
   * - ``mlflow.user``
     - Identifier of the user who created the run.
   * - ``mlflow.source.type``
     - Source type. Possible values: ``"NOTEBOOK"``, ``"JOB"``, ``"PROJECT"``, ``"LOCAL"``, and ``"UNKNOWN"``
   * - ``mlflow.source.name``
     - Source identifier (e.g., GitHub URL, local Python filename, name of notebook)
   * - ``mlflow.source.git.commit``
     - Commit hash of the executed code, if in a git repository. This tag is only logged when the code is executed as a Python script like ``python train.py`` or as a project. If the code is executed in a notebook, this tag is not logged.
   * - ``mlflow.source.git.branch``
     - Name of the branch of the executed code, if in a git repository. This tag is only logged within the context of `MLflow Projects <../projects.html>`_ and `MLflow Recipe <../recipes.html>`_.
   * - ``mlflow.source.git.repoURL``
     - URL that the executed code was cloned from. This tag is only logged within the context of `MLflow Projects <../projects.html>`_ and `MLflow Recipe <../recipes.html>`_.
   * - ``mlflow.project.env``
     - The runtime context used by the MLflow project. Possible values: ``"docker"`` and ``"conda"``.
   * - ``mlflow.project.entryPoint``
     - Name of the project entry point associated with the current run, if any.
   * - ``mlflow.docker.image.name``
     - Name of the Docker image used to execute this run.
   * - ``mlflow.docker.image.id``
     - ID of the Docker image used to execute this run.
   * - ``mlflow.log-model.history``
     - Model metadata collected by log-model calls. Includes the serialized form of the MLModel model files logged to a run, although the exact format and information captured is subject to change.


Custom Tags
~~~~~~~~~~~

The :py:func:`MlflowClient.set_tag() <mlflow.client.MlflowClient.set_tag>` function lets you add custom tags to runs. A tag can only have a single unique value mapped to it at a time. For example:

.. code-block:: python

  client.set_tag(run.info.run_id, "tag_key", "tag_value")


Get MLflow Run instance from autologged results
-----------------------------------------------

In some cases, you may want to access the MLflow Run instance associated with the autologged results, similarly to you can get the run context with `mlflow.start_run()` 
You can access the most recent autolog run through the :py:func:`mlflow.last_active_run` function. Here's a short sklearn autolog example that makes use of this function:

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
    autolog_run = mlflow.last_active_run()
    print(autolog_run)
    # <Run:
    #    data=<RunData:
    #        metrics={'accuracy': 0.0},
    #        params={'n_estimators': '100', 'max_depth': '6', 'max_features': '3'},
    #        tags={'estimator_class': 'sklearn.ensemble._forest.RandomForestRegressor', 'estimator_name': 'RandomForestRegressor'}
    #    >,
    #    info=<RunInfo:
    #        artifact_uri='file:///Users/andrew/Code/mlflow/mlruns/0/0c0b.../artifacts',
    #        end_time=163...0,
    #        run_id='0c0b...',
    #        run_uuid='0c0b...',
    #        start_time=163...0,
    #        status='FINISHED',
    #        user_id='ME'>
    #    >
