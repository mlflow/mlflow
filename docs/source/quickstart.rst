.. _quickstart:

Quickstart
==========

Installing MLflow
-----------------

You install MLflow by running:

.. code:: bash

    pip install mlflow

.. note::

    You cannot install MLflow on the MacOS system installation of Python. We recommend installing
    Python 3 through the `Homebrew <https://brew.sh/>`_ package manager using
    ``brew install python``. (In this case, installing MLflow is now ``pip3 install mlflow``).

At this point we recommend you follow the :doc:`tutorial` for a walk-through on how you
can use MLflow in your daily workflow.

Using the Tracking API
----------------------

The :doc:`MLflow Tracking API<tracking/>` lets you log metrics and artifacts (files) from your data
science code and see a history of your runs. You can try it out by writing a simple Python script
as follows (this example is also included in ``example/quickstart/test.py``):

.. code:: python

    import os
    from mlflow import log_metric, log_param, log_artifact

    if __name__ == "__main__":
        # Log a parameter (key-value pair)
        log_param("param1", 5)

        # Log a metric; metrics can be updated throughout the run
        log_metric("foo", 1)
        log_metric("foo", 2)
        log_metric("foo", 3)

        # Log an artifact (output file)
        with open("output.txt", "w") as f:
            f.write("Hello world!")
        log_artifact("output.txt")

Viewing the Tracking UI
-----------------------

By default, wherever you run your program, the tracking API writes data into files into an ``mlruns`` directory.
You can then run MLflow's Tracking UI:

.. code:: bash

    mlflow ui

and view it at `<http://localhost:5000>`_. 

Alternatively, you can configure MLflow to :ref:`log runs to a remote server<tracking>` to manage
your results centrally or share them across a team.

Running MLflow Projects
-----------------------

MLflow allows you to package code and its dependencies as a *project* that can be run in a
reproducible fashion on other data. Each project includes its code and a ``MLproject`` file that
defines its dependencies (for example, Python environment) as well as what commands can be run into the
project and what arguments they take.

You can easily run existing projects with the ``mlflow run`` command, which runs a project from
either a local directory or a GitHub URI:

.. code:: bash

    mlflow run example/tutorial -P alpha=0.5

    mlflow run git@github.com:databricks/mlflow-example.git -P alpha=5

There's a sample project in ``example/tutorial``, including a ``MLproject`` file that
specifies its dependencies. All projects that run also log their Tracking API data in the local
``mlruns`` directory (or on your tracking server if you've configured one), so you should be able
to see these runs using ``mlflow ui``.

.. note::
    By default ``mlflow run`` installs all dependencies using `conda <https://conda.io/>`_.
    To run a project without using ``conda``, you can provide the ``--no-conda`` option to
    ``mlflow run``. In this case, you must ensure that the necessary dependencies are already installed
    in your Python environment.

For more information, see :doc:`projects`.

Saving and Serving Models
-------------------------

MLflow includes a generic ``MLmodel`` format for saving *models* from a variety of tools in diverse
*flavors*. For example, many models can be served as Python functions, so an ``MLmodel`` file can
declare how each model should be interpreted as a Python function in order to let various tools
serve it. MLflow also includes tools for running such models locally and exporting them to Docker
containers or commercial serving platforms.

To illustrate this functionality, the ``mlflow.sklearn`` package can log scikit-learn models as
MLflow artifacts and then load them again for serving. There is an example training application in
``example/quickstart/test_sklearn.py`` that you can run as follows:

.. code:: bash

    python example/quickstart/test_sklearn.py

When you run the example, it outputs an MLflow run ID for that experiment. If you look at
``mlflow ui``, you will also see that the run saved a ``model`` folder containing an ``MLmodel``
description file and a pickled scikit-learn model. You can pass the run ID and the path of the model
within the artifacts directory (here "model") to various tools. For example, MLflow includes a
simple REST server for scikit-learn models:

.. code:: bash

    mlflow sklearn serve -r <RUN_ID> model

.. note::

    By default the server runs on port 5000. If that port is already in use, use the `--port` option to
    specify a different port. For example: ``mlflow sklearn serve --port 1234 -r <RUN_ID> model``

Once you have started the server, you can pass it some sample data with ``curl`` and see the
predictions:

.. code:: bash

    curl -d '[{"x": 1}, {"x": -1}]' -H 'Content-Type: application/json' -X POST localhost:5000/invocations
         
which returns::
  
    {"predictions": [1, 0]}

.. note::

    The ``example/quickstart/test_sklearn.py`` script must be run with the same Python version as
    the version of Python that runs ``mlflow sklearn serve``. If they are not the same version,
    the stacktrace below may appear::

        File "/usr/local/lib/python3.6/site-packages/mlflow/sklearn.py", line 54, in _load_model_from_local_file
        return pickle.load(f)
        UnicodeDecodeError: 'ascii' codec can't decode byte 0xc6 in position 0: ordinal not in range(128)


For more information, see :doc:`models`.
