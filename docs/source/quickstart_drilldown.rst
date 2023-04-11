:orphan:

.. _quickstart_drilldown:

Quickstart additional material: Options and troubleshooting
===========================================================


.. 
    Hmm... At the moment, I have this as a single separate document.
    Other options would be to inline this below the 'next steps' section of the quickstart doc itself. That would probably be a net gain for SEO, at the expense of a longer doc. 

    The other option would be to have each section here as a separate doc, and link to them from the quickstart doc. Would that be a gain for SEO (doc is better than section?)


.. _quickstart_drilldown_install:

Customizing and troubleshooting MLflow installation
---------------------------------------------------

tk
- Java exists
- R exists
- mlflow-skinny
- mlflow[extras]
- Mac default python
- environment managers
- conda channel exists?
tk

.. _quickstart_drilldown_autolog:

Autolog options
---------------

tk

.. _quickstart_drilldown_tracking_api:

Effectively using the MLflow Tracking API
-----------------------------------------

tk

.. _quickstart_drilldown_tracking_ui:

Navigating the MLflow Tracking UI
---------------------------------

tk

- WORKER_TIMEOUT issue 


.. _quickstart_drilldown_tracking_backend:

Choosing and configuring the MLflow tracking backend 
----------------------------------------------------

tk

.. _quickstart_drilldown_log_and_load_model:

Storing and serving MLflow models
---------------------------------

tk
{>> Need to talk about `pyfunc` <<}

Running MLflow Projects
-----------------------

You can easily run existing projects with the ``mlflow run`` command, which runs a project from
either a local directory or a GitHub URI:

.. code-block:: bash

    mlflow run sklearn_elasticnet_wine -P alpha=0.5

    mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0

There's a sample project in ``tutorial``, including a ``MLproject`` file that
specifies its dependencies. if you haven't configured a :ref:`tracking server <tracking_server>`,
projects log their Tracking API data in the local ``mlruns`` directory so you can see these
runs using ``mlflow ui``.

.. note::
    By default ``mlflow run`` installs all dependencies using `virtualenv <https://virtualenv.pypa.io/en/latest//>`_.
    To run a project without using ``virtualenv``, you can provide the ``--env-manager=local`` option to
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
``sklearn_logistic_regression/train.py`` that you can run as follows:

.. code-block:: bash

    python sklearn_logistic_regression/train.py

When you run the example, it outputs an MLflow run ID for that experiment. If you look at
``mlflow ui``, you will also see that the run saved a ``model`` folder containing an ``MLmodel``
description file and a pickled scikit-learn model. You can pass the run ID and the path of the model
within the artifacts directory (here "model") to various tools. For example, MLflow includes a
simple REST server for python-based models:

.. code-block:: bash

    mlflow models serve -m runs:/<RUN_ID>/model

.. note::

    By default the server runs on port 5000. If that port is already in use, use the `--port` option to
    specify a different port. For example: ``mlflow models serve -m runs:/<RUN_ID>/model --port 1234``

Once you have started the server, you can pass it some sample data and see the
predictions.

The following example uses ``curl`` to send a JSON-serialized pandas DataFrame with the ``split``
orientation to the model server. For more information about the input data formats accepted by
the pyfunc model server, see the :ref:`MLflow deployment tools documentation <local_model_deployment>`.

.. code-block:: bash

    curl -d '{"dataframe_split": {"columns": ["x"], "data": [[1], [-1]]}}' -H 'Content-Type: application/json' -X POST localhost:5000/invocations

which returns::

    [1, 0]

For more information, see :doc:`models`.


