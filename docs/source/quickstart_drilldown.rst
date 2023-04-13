:orphan:

.. _quickstart_drilldown:

Quickstart additional material: Options and troubleshooting
============================================================

..
    Eventually, these H2s will probably all be separate articles. For now, I'm 
    avoiding that so as not to create a bunch of super-skinny pages. 

.. _quickstart_drilldown_install:

Customizing and troubleshooting MLflow installation
---------------------------------------------------

Python library options
**********************

Rather the default MLflow library, you can install the following optional libraries:

.. list-table::
   :widths: 10 20 70
   :header-rows: 1

   * - Name
     - ``pip install`` command
     - Description
   * - mlflow-skinny
     - ``pip install mlflow-skinny``
     - Lightweight MLflow package without SQL storage, server, UI, or data science dependencies.
   * - mlflow[extras]
     - ``pip install mlflow[extras]``
     - MLflow package with all dependencies needed to run various MLflow flavors. These dependencies are listed in doc:`https://github.com/mlflow/mlflow/blob/master/requirements/extra-ml-requirements.txt`



Python and Mac OS X
**********************

We strongly recommend using a virtual environment manager on Macs. We always recommend using virtual environments, but they are especially important on Mac OS X because the system ``python`` version varies depending on the installation and whether you've installed the Xcode command line tools.

Virtual Environment managers
****************************

tk If there is no conda channel, we should probably just remove this section.



R and Java
**********

See :ref:`installing MLflow for R<R-api>` . Since the Java ecosystem supports many routes towards installing libraries, we don't have a single installation command for Java. See :ref:`java_api` for more information.

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

    {>> TODO: Add --env-manager local and validate <<}

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


